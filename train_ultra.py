"""
Training script for DocCornerNet V2 — cross-platform (CUDA/MPS/CPU).

Usage:
    python -m v2.train_ultra \
        --data_root dataset/DocCornerDataset_small \
        --output_dir runs/v2_smoke \
        --epochs 5 --batch_size 16
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from tqdm import tqdm

from model import create_model
from losses import DocCornerNetV2Trainer
from dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_split_file,
    load_label_yolo_obb,
    augment_sample,
    normalize_image,
    DEFAULT_AUG_CONFIG,
)
from metrics import ValidationMetrics


# --------------------------------------------------------------------------
# Platform setup
# --------------------------------------------------------------------------

def setup_platform():
    """Configure platform for maximum performance."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            from tensorflow.python.client import device_lib
            devices = device_lib.list_local_devices()
            for d in devices:
                if "GPU" in d.device_type:
                    desc = d.physical_device_desc.lower()
                    if "nvidia" in desc or "cuda" in desc:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        print(f"[platform] NVIDIA GPU detected: {d.physical_device_desc}")
                        return "cuda"
        except Exception:
            pass

    if sys.platform == "darwin":
        print("[platform] Using Metal Performance Shaders (MPS)")
        return "mps"

    cpu_count = os.cpu_count() or 4
    print(f"[platform] Using CPU with {cpu_count} threads")
    return "cpu"


# --------------------------------------------------------------------------
# Fast threaded data loading
# --------------------------------------------------------------------------

def _load_single_image(args):
    """Thread-safe image loader."""
    name, data_root, img_size = args
    data_root = Path(data_root)
    image_dir = data_root / "images"
    negative_dir = data_root / "images-negative"
    label_dir = data_root / "labels"

    coords = np.zeros(8, dtype=np.float32)
    has_doc = 0.0

    if name.startswith("negative_"):
        img_path = negative_dir / name
    else:
        img_path = image_dir / name
        label_path = label_dir / f"{Path(name).stem}.txt"
        if label_path.exists():
            try:
                coords = load_label_yolo_obb(str(label_path))
                if coords.sum() > 0:
                    has_doc = 1.0
            except Exception:
                pass

    if not img_path.exists():
        return None

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize((img_size, img_size), Image.BILINEAR)
            img_array = np.asarray(img, dtype=np.uint8).copy()
        return (img_array, coords, has_doc)
    except Exception:
        return None


def load_dataset_parquet(data_root, split, img_size):
    """Load dataset from parquet files (HuggingFace format)."""
    import io
    import pyarrow.parquet as pq

    split_dir = Path(data_root) / split
    parquet_files = sorted(split_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {split_dir}")

    table = pq.read_table(split_dir)
    n_images = table.num_rows
    print(f"Loading {split}: {n_images} images from {len(parquet_files)} parquet file(s)", flush=True)

    images = np.empty((n_images, img_size, img_size, 3), dtype=np.uint8)
    coords = np.empty((n_images, 8), dtype=np.float32)
    has_doc = np.empty(n_images, dtype=np.float32)

    img_col = table.column("image")
    is_neg_col = table.column("is_negative")
    coord_cols = ["corner_tl_x", "corner_tl_y", "corner_tr_x", "corner_tr_y",
                  "corner_br_x", "corner_br_y", "corner_bl_x", "corner_bl_y"]

    n_valid = 0
    for i in tqdm(range(n_images), desc=f"  Loading {split}", ncols=80, leave=True):
        try:
            img_struct = img_col[i].as_py()
            img_bytes = img_struct["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((img_size, img_size), Image.BILINEAR)
            images[n_valid] = np.asarray(img, dtype=np.uint8)

            is_neg = is_neg_col[i].as_py()
            if is_neg:
                coords[n_valid] = 0.0
                has_doc[n_valid] = 0.0
            else:
                c = np.array([table.column(cn)[i].as_py() for cn in coord_cols], dtype=np.float32)
                coords[n_valid] = c
                has_doc[n_valid] = 1.0 if c.sum() > 0 else 0.0

            n_valid += 1
        except Exception:
            continue

    images = images[:n_valid]
    coords = coords[:n_valid]
    has_doc = has_doc[:n_valid]
    print(f"  {n_valid}/{n_images} valid ({int((has_doc == 1).sum())} pos, {int((has_doc == 0).sum())} neg)", flush=True)
    return images, coords, has_doc


def load_dataset_fast(data_root, split, img_size, num_workers=32):
    """Load dataset using threading (file-based) or parquet."""
    data_root = Path(data_root)

    # Check for parquet format first
    split_dir = data_root / split
    if split_dir.is_dir() and any(split_dir.glob("*.parquet")):
        return load_dataset_parquet(data_root, split, img_size)

    # Fall back to file-based format
    split_file = data_root / f"{split}.txt"
    if not split_file.exists():
        for suffix in ["_with_negative_v2", "_with_negative"]:
            candidate = data_root / f"{split}{suffix}.txt"
            if candidate.exists():
                split_file = candidate
                break
    if not split_file.exists():
        raise FileNotFoundError(f"No split file for {split} in {data_root}")

    image_names = load_split_file(str(split_file))
    n_images = len(image_names)
    print(f"Loading {split}: {n_images} images from {split_file.name}", flush=True)

    args_list = [(name, str(data_root), img_size) for name in image_names]
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(_load_single_image, args_list),
                           total=n_images, desc=f"  Loading {split}",
                           ncols=80, leave=True):
            if result is not None:
                results.append(result)

    n_valid = len(results)
    images = np.empty((n_valid, img_size, img_size, 3), dtype=np.uint8)
    coords = np.empty((n_valid, 8), dtype=np.float32)
    has_doc = np.empty(n_valid, dtype=np.float32)

    for i, (img, c, h) in enumerate(results):
        images[i] = img
        coords[i] = c
        has_doc[i] = h

    del results
    gc.collect()
    print(f"  {n_valid}/{n_images} valid ({int((has_doc == 1).sum())} pos, {int((has_doc == 0).sum())} neg)", flush=True)
    return images, coords, has_doc


# --------------------------------------------------------------------------
# tf.data pipeline builder
# --------------------------------------------------------------------------

def make_tf_dataset(
    images, coords, has_doc,
    batch_size, shuffle, augment, image_norm="imagenet",
    drop_remainder=False,
):
    """Build tf.data pipeline from numpy arrays."""
    # Normalize images to float32
    imgs_f = images.astype(np.float32)
    if image_norm == "imagenet":
        imgs_f = imgs_f / 255.0
        imgs_f = (imgs_f - IMAGENET_MEAN) / IMAGENET_STD
    elif image_norm == "zero_one":
        imgs_f = imgs_f / 255.0
    # else: raw255

    ds = tf.data.Dataset.from_tensor_slices((
        imgs_f,
        {"coords": coords, "has_doc": has_doc},
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(images), 10000))
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# --------------------------------------------------------------------------
# Cosine annealing LR schedule
# --------------------------------------------------------------------------

class WarmupCosineSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Cosine annealing with linear warmup."""

    def __init__(self, base_lr, total_steps, warmup_steps=0, min_lr=1e-6):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        if self.warmup_steps > 0:
            warmup_lr = self.base_lr * (step / float(self.warmup_steps))
        else:
            warmup_lr = self.base_lr

        progress = (step - float(self.warmup_steps)) / float(max(1, self.total_steps - self.warmup_steps))
        progress = tf.minimum(tf.maximum(progress, 0.0), 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + tf.cos(math.pi * progress))

        return tf.where(step < float(self.warmup_steps), warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }


# --------------------------------------------------------------------------
# Argument parser
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DocCornerNet V2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./runs/v2", help="Output directory")
    parser.add_argument("--input_norm", type=str, default="imagenet",
                        choices=["imagenet", "zero_one", "raw255"])
    parser.add_argument("--num_workers", type=int, default=32)

    # Model
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--fpn_ch", type=int, default=32)
    parser.add_argument("--simcc_ch", type=int, default=96)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_bins", type=int, default=224)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--simcc_kernel_size", type=int, default=5)
    parser.add_argument("--backbone_weights", type=str, default="imagenet")

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    # Loss
    parser.add_argument("--sigma_px", type=float, default=2.0)
    parser.add_argument("--loss_tau", type=float, default=0.5)
    parser.add_argument("--w_simcc", type=float, default=1.0)
    parser.add_argument("--w_coord", type=float, default=0.2)
    parser.add_argument("--w_score", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    # Init
    parser.add_argument("--init_weights", type=str, default=None,
                        help="Path to initial weights for warm start")

    return parser.parse_args()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    platform = setup_platform()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["platform"] = platform
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("DocCornerNet V2 Training")
    print("=" * 60)

    # Load data
    print("\n--- Data Loading ---")
    train_images, train_coords, train_has_doc = load_dataset_fast(
        args.data_root, "train", args.img_size, args.num_workers,
    )
    val_images, val_coords, val_has_doc = load_dataset_fast(
        args.data_root, "val", args.img_size, args.num_workers,
    )

    n_train = len(train_images)
    n_val = len(val_images)

    train_ds = make_tf_dataset(
        train_images, train_coords, train_has_doc,
        args.batch_size, shuffle=True, augment=True, image_norm=args.input_norm,
        drop_remainder=True,
    )
    val_ds = make_tf_dataset(
        val_images, val_coords, val_has_doc,
        args.batch_size, shuffle=False, augment=False, image_norm=args.input_norm,
    )

    train_steps = n_train // args.batch_size
    val_steps = math.ceil(n_val / args.batch_size)

    # Build model
    bw = args.backbone_weights
    if bw and bw.strip().lower() in ("none", "null"):
        bw = None
    net = create_model(
        alpha=args.alpha,
        fpn_ch=args.fpn_ch,
        simcc_ch=args.simcc_ch,
        img_size=args.img_size,
        num_bins=args.num_bins,
        tau=args.tau,
        backbone_weights=bw,
        simcc_kernel_size=args.simcc_kernel_size,
    )
    print(f"Model parameters: {net.count_params():,}")

    # Optionally load init weights
    if args.init_weights:
        net.load_weights(args.init_weights)
        print(f"Loaded init weights from {args.init_weights}")

    # Build trainer
    trainer = DocCornerNetV2Trainer(
        net,
        bins=args.num_bins,
        sigma_px=args.sigma_px,
        tau=args.loss_tau,
        w_simcc=args.w_simcc,
        w_coord=args.w_coord,
        w_score=args.w_score,
        label_smoothing=args.label_smoothing,
    )

    # Optimizer with cosine schedule
    steps_per_epoch = len(train_images) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    lr_schedule = WarmupCosineSchedule(
        base_lr=args.learning_rate,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=args.weight_decay,
    )

    trainer.compile(optimizer=optimizer)

    # Print config summary
    print(f"\n--- Configuration ---")
    print(f"  Model:    alpha={args.alpha} fpn_ch={args.fpn_ch} simcc_ch={args.simcc_ch}")
    print(f"  Input:    {args.img_size}x{args.img_size} bins={args.num_bins} norm={args.input_norm}")
    print(f"  Training: epochs={args.epochs} batch={args.batch_size} lr={args.learning_rate}")
    print(f"  Loss:     sigma_px={args.sigma_px} loss_tau={args.loss_tau} "
          f"w_simcc={args.w_simcc} w_coord={args.w_coord} w_score={args.w_score}")
    print(f"  Schedule: warmup={args.warmup_epochs}ep cosine wd={args.weight_decay}")
    print(f"  Steps:    {train_steps}/ep train, {val_steps}/ep val")

    # Training loop
    print(f"\n--- Training ---")
    val_metrics = ValidationMetrics(img_size=args.img_size)
    best_iou = 0.0
    best_val_loss = float("inf")
    prev_val_loss = None
    prev_iou = None
    prev_err = None
    prev_lr = None
    log_rows = []
    total_t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        trainer.reset_metrics()
        train_pbar = tqdm(train_ds, total=train_steps,
                          desc=f"  Epoch {epoch}/{args.epochs} [Train]",
                          leave=False, ncols=100)
        for batch in train_pbar:
            trainer.train_step(batch)
            train_pbar.set_postfix(loss=f"{float(trainer.loss_tracker.result()):.4f}")
        train_pbar.close()
        train_m = {k: float(v) for k, v in trainer._get_metrics_dict().items()}

        # Validate
        trainer.reset_metrics()
        val_metrics.reset()
        val_pbar = tqdm(val_ds, total=val_steps,
                        desc=f"  Epoch {epoch}/{args.epochs} [Val]  ",
                        leave=False, ncols=100)
        for images, targets in val_pbar:
            trainer.test_step((images, targets))
            outputs = net(images, training=False)
            coords_pred = outputs["coords"].numpy()
            score_logit = outputs["score_logit"].numpy()
            score_pred = 1.0 / (1.0 + np.exp(-np.clip(score_logit, -60.0, 60.0)))
            val_metrics.update(
                coords_pred, targets["coords"].numpy(),
                score_pred, targets["has_doc"].numpy(),
            )
            val_pbar.set_postfix(loss=f"{float(trainer.loss_tracker.result()):.4f}")
        val_pbar.close()
        val_m = {k: float(v) for k, v in trainer._get_metrics_dict().items()}
        detailed = val_metrics.compute()

        epoch_time = time.time() - t0
        iou = detailed["mean_iou"]
        val_loss = val_m["loss"]
        val_err = detailed["corner_error_px"]
        lr_val = optimizer.learning_rate
        if callable(lr_val):
            lr_now = float(lr_val(optimizer.iterations))
        else:
            lr_now = float(lr_val)

        # Track bests
        is_best_iou = iou > best_iou
        if is_best_iou:
            best_iou = iou
            net.save_weights(str(output_dir / "best_model.weights.h5"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # ETA
        elapsed = time.time() - total_t0
        avg_epoch = elapsed / epoch
        remaining = avg_epoch * (args.epochs - epoch)
        if remaining >= 3600:
            eta_str = f"{remaining/3600:.1f}h"
        elif remaining >= 60:
            eta_str = f"{remaining/60:.0f}m"
        else:
            eta_str = f"{remaining:.0f}s"

        # Delta helpers
        def _delta_str(curr, prev, lower_is_better=True):
            if prev is None:
                return ""
            d = curr - prev
            arrow = "▼" if d < 0 else "▲" if d > 0 else "="
            # For lower-is-better: ▼ is good. For higher-is-better: ▲ is good.
            sign = "+" if d > 0 else ""
            return f" {arrow} {sign}{d:.4f}"

        def _delta_str_px(curr, prev):
            if prev is None:
                return ""
            d = curr - prev
            arrow = "▼" if d < 0 else "▲" if d > 0 else "="
            sign = "+" if d > 0 else ""
            return f" {arrow} {sign}{d:.2f}px"

        def _lr_delta(curr, prev):
            if prev is None or abs(curr - prev) < 1e-12:
                return ""
            arrow = "▼" if curr < prev else "▲"
            return f" {arrow} (was {prev:.2e})"

        # Loss gap (overfitting indicator)
        gap = val_loss - train_m["loss"]
        gap_warn = ""
        if len(log_rows) >= 2:
            prev_gap = log_rows[-1]["val_loss"] - log_rows[-1]["train_loss"]
            if gap > prev_gap + 0.1:
                gap_warn = " (growing)"

        # Print epoch summary
        sep = "-" * 70
        print(sep)
        print(f"  Epoch {epoch:3d}/{args.epochs}"
              f"{'':>40}{epoch_time:.1f}s  ETA {eta_str}")
        print(f"  |- Loss       train={train_m['loss']:.4f}"
              f"       val={val_loss:.4f}"
              f"{_delta_str(val_loss, prev_val_loss)}"
              f"  gap={gap:.2f}{gap_warn}")
        iou_line = (f"  |- IoU        train={train_m['iou']:.4f}"
                    f"       val={iou:.4f}"
                    f"{_delta_str(iou, prev_iou, lower_is_better=False)}"
                    f"  best={best_iou:.4f}")
        if is_best_iou:
            iou_line += " ★ NEW BEST"
        print(iou_line)
        print(f"  |- Error      train={train_m['corner_err_px']:.2f}px"
              f"      val={val_err:.2f}px"
              f"{_delta_str_px(val_err, prev_err)}"
              f"  p95={detailed['corner_error_p95_px']:.2f}px")
        print(f"  |- Recall     @90={detailed['recall_90']*100:.0f}%"
              f"  @95={detailed['recall_95']*100:.0f}%"
              f"  @99={detailed['recall_99']*100:.0f}%")
        print(f"  |- Score      cls_f1={detailed['cls_f1']*100:.0f}%"
              f"  acc={detailed['cls_accuracy']*100:.0f}%")
        print(f"  '- LR         {lr_now:.2e}{_lr_delta(lr_now, prev_lr)}")

        # Update prev values for next epoch
        prev_val_loss = val_loss
        prev_iou = iou
        prev_err = val_err
        prev_lr = lr_now

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_iou": train_m["iou"],
            "train_corner_err_px": train_m["corner_err_px"],
            "val_loss": val_loss,
            "val_mean_iou": iou,
            "val_median_iou": detailed["median_iou"],
            "val_corner_error_px": val_err,
            "val_corner_error_p95_px": detailed["corner_error_p95_px"],
            "val_recall_90": detailed["recall_90"],
            "val_recall_95": detailed["recall_95"],
            "val_recall_99": detailed["recall_99"],
            "val_cls_f1": detailed["cls_f1"],
            "val_cls_accuracy": detailed["cls_accuracy"],
            "lr": lr_now,
            "loss_gap": gap,
            "epoch_time": epoch_time,
        })

    # Save final
    total_time = time.time() - total_t0
    net.save_weights(str(output_dir / "final_model.weights.h5"))

    # Save training log
    import csv
    with open(output_dir / "training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    # Final summary
    best_row = max(log_rows, key=lambda r: r["val_mean_iou"])
    print("=" * 70)
    print(f"  TRAINING COMPLETE")
    print(f"  |- Time        {total_time/60:.1f} min ({args.epochs} epochs)")
    print(f"  |- Best epoch  {best_row['epoch']}")
    print(f"  |- Best IoU    {best_row['val_mean_iou']:.4f} (median {best_row['val_median_iou']:.4f})")
    print(f"  |- Error       {best_row['val_corner_error_px']:.2f}px (p95 {best_row['val_corner_error_p95_px']:.2f}px)")
    print(f"  |- Recall      @90={best_row['val_recall_90']*100:.1f}%  @95={best_row['val_recall_95']*100:.1f}%  @99={best_row['val_recall_99']*100:.1f}%")
    print(f"  |- Score       cls_f1={best_row['val_cls_f1']*100:.1f}%  acc={best_row['val_cls_accuracy']*100:.1f}%")
    print(f"  '- Artifacts   {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
