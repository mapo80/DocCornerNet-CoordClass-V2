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
import re
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
    tf_augment_batch,
    tf_augment_color_only,
)
from metrics import ValidationMetrics

_SOURCE_SUFFIX_RE = re.compile(r"_(\d+)\.[^.]+$")
_HARD_SELECTOR_PREFIXES = {"source", "sample"}
_WEIGHT_RULE_PREFIXES = {"source", "sample"}


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


def _decode_parquet_row(args):
    """Thread-safe: decode a single parquet row (JPEG bytes → resized numpy)."""
    import io
    img_bytes, is_neg, coord_values, img_size, filename = args
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((img_size, img_size), Image.BILINEAR)
        img_array = np.asarray(img, dtype=np.uint8).copy()
        if is_neg:
            coords = np.zeros(8, dtype=np.float32)
            has_doc = 0.0
        else:
            coords = np.array(coord_values, dtype=np.float32)
            has_doc = 1.0 if coords.sum() > 0 else 0.0
        return (img_array, coords, has_doc, filename)
    except Exception:
        return None


def load_dataset_parquet(data_root, split, img_size, num_workers=16, return_names=False):
    """Load dataset from parquet files (HuggingFace format) with threaded decoding."""
    import pyarrow.parquet as pq

    split_dir = Path(data_root) / split
    parquet_files = sorted(split_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {split_dir}")

    table = pq.read_table(split_dir)
    n_images = table.num_rows
    print(f"Loading {split}: {n_images} images from {len(parquet_files)} parquet file(s)", flush=True)

    # Pre-extract all data from pyarrow (fast, single-thread)
    img_col = table.column("image")
    filename_col = table.column("filename")
    is_neg_col = table.column("is_negative")
    coord_cols = ["corner_tl_x", "corner_tl_y", "corner_tr_x", "corner_tr_y",
                  "corner_br_x", "corner_br_y", "corner_bl_x", "corner_bl_y"]

    args_list = []
    for i in range(n_images):
        img_bytes = img_col[i].as_py()["bytes"]
        filename = filename_col[i].as_py()
        is_neg = is_neg_col[i].as_py()
        coord_values = [table.column(cn)[i].as_py() for cn in coord_cols]
        args_list.append((img_bytes, is_neg, coord_values, img_size, filename))

    # Parallel JPEG decode + resize
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(_decode_parquet_row, args_list),
                           total=n_images, desc=f"  Decoding {split}",
                           ncols=80, leave=True):
            if result is not None:
                results.append(result)

    n_valid = len(results)
    images = np.empty((n_valid, img_size, img_size, 3), dtype=np.uint8)
    coords = np.empty((n_valid, 8), dtype=np.float32)
    has_doc = np.empty(n_valid, dtype=np.float32)
    names = [] if return_names else None

    for i, (img, c, h, name) in enumerate(results):
        images[i] = img
        coords[i] = c
        has_doc[i] = h
        if return_names:
            names.append(name)

    del results
    gc.collect()
    print(f"  {n_valid}/{n_images} valid ({int((has_doc == 1).sum())} pos, {int((has_doc == 0).sum())} neg)", flush=True)
    if return_names:
        return images, coords, has_doc, np.asarray(names, dtype=object)
    return images, coords, has_doc


def load_dataset_fast(data_root, split, img_size, num_workers=32, return_names=False):
    """Load dataset using threading (file-based) or parquet."""
    data_root = Path(data_root)

    # Check for parquet format first
    split_dir = data_root / split
    if split_dir.is_dir() and any(split_dir.glob("*.parquet")):
        return load_dataset_parquet(
            data_root, split, img_size, num_workers=num_workers, return_names=return_names,
        )

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
    valid_names = [] if return_names else None
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for name, result in zip(
            image_names,
            tqdm(executor.map(_load_single_image, args_list),
                           total=n_images, desc=f"  Loading {split}",
                           ncols=80, leave=True),
        ):
            if result is not None:
                results.append(result)
                if return_names:
                    valid_names.append(name)

    n_valid = len(results)
    images = np.empty((n_valid, img_size, img_size, 3), dtype=np.uint8)
    coords = np.empty((n_valid, 8), dtype=np.float32)
    has_doc = np.empty(n_valid, dtype=np.float32)
    names = [] if return_names else None

    for i, result in enumerate(results):
        img, c, h = result
        images[i] = img
        coords[i] = c
        has_doc[i] = h
        if return_names:
            names.append(valid_names[i])

    del results
    gc.collect()
    print(f"  {n_valid}/{n_images} valid ({int((has_doc == 1).sum())} pos, {int((has_doc == 0).sum())} neg)", flush=True)
    if return_names:
        return images, coords, has_doc, np.asarray(names, dtype=object)
    return images, coords, has_doc


def extract_source_name(filename):
    """Collapse numbered filenames into a source/family identifier."""
    base = Path(filename).name
    return _SOURCE_SUFFIX_RE.sub("", base)


def load_hard_selector_file(path):
    """Load hard-sample selectors from a plain-text file.

    Supported line formats:
    - bare token: treated as `source:<token>`
    - `source:<source_name>`
    - `sample:<filename>`
    Blank lines and lines starting with `#` are ignored.
    """
    selector_path = Path(path)
    if not selector_path.exists():
        raise ValueError(f"Hard selector file not found: {selector_path}")

    selectors = {"sources": set(), "samples": set()}
    with selector_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            kind = "source"
            value = line
            if ":" in line:
                prefix, suffix = line.split(":", 1)
                kind = prefix.strip().lower()
                value = suffix.strip()
                if kind not in _HARD_SELECTOR_PREFIXES:
                    raise ValueError(
                        f"Unsupported selector prefix '{prefix}' in {selector_path}:{line_no}. "
                        "Use 'source:' or 'sample:'."
                    )

            if not value:
                raise ValueError(f"Empty hard selector in {selector_path}:{line_no}")

            if kind == "source":
                selectors["sources"].add(value)
            else:
                selectors["samples"].add(Path(value).name)

    return selectors


def load_selector_weight_file(path):
    """Load source/sample weighting rules from a plain-text file."""
    weight_path = Path(path)
    if not weight_path.exists():
        raise ValueError(f"Selector weight file not found: {weight_path}")

    rules = {"sources": {}, "samples": {}, "negative": None}
    with weight_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(
                    f"Malformed selector weight in {weight_path}:{line_no}. "
                    "Expected '<selector>=<weight>'."
                )

            selector_text, weight_text = line.split("=", 1)
            selector_text = selector_text.strip()
            weight_text = weight_text.strip()
            if not selector_text or not weight_text:
                raise ValueError(f"Malformed selector weight in {weight_path}:{line_no}")

            try:
                weight = float(weight_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid weight '{weight_text}' in {weight_path}:{line_no}"
                ) from exc
            if weight <= 0.0:
                raise ValueError(f"Selector weights must be > 0 in {weight_path}:{line_no}")

            lowered = selector_text.lower()
            if lowered == "negative":
                rules["negative"] = weight
                continue

            kind = "source"
            value = selector_text
            if ":" in selector_text:
                prefix, suffix = selector_text.split(":", 1)
                kind = prefix.strip().lower()
                value = suffix.strip()
                if kind not in _WEIGHT_RULE_PREFIXES:
                    raise ValueError(
                        f"Unsupported weight selector prefix '{prefix}' in {weight_path}:{line_no}. "
                        "Use 'source:' or 'sample:'."
                    )

            if not value:
                raise ValueError(f"Empty weight selector in {weight_path}:{line_no}")

            if kind == "source":
                rules["sources"][value] = weight
            else:
                rules["samples"][Path(value).name] = weight

    return rules


def build_hard_selector_mask(sample_names, has_doc, selectors):
    """Mark positive samples matched by the configured hard selectors."""
    if sample_names is None:
        raise ValueError("sample_names are required for hard selector masking")
    if not selectors:
        return np.zeros(len(sample_names), dtype=bool)

    source_names = selectors.get("sources", set())
    sample_names_exact = selectors.get("samples", set())
    if not source_names and not sample_names_exact:
        return np.zeros(len(sample_names), dtype=bool)

    match = np.zeros(len(sample_names), dtype=bool)
    normalized_names = np.asarray([Path(name).name for name in sample_names], dtype=object)

    if source_names:
        sources = np.asarray([extract_source_name(name) for name in sample_names], dtype=object)
        match |= np.isin(sources, list(source_names))

    if sample_names_exact:
        match |= np.isin(normalized_names, list(sample_names_exact))

    return (has_doc.astype(np.float32) > 0.5) & match


def build_selector_sample_weights(
    sample_names,
    has_doc,
    weight_rules=None,
    source_balance_power=0.0,
    source_balance_cap=4.0,
):
    """Build normalized per-sample weights from source-frequency and file rules."""
    if sample_names is None:
        raise ValueError("sample_names are required for selector-based sample weights")
    if source_balance_power < 0.0:
        raise ValueError("source_balance_power must be >= 0")
    if source_balance_cap <= 0.0:
        raise ValueError("source_balance_cap must be > 0")

    sample_names = np.asarray(sample_names, dtype=object)
    has_doc = np.asarray(has_doc, dtype=np.float32)
    pos_mask = has_doc > 0.5
    weights = np.ones(len(sample_names), dtype=np.float32)
    normalized_names = np.asarray([Path(name).name for name in sample_names], dtype=object)
    source_names = np.asarray([extract_source_name(name) for name in sample_names], dtype=object)

    if source_balance_power > 0.0 and np.any(pos_mask):
        pos_sources = source_names[pos_mask]
        unique_sources, counts = np.unique(pos_sources, return_counts=True)
        max_count = float(np.max(counts))
        for source_name, count in zip(unique_sources, counts):
            raw_weight = (max_count / float(count)) ** source_balance_power
            clipped_weight = min(raw_weight, source_balance_cap)
            weights[pos_mask & (source_names == source_name)] *= clipped_weight

    if weight_rules:
        for source_name, multiplier in weight_rules.get("sources", {}).items():
            weights[source_names == source_name] *= multiplier
        for sample_name, multiplier in weight_rules.get("samples", {}).items():
            weights[normalized_names == sample_name] *= multiplier
        negative_weight = weight_rules.get("negative")
        if negative_weight is not None:
            weights[~pos_mask] *= float(negative_weight)

    mean_weight = float(np.mean(weights))
    if mean_weight <= 0.0:
        raise ValueError("selector sample weights collapsed to a non-positive mean")
    weights /= mean_weight
    return weights.astype(np.float32)


def build_source_sampling_plan(sample_names, has_doc, sample_weights):
    """Aggregate per-sample weights into source-level dataset sampling masses."""
    if sample_names is None:
        raise ValueError("sample_names are required for source sampling")

    sample_names = np.asarray(sample_names, dtype=object)
    has_doc = np.asarray(has_doc, dtype=np.float32)
    sample_weights = np.asarray(sample_weights, dtype=np.float32)
    pos_mask = has_doc > 0.5
    neg_mask = ~pos_mask
    source_names = np.asarray([extract_source_name(name) for name in sample_names], dtype=object)

    plan = {"negative_mass": float(np.sum(sample_weights[neg_mask])), "sources": {}}
    for source_name in np.unique(source_names[pos_mask]):
        mask = pos_mask & (source_names == source_name)
        mass = float(np.sum(sample_weights[mask]))
        if mass > 0.0:
            plan["sources"][source_name] = mass

    total_mass = plan["negative_mass"] + sum(plan["sources"].values())
    if total_mass <= 0.0:
        raise ValueError("source sampling plan has zero total mass")
    plan["negative_mass"] /= total_mass
    for source_name in list(plan["sources"].keys()):
        plan["sources"][source_name] /= total_mass
    return plan


def format_hard_selector_summary(selectors):
    """Return a short printable summary for the selector config."""
    if not selectors:
        return "sources=0 samples=0"
    return f"sources={len(selectors.get('sources', ()))}, samples={len(selectors.get('samples', ()))}"


def format_selector_weight_summary(weight_rules):
    """Return a short printable summary for selector weighting rules."""
    if not weight_rules:
        return "sources=0, samples=0, negative=default"
    negative = "set" if weight_rules.get("negative") is not None else "default"
    return (
        f"sources={len(weight_rules.get('sources', ()))}, "
        f"samples={len(weight_rules.get('samples', ()))}, "
        f"negative={negative}"
    )


def preview_hard_selector_values(selectors, limit=8):
    """Return a short human-readable preview of selector values."""
    preview = []
    preview.extend(f"source:{value}" for value in sorted(selectors.get("sources", ()))[:limit])
    remaining = max(0, limit - len(preview))
    if remaining > 0:
        preview.extend(f"sample:{value}" for value in sorted(selectors.get("samples", ()))[:remaining])
    total = len(selectors.get("sources", ())) + len(selectors.get("samples", ()))
    suffix = " ..." if total > len(preview) else ""
    return ", ".join(preview) + suffix if preview else "(none)"


def preview_selector_weight_values(weight_rules, limit=8):
    """Return a short human-readable preview of selector weight rules."""
    preview = []
    preview.extend(
        f"source:{value}={weight_rules['sources'][value]:.2f}"
        for value in sorted(weight_rules.get("sources", ()))[:limit]
    )
    remaining = max(0, limit - len(preview))
    if remaining > 0:
        preview.extend(
            f"sample:{value}={weight_rules['samples'][value]:.2f}"
            for value in sorted(weight_rules.get("samples", ()))[:remaining]
        )
    if weight_rules.get("negative") is not None and len(preview) < limit:
        preview.append(f"negative={weight_rules['negative']:.2f}")
    total = len(weight_rules.get("sources", ())) + len(weight_rules.get("samples", ()))
    if weight_rules.get("negative") is not None:
        total += 1
    suffix = " ..." if total > len(preview) else ""
    return ", ".join(preview) + suffix if preview else "(none)"


def resolve_hard_selector_config(args):
    """Validate hard-selector arguments and load the selector file if present."""
    if args.hard_selector_mix_weight < 0.0 or args.hard_selector_mix_weight >= 1.0:
        raise ValueError("--hard_selector_mix_weight must be in [0.0, 1.0)")

    if not args.hard_selector_file:
        if args.hard_selector_mix_weight > 0.0:
            raise ValueError("--hard_selector_mix_weight > 0 requires --hard_selector_file")
        if args.report_val_hard:
            raise ValueError("--report_val_hard requires --hard_selector_file")
        if getattr(args, "augment_selector_only", False):
            raise ValueError("--augment_selector_only requires --hard_selector_file")
        return None

    if getattr(args, "augment_selector_only", False) and not args.augment:
        raise ValueError("--augment_selector_only requires --augment")

    selectors = load_hard_selector_file(args.hard_selector_file)
    if not selectors["sources"] and not selectors["samples"]:
        raise ValueError(f"--hard_selector_file {args.hard_selector_file} did not define any selectors")
    return selectors


def resolve_selector_weight_config(args):
    """Validate selector-weighting arguments and load rules if configured."""
    if args.source_balance_power < 0.0:
        raise ValueError("--source_balance_power must be >= 0")
    if args.source_balance_cap <= 0.0:
        raise ValueError("--source_balance_cap must be > 0")

    if not args.selector_weight_file:
        if args.source_weight_sampling and args.source_balance_power <= 0.0:
            raise ValueError("--source_weight_sampling requires --selector_weight_file or --source_balance_power > 0")
        return None

    rules = load_selector_weight_file(args.selector_weight_file)
    if not rules["sources"] and not rules["samples"] and rules["negative"] is None:
        raise ValueError(f"--selector_weight_file {args.selector_weight_file} did not define any weights")
    return rules


# --------------------------------------------------------------------------
# tf.data pipeline builder
# --------------------------------------------------------------------------

def _normalize_image(image, image_norm):
    """Normalize a single uint8 image to float32 (used inside tf.data.map)."""
    img = tf.cast(image, tf.float32)
    if image_norm == "imagenet":
        img = img / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
    elif image_norm == "zero_one":
        img = img / 255.0
    # else: raw255
    return img


def make_tf_dataset(
    images, coords, has_doc,
    batch_size, shuffle, augment=False, image_norm="imagenet",
    drop_remainder=False,
    repeat_forever=False,
    is_hard_source=None,
    sample_weight=None,
):
    """Build tf.data pipeline from numpy arrays (no augmentation).

    Images are kept as uint8 and normalized lazily per-batch to avoid
    allocating a full float32 copy of the dataset in memory.
    """
    del augment  # Batch augmentation happens in the training loop.
    ds = tf.data.Dataset.from_tensor_slices((
        images,  # keep uint8, normalize lazily
        {
            "coords": coords,
            "has_doc": has_doc,
            **({"is_hard_source": is_hard_source} if is_hard_source is not None else {}),
            **({"sample_weight": sample_weight} if sample_weight is not None else {}),
        },
    ))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(images), 10000),
            reshuffle_each_iteration=True,
        )
    if repeat_forever:
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(
        lambda img, tgt: (_normalize_image(img, image_norm), tgt),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_source_sampled_train_dataset(
    images,
    coords,
    has_doc,
    sample_names,
    sample_weights,
    batch_size,
    image_norm="imagenet",
    drop_remainder=True,
    is_hard_source=None,
):
    """Build an infinite train dataset sampled by source mass instead of raw frequency."""
    plan = build_source_sampling_plan(sample_names, has_doc, sample_weights)
    sample_names = np.asarray(sample_names, dtype=object)
    has_doc = np.asarray(has_doc, dtype=np.float32)
    source_names = np.asarray([extract_source_name(name) for name in sample_names], dtype=object)

    datasets = []
    weights = []
    summary = []

    neg_mask = has_doc <= 0.5
    if np.any(neg_mask) and plan["negative_mass"] > 0.0:
        datasets.append(
            make_tf_dataset(
                images[neg_mask],
                coords[neg_mask],
                has_doc[neg_mask],
                batch_size,
                shuffle=True,
                image_norm=image_norm,
                drop_remainder=drop_remainder,
                repeat_forever=True,
                is_hard_source=is_hard_source[neg_mask] if is_hard_source is not None else None,
                sample_weight=sample_weights[neg_mask],
            )
        )
        weights.append(plan["negative_mass"])
        summary.append(("negative", int(np.sum(neg_mask)), plan["negative_mass"]))

    pos_mask = has_doc > 0.5
    for source_name in sorted(plan["sources"].keys()):
        src_mask = pos_mask & (source_names == source_name)
        datasets.append(
            make_tf_dataset(
                images[src_mask],
                coords[src_mask],
                has_doc[src_mask],
                batch_size,
                shuffle=True,
                image_norm=image_norm,
                drop_remainder=drop_remainder,
                repeat_forever=True,
                is_hard_source=is_hard_source[src_mask] if is_hard_source is not None else None,
                sample_weight=sample_weights[src_mask],
            )
        )
        weights.append(plan["sources"][source_name])
        summary.append((source_name, int(np.sum(src_mask)), plan["sources"][source_name]))

    if not datasets:
        raise ValueError("source-sampled dataset could not be built from the provided weights")

    mixed = tf.data.Dataset.sample_from_datasets(datasets, weights=weights).prefetch(tf.data.AUTOTUNE)
    return mixed, summary


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
    parser.add_argument("--w_heatmap", type=float, default=0.0)
    parser.add_argument("--w_coord2d", type=float, default=0.2)
    parser.add_argument("--w_score", type=float, default=1.0)
    parser.add_argument("--heatmap_sigma_cells", type=float, default=1.5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    # Init
    parser.add_argument("--init_weights", type=str, default=None,
                        help="Path to initial weights for warm start")

    # Augmentation
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation during training")
    parser.add_argument("--rotation_range", type=float, default=5.0,
                        help="Random rotation range in degrees (requires --augment)")
    parser.add_argument("--scale_range", type=float, default=0.0,
                        help="Random scale range (0.15 means 0.85x-1.0x, 0.0=disabled)")
    parser.add_argument("--aug_weak_epochs", type=int, default=0,
                        help="Final N epochs use color-only augmentation (0=disabled)")
    parser.add_argument("--aug_start_epoch", type=int, default=1,
                        help="Epoch from which augmentation can activate (1-based, default=1)")
    parser.add_argument("--aug_min_iou", type=float, default=0.0,
                        help="Min best_iou before augmentation activates (0.0=immediate)")
    parser.add_argument("--aug_factor", type=int, default=1,
                        help="Virtual train multiplier via repeated stochastic views per epoch (1=default)")
    parser.add_argument("--hard_selector_file", type=str, default=None,
                        help="Path to a txt file describing hard source/sample selectors "
                             "(see docs/HARD_SELECTOR_FORMAT.md)")
    parser.add_argument("--hard_selector_mix_weight", "--hard_source_mix_weight",
                        dest="hard_selector_mix_weight", type=float, default=0.0,
                        help="Fraction of train batches sampled from the hard-selector subset (0 disables)")
    parser.add_argument("--report_val_hard", action="store_true",
                        help="Report separate metrics on validation samples matched by --hard_selector_file")
    parser.add_argument("--augment_selector_only", action="store_true",
                        help="Apply geometric augmentation only to selector-matched samples")
    parser.add_argument("--selector_weight_file", type=str, default=None,
                        help="Path to a txt file defining source/sample loss weights "
                             "(see docs/SELECTOR_WEIGHT_FORMAT.md)")
    parser.add_argument("--source_balance_power", type=float, default=0.0,
                        help="Inverse-frequency exponent for positive source balancing (0 disables)")
    parser.add_argument("--source_balance_cap", type=float, default=4.0,
                        help="Maximum multiplier produced by --source_balance_power")
    parser.add_argument("--source_weight_sampling", action="store_true",
                        help="Sample train batches by source mass derived from selector weights / source balancing")

    return parser.parse_args()


def should_activate_augmentation(epoch, best_iou, aug_start_epoch, aug_min_iou):
    """Check if delayed augmentation conditions are met (AND logic)."""
    return epoch >= aug_start_epoch and best_iou >= aug_min_iou


def resolve_effective_aug_factor(args):
    """Validate augmentation-factor constraints and return the effective factor."""
    if args.aug_factor < 1:
        raise ValueError("--aug_factor must be >= 1")

    if not args.augment:
        if args.aug_factor > 1:
            raise ValueError("--aug_factor > 1 requires --augment")
        return 1

    if args.aug_factor > 1:
        if args.aug_start_epoch != 1:
            raise ValueError("--aug_factor > 1 requires --aug_start_epoch 1")
        if args.aug_min_iou > 0.0:
            raise ValueError("--aug_factor > 1 requires --aug_min_iou 0.0")

    return args.aug_factor


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    effective_aug_factor = resolve_effective_aug_factor(args)
    hard_selectors = resolve_hard_selector_config(args)
    selector_weight_rules = resolve_selector_weight_config(args)
    if args.source_weight_sampling and args.hard_selector_mix_weight > 0.0:
        raise ValueError("--source_weight_sampling cannot be combined with --hard_selector_mix_weight")
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
    need_names = (
        bool(hard_selectors)
        or args.report_val_hard
        or selector_weight_rules is not None
        or args.source_balance_power > 0.0
        or args.source_weight_sampling
    )
    if need_names:
        train_images, train_coords, train_has_doc, train_names = load_dataset_fast(
            args.data_root, "train", args.img_size, args.num_workers, return_names=True,
        )
        val_images, val_coords, val_has_doc, val_names = load_dataset_fast(
            args.data_root, "val", args.img_size, args.num_workers, return_names=True,
        )
    else:
        train_images, train_coords, train_has_doc = load_dataset_fast(
            args.data_root, "train", args.img_size, args.num_workers,
        )
        val_images, val_coords, val_has_doc = load_dataset_fast(
            args.data_root, "val", args.img_size, args.num_workers,
        )
        train_names = None
        val_names = None

    n_train = len(train_images)
    n_val = len(val_images)
    base_train_steps = n_train // args.batch_size
    train_steps = base_train_steps * effective_aug_factor
    val_steps = math.ceil(n_val / args.batch_size)
    train_hard_mask = None
    val_hard_mask = None
    mixed_hard = False
    train_sample_weights = None
    source_sampling_summary = None

    if hard_selectors:
        train_hard_mask = build_hard_selector_mask(train_names, train_has_doc, hard_selectors)
        val_hard_mask = build_hard_selector_mask(val_names, val_has_doc, hard_selectors)

    if selector_weight_rules is not None or args.source_balance_power > 0.0:
        train_sample_weights = build_selector_sample_weights(
            train_names,
            train_has_doc,
            weight_rules=selector_weight_rules,
            source_balance_power=args.source_balance_power,
            source_balance_cap=args.source_balance_cap,
        )

    repeat_train = effective_aug_factor > 1 or args.hard_selector_mix_weight > 0.0 or args.source_weight_sampling

    print(f"Creating train dataset ({n_train} images)...", end=" ", flush=True)
    if args.source_weight_sampling:
        base_train_ds, source_sampling_summary = make_source_sampled_train_dataset(
            train_images,
            train_coords,
            train_has_doc,
            train_names,
            train_sample_weights if train_sample_weights is not None else np.ones(n_train, dtype=np.float32),
            args.batch_size,
            image_norm=args.input_norm,
            drop_remainder=True,
            is_hard_source=train_hard_mask.astype(np.float32) if train_hard_mask is not None else None,
        )
    else:
        base_train_ds = make_tf_dataset(
            train_images, train_coords, train_has_doc,
            args.batch_size, shuffle=True, image_norm=args.input_norm,
            drop_remainder=True,
            repeat_forever=repeat_train,
            is_hard_source=train_hard_mask.astype(np.float32) if train_hard_mask is not None else None,
            sample_weight=train_sample_weights,
        )
    train_ds = base_train_ds
    if args.hard_selector_mix_weight > 0.0:
        n_hard = int(train_hard_mask.sum())
        if n_hard == 0:
            print("no hard-selector samples found; disabling hard-selector mixing", flush=True)
            args.hard_selector_mix_weight = 0.0
        else:
            hard_train_ds = make_tf_dataset(
                train_images[train_hard_mask],
                train_coords[train_hard_mask],
                train_has_doc[train_hard_mask],
                args.batch_size,
                shuffle=True,
                image_norm=args.input_norm,
                drop_remainder=True,
                repeat_forever=True,
                is_hard_source=np.ones(n_hard, dtype=np.float32),
                sample_weight=train_sample_weights[train_hard_mask] if train_sample_weights is not None else None,
            )
            train_ds = tf.data.Dataset.sample_from_datasets(
                [base_train_ds, hard_train_ds],
                weights=[1.0 - args.hard_selector_mix_weight, args.hard_selector_mix_weight],
            ).prefetch(tf.data.AUTOTUNE)
            mixed_hard = True
    print("done", flush=True)

    print(f"Creating val dataset ({n_val} images)...", end=" ", flush=True)
    val_ds = make_tf_dataset(
        val_images, val_coords, val_has_doc,
        args.batch_size, shuffle=False, image_norm=args.input_norm,
        repeat_forever=False,
        is_hard_source=val_hard_mask.astype(np.float32) if val_hard_mask is not None else None,
    )
    print("done", flush=True)

    # Build model
    bw = args.backbone_weights
    if bw and bw.strip().lower() in ("none", "null"):
        bw = None

    # Skip backbone download when using init_weights (all weights will be overwritten)
    if args.init_weights and bw == "imagenet":
        print("Skipping ImageNet backbone download (--init_weights will overwrite all weights)")
        bw = None

    print("Building model...", end=" ", flush=True)
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
    print(f"done ({net.count_params():,} params)")

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
        w_heatmap=args.w_heatmap,
        w_coord2d=args.w_coord2d,
        w_score=args.w_score,
        heatmap_sigma_cells=args.heatmap_sigma_cells,
        label_smoothing=args.label_smoothing,
    )

    # Optimizer with cosine schedule
    steps_per_epoch = train_steps
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
          f"w_simcc={args.w_simcc} w_coord={args.w_coord} "
          f"w_heatmap={args.w_heatmap} w_coord2d={args.w_coord2d} "
          f"w_score={args.w_score} hm_sigma={args.heatmap_sigma_cells}")
    print(f"  Schedule: warmup={args.warmup_epochs}ep cosine wd={args.weight_decay}")
    if effective_aug_factor > 1:
        print(f"  Aug factor: {effective_aug_factor}x "
              f"(base_steps={base_train_steps}/ep -> effective_steps={train_steps}/ep)")
    else:
        print(f"  Aug factor: 1x (default)")
    if hard_selectors:
        train_hard_count = int(train_hard_mask.sum())
        val_hard_count = int(val_hard_mask.sum())
        print(f"  Hard sel:  file={args.hard_selector_file} mix={args.hard_selector_mix_weight:.2f} "
              f"train_pos={train_hard_count} val_pos={val_hard_count}")
        print(f"  Hard sel definitions: {format_hard_selector_summary(hard_selectors)}")
        if mixed_hard or args.augment_selector_only:
            print(f"  Hard sel preview: {preview_hard_selector_values(hard_selectors)}")
    if selector_weight_rules or args.source_balance_power > 0.0:
        mean_w = float(np.mean(train_sample_weights)) if train_sample_weights is not None else 1.0
        max_w = float(np.max(train_sample_weights)) if train_sample_weights is not None else 1.0
        print(
            f"  Weights:   file={args.selector_weight_file or '-'} "
            f"balance_pow={args.source_balance_power:.2f} cap={args.source_balance_cap:.2f} "
            f"mean={mean_w:.2f} max={max_w:.2f}"
        )
        if selector_weight_rules:
            print(f"  Weight defs: {format_selector_weight_summary(selector_weight_rules)}")
            print(f"  Weight prev: {preview_selector_weight_values(selector_weight_rules)}")
    if source_sampling_summary is not None:
        top_sources = sorted(source_sampling_summary, key=lambda item: item[2], reverse=True)[:8]
        preview = ", ".join(f"{name}={mass:.3f}" for name, _, mass in top_sources)
        print(f"  Src samp:  enabled ({preview})")
    if args.augment_selector_only:
        print("  Aug sel:   geometry limited to selector-matched samples")
    print(f"  Steps:    {train_steps}/ep train, {val_steps}/ep val")
    if args.augment:
        print(f"  Augment:  rotation={args.rotation_range}\u00b0 scale={args.scale_range} "
              f"weak_last={args.aug_weak_epochs}ep")
        if args.aug_start_epoch > 1:
            print(f"  Aug start epoch: {args.aug_start_epoch}")
        if args.aug_min_iou > 0:
            print(f"  Aug min IoU:     {args.aug_min_iou}")
        # Check ImageProjectiveTransformV3 availability for rotation
        if args.rotation_range > 0:
            try:
                _test = tf.zeros([1, 4, 4, 3])
                _t = tf.constant([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
                tf.raw_ops.ImageProjectiveTransformV3(
                    images=_test, transforms=_t,
                    output_shape=tf.constant([4, 4]),
                    interpolation="BILINEAR", fill_mode="NEAREST", fill_value=0.0)
            except Exception:
                print("  WARNING: ImageProjectiveTransformV3 not available, disabling rotation")
                args.rotation_range = 0.0
    else:
        print(f"  Augment:  DISABLED")

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

    # Delayed augmentation activation
    aug_active = args.augment and (args.aug_start_epoch <= 1 and args.aug_min_iou <= 0.0)

    logged_weak = False

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Determine augmentation mode for this epoch
        use_weak_aug = (args.augment and args.aug_weak_epochs > 0
                        and epoch >= args.epochs - args.aug_weak_epochs + 1)
        if use_weak_aug and not logged_weak:
            print(f"  -> Switching to weak augmentation (color only) "
                  f"for final {args.aug_weak_epochs} epochs")
            logged_weak = True

        # Train
        trainer.reset_metrics()
        epoch_ds = train_ds.take(train_steps) if repeat_train else train_ds
        train_pbar = tqdm(epoch_ds, total=train_steps,
                          desc=f"  Epoch {epoch}/{args.epochs} [Train]",
                          leave=False, ncols=100)
        for images, targets in train_pbar:
            if aug_active:
                coords = targets["coords"]
                has_doc = targets["has_doc"]
                aug_mask = targets.get("is_hard_source") if args.augment_selector_only else None
                sample_weight = targets.get("sample_weight")
                if use_weak_aug:
                    images = tf_augment_color_only(images, image_norm=args.input_norm)
                else:
                    images, coords = tf_augment_batch(
                        images, coords, has_doc,
                        img_size=args.img_size,
                        image_norm=args.input_norm,
                        rotation_range=args.rotation_range,
                        scale_range=args.scale_range,
                        aug_mask=aug_mask,
                    )
                targets = {
                    "coords": coords,
                    "has_doc": has_doc,
                    **({"is_hard_source": aug_mask} if aug_mask is not None else {}),
                    **({"sample_weight": sample_weight} if sample_weight is not None else {}),
                }
            trainer.train_step((images, targets))
            train_pbar.set_postfix(loss=f"{float(trainer.loss_tracker.result()):.4f}")
        train_pbar.close()
        train_m = {k: float(v) for k, v in trainer._get_metrics_dict().items()}

        # Validate
        trainer.reset_metrics()
        val_metrics.reset()
        val_hard_metrics = ValidationMetrics(img_size=args.img_size) if args.report_val_hard else None
        val_pbar = tqdm(val_ds, total=val_steps,
                        desc=f"  Epoch {epoch}/{args.epochs} [Val]  ",
                        leave=False, ncols=100)
        for images, targets in val_pbar:
            val_result, coords_pred_t, score_logit_t = trainer.test_step((images, targets))
            coords_pred = coords_pred_t.numpy()
            score_logit = score_logit_t.numpy()
            score_pred = 1.0 / (1.0 + np.exp(-np.clip(score_logit, -60.0, 60.0)))
            val_metrics.update(
                coords_pred, targets["coords"].numpy(),
                score_pred, targets["has_doc"].numpy(),
            )
            if val_hard_metrics is not None and "is_hard_source" in targets:
                hard_mask_batch = targets["is_hard_source"].numpy() > 0.5
                if np.any(hard_mask_batch):
                    val_hard_metrics.update(
                        coords_pred[hard_mask_batch],
                        targets["coords"].numpy()[hard_mask_batch],
                        score_pred[hard_mask_batch],
                        targets["has_doc"].numpy()[hard_mask_batch],
                    )
            val_pbar.set_postfix(loss=f"{float(trainer.loss_tracker.result()):.4f}")
        val_pbar.close()
        val_m = {k: float(v) for k, v in trainer._get_metrics_dict().items()}
        detailed = val_metrics.compute()
        detailed_hard = None
        if val_hard_metrics is not None and val_hard_metrics.pred_coords_list:
            detailed_hard = val_hard_metrics.compute()

        epoch_time = time.time() - t0
        iou = detailed["mean_iou"]
        val_loss = val_m["loss"]
        val_err = detailed["corner_error_px"]
        try:
            lr_now = float(lr_schedule(optimizer.iterations))
        except Exception:
            lr_now = float(args.learning_rate)

        # Track bests
        is_best_iou = iou > best_iou and iou > 1e-4
        if is_best_iou:
            best_iou = iou
            net.save_weights(str(output_dir / "best_model.weights.h5"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Check delayed augmentation activation
        if args.augment and not aug_active:
            if should_activate_augmentation(epoch, best_iou,
                                            args.aug_start_epoch, args.aug_min_iou):
                aug_active = True
                print(f"  -> Augmentation activated at epoch {epoch} "
                      f"(best_iou={best_iou:.4f})")

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
                    f"  med={detailed['median_iou']:.4f}"
                    f"  best={best_iou:.4f}")
        if is_best_iou:
            iou_line += " ★ NEW BEST"
        print(iou_line)
        print(f"  |- Error      train={train_m['corner_err_px']:.2f}px"
              f"      val={val_err:.2f}px"
              f"{_delta_str_px(val_err, prev_err)}")
        print(f"  |             min={detailed['corner_error_min_px']:.2f}px"
              f"  mean={val_err:.2f}px"
              f"  p95={detailed['corner_error_p95_px']:.2f}px"
              f"  max={detailed['corner_error_max_px']:.2f}px")
        print(f"  |- Recall     @50={detailed['recall_50']*100:.2f}%"
              f"  @75={detailed['recall_75']*100:.2f}%"
              f"  @90={detailed['recall_90']*100:.2f}%"
              f"  @95={detailed['recall_95']*100:.2f}%")
        n_doc = detailed['num_with_doc']
        if n_doc > 0:
            _pct = lambda v: f"{v/n_doc*100:.1f}%"
            print(f"  |- Outliers   IoU<.90={detailed['num_iou_lt_90']}/{n_doc} ({_pct(detailed['num_iou_lt_90'])})"
                  f"  <.95={detailed['num_iou_lt_95']}/{n_doc} ({_pct(detailed['num_iou_lt_95'])})"
                  f"  <.99={detailed['num_iou_lt_99']}/{n_doc} ({_pct(detailed['num_iou_lt_99'])})")
            print(f"  |             err>10px={detailed['num_err_gt_10']}/{n_doc} ({_pct(detailed['num_err_gt_10'])})"
                  f"  >20px={detailed['num_err_gt_20']}/{n_doc} ({_pct(detailed['num_err_gt_20'])})"
                  f"  >50px={detailed['num_err_gt_50']}/{n_doc} ({_pct(detailed['num_err_gt_50'])})")
        print(f"  |- Score      f1={detailed['cls_f1']*100:.2f}%"
              f"  prec={detailed['cls_precision']*100:.2f}%"
              f"  rec={detailed['cls_recall']*100:.2f}%"
              f"  acc={detailed['cls_accuracy']*100:.2f}%"
              f"  (TP={detailed['cls_tp']} FP={detailed['cls_fp']}"
              f" FN={detailed['cls_fn']} TN={detailed['cls_tn']})")
        if detailed_hard is not None:
            print(f"  |- ValHard    n={detailed_hard['num_with_doc']}"
                  f"  iou={detailed_hard['mean_iou']:.4f}"
                  f"  med={detailed_hard['median_iou']:.4f}"
                  f"  err={detailed_hard['corner_error_px']:.2f}px"
                  f"  @95={detailed_hard['recall_95']*100:.2f}%")
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
            "val_hard_mean_iou": detailed_hard["mean_iou"] if detailed_hard is not None else None,
            "val_hard_median_iou": detailed_hard["median_iou"] if detailed_hard is not None else None,
            "val_hard_corner_error_px": detailed_hard["corner_error_px"] if detailed_hard is not None else None,
            "val_hard_recall_95": detailed_hard["recall_95"] if detailed_hard is not None else None,
            "val_hard_num_with_doc": detailed_hard["num_with_doc"] if detailed_hard is not None else None,
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
