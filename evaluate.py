"""
Evaluation script for DocCornerNet V2.

Usage:
    python -m v2.evaluate \
        --model_path runs/v2_smoke \
        --data_root dataset/DocCornerDataset_small \
        --split val
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import create_model
from dataset import create_dataset
from metrics import ValidationMetrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DocCornerNet V2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model directory or weights file")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--input_norm", type=str, default="imagenet",
                        choices=["imagenet", "zero_one", "raw255"])

    # Model config overrides
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--fpn_ch", type=int, default=32)
    parser.add_argument("--simcc_ch", type=int, default=96)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_bins", type=int, default=224)
    parser.add_argument("--tau", type=float, default=1.0)

    return parser.parse_args()


def _find_config_path(model_path: Path) -> Optional[Path]:
    candidates = []
    if model_path.is_dir():
        candidates.extend([model_path / "config.json", model_path.parent / "config.json"])
    else:
        candidates.extend([model_path.parent / "config.json"])
    for c in candidates:
        if c.exists():
            return c
    return None


def load_model(args):
    """Load model from path (weights or directory)."""
    model_path = Path(args.model_path)

    # Try config
    config_path = _find_config_path(model_path)
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        alpha = config.get("alpha", args.alpha)
        fpn_ch = config.get("fpn_ch", args.fpn_ch)
        simcc_ch = config.get("simcc_ch", args.simcc_ch)
        img_size = config.get("img_size", args.img_size)
        num_bins = config.get("num_bins", args.num_bins)
        tau = config.get("tau", args.tau)
        print(f"Loaded config from {config_path}")
    else:
        alpha = args.alpha
        fpn_ch = args.fpn_ch
        simcc_ch = args.simcc_ch
        img_size = args.img_size
        num_bins = args.num_bins
        tau = args.tau

    model = create_model(
        alpha=alpha, fpn_ch=fpn_ch, simcc_ch=simcc_ch,
        img_size=img_size, num_bins=num_bins, tau=tau,
        backbone_weights=None,
    )

    # Find and load weights
    if model_path.suffix == ".h5":
        model.load_weights(str(model_path))
        print(f"Loaded weights from {model_path}")
    elif model_path.is_dir():
        for wf in [
            "best_model.weights.h5",
            "final_model.weights.h5",
        ]:
            wp = model_path / wf
            if wp.exists():
                model.load_weights(str(wp))
                print(f"Loaded weights from {wp}")
                break
        else:
            raise ValueError(f"No weights found in {model_path}")
    else:
        raise ValueError(f"Cannot load model from {model_path}")

    return model, img_size


def main():
    args = parse_args()

    print("=" * 60)
    print("DocCornerNet V2 Evaluation")
    print("=" * 60)

    model, img_size = load_model(args)
    print(f"Model parameters: {model.count_params():,}")

    # Create dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = create_dataset(
        data_root=args.data_root,
        split=args.split,
        img_size=img_size,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        drop_remainder=False,
        image_norm=args.input_norm,
    )

    # Evaluate
    metrics = ValidationMetrics(img_size=img_size)
    for images, targets in dataset:
        outputs = model(images, training=False)
        coords_pred = outputs["coords"].numpy()
        score_logit = outputs["score_logit"].numpy()
        score_pred = 1.0 / (1.0 + np.exp(-np.clip(score_logit, -60.0, 60.0)))
        metrics.update(
            coords_pred, targets["coords"].numpy(),
            score_pred, targets["has_doc"].numpy(),
        )

    results = metrics.compute()

    # Print results
    print(f"\nGeometry Metrics (on {results['num_with_doc']} images with documents):")
    print(f"  Mean IoU:             {results['mean_iou']:.4f}")
    print(f"  Median IoU:           {results['median_iou']:.4f}")
    print(f"  Corner Error (mean):  {results['corner_error_px']:.2f} px")
    print(f"  Corner Error (p95):   {results['corner_error_p95_px']:.2f} px")
    print(f"  Recall@90:            {results['recall_90']*100:.1f}%")
    print(f"  Recall@95:            {results['recall_95']*100:.1f}%")
    print(f"  Recall@99:            {results['recall_99']*100:.1f}%")

    print(f"\nClassification Metrics (on {results['num_samples']} total images):")
    print(f"  Accuracy:             {results['cls_accuracy']*100:.1f}%")
    print(f"  F1 Score:             {results['cls_f1']*100:.1f}%")

    # Target comparison
    iou = results["mean_iou"]
    ce_mean = results["corner_error_px"]
    iou_ok = "pass" if iou >= 0.99 else "FAIL"
    ce_ok = "pass" if ce_mean <= 1.0 else "FAIL"
    print(f"\nTargets: IoU >= 99%: {iou*100:.2f}% [{iou_ok}] | Error <= 1px: {ce_mean:.2f}px [{ce_ok}]")


if __name__ == "__main__":
    main()
