"""
Export DocCornerNet V2 to deployment formats.

Supports: SavedModel, TFLite (float32, int8), ONNX (optional).

Usage:
    python -m v2.export \
        --weights runs/v2_smoke/best_model.weights.h5 \
        --output_dir exported_v2 \
        --format savedmodel tflite
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import create_model, create_inference_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export DocCornerNet V2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights (.h5) or directory")
    parser.add_argument("--output_dir", type=str, default="./exported_v2")
    parser.add_argument("--format", type=str, nargs="+",
                        default=["savedmodel", "tflite"],
                        choices=["savedmodel", "tflite", "tflite_int8", "onnx"])

    # Model config
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--fpn_ch", type=int, default=32)
    parser.add_argument("--simcc_ch", type=int, default=96)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_bins", type=int, default=224)
    parser.add_argument("--tau", type=float, default=1.0)

    # TFLite
    parser.add_argument("--representative_data", type=str, default=None)

    return parser.parse_args()


def _find_config_path(weights_path: Path) -> Optional[Path]:
    candidates = []
    if weights_path.is_dir():
        candidates.extend([weights_path / "config.json", weights_path.parent / "config.json"])
    else:
        candidates.extend([weights_path.parent / "config.json"])
    for c in candidates:
        if c.exists():
            return c
    return None


def load_model_for_export(args):
    """Load model and return inference model."""
    weights_path = Path(args.weights)

    # Try config
    config_path = _find_config_path(weights_path)
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

    train_model = create_model(
        alpha=alpha, fpn_ch=fpn_ch, simcc_ch=simcc_ch,
        img_size=img_size, num_bins=num_bins, tau=tau,
        backbone_weights=None,
    )

    # Load weights
    if weights_path.suffix == ".h5":
        train_model.load_weights(str(weights_path))
    elif weights_path.is_dir():
        for wf in ["best_model.weights.h5", "final_model.weights.h5"]:
            wp = weights_path / wf
            if wp.exists():
                train_model.load_weights(str(wp))
                break
        else:
            raise ValueError(f"No weights found in {weights_path}")
    else:
        raise ValueError(f"Cannot load weights from {weights_path}")

    print(f"Loaded weights, creating inference model...")
    model = create_inference_model(train_model)
    return model, img_size


def export_savedmodel(model, output_path: Path, img_size: int) -> float:
    """Export to SavedModel format."""
    print(f"\nExporting SavedModel to {output_path}...")
    if hasattr(model, "export"):
        model.export(str(output_path))
    else:
        model.save(str(output_path), save_format="tf")
    size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB")
    return size_mb


def export_tflite(model, output_path: Path, img_size: int,
                  quantize: bool = False, representative_data_path: str = None) -> float:
    """Export to TFLite format."""
    label = "int8" if quantize else "float32"
    print(f"\nExporting TFLite ({label}) to {output_path}...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_data_path:
            def representative_dataset():
                data_path = Path(representative_data_path)
                images = list(data_path.glob("*.jpg"))[:100]
                for img_path in images:
                    img = tf.io.read_file(str(img_path))
                    img = tf.image.decode_jpeg(img, channels=3)
                    img = tf.image.resize(img, [img_size, img_size])
                    img = tf.cast(img, tf.float32) / 255.0
                    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                    yield [tf.expand_dims(img, 0)]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB")
    return size_mb


def benchmark_tflite(tflite_path: Path, img_size: int, num_runs: int = 100) -> dict:
    """Benchmark TFLite inference speed."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()

    input_dtype = input_details[0]["dtype"]
    if input_dtype == np.int8:
        dummy = np.random.randint(-128, 127, (1, img_size, img_size, 3)).astype(np.int8)
    else:
        dummy = np.random.randn(1, img_size, img_size, 3).astype(np.float32)

    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DocCornerNet V2 Export")
    print("=" * 60)

    model, img_size = load_model_for_export(args)
    print(f"Model parameters: {model.count_params():,}")

    # Test forward pass
    dummy = np.random.randn(1, img_size, img_size, 3).astype(np.float32)
    outputs = model(dummy, training=False)
    if isinstance(outputs, (list, tuple)):
        print(f"  coords shape: {outputs[0].shape}, score shape: {outputs[1].shape}")
    else:
        print(f"  outputs: {type(outputs)}")

    results = {"formats": {}}

    if "savedmodel" in args.format:
        size = export_savedmodel(model, output_dir / "savedmodel", img_size)
        results["formats"]["savedmodel"] = {"size_mb": size}

    if "tflite" in args.format:
        path = output_dir / "model_float32.tflite"
        size = export_tflite(model, path, img_size)
        results["formats"]["tflite_float32"] = {"size_mb": size}
        bench = benchmark_tflite(path, img_size)
        results["formats"]["tflite_float32"]["benchmark"] = bench

    if "tflite_int8" in args.format:
        path = output_dir / "model_int8.tflite"
        size = export_tflite(model, path, img_size, quantize=True,
                             representative_data_path=args.representative_data)
        results["formats"]["tflite_int8"] = {"size_mb": size}
        bench = benchmark_tflite(path, img_size)
        results["formats"]["tflite_int8"]["benchmark"] = bench

    with open(output_dir / "export_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nExport complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
