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
from collections import Counter, deque
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    from .model import create_model, create_inference_model
except ImportError:
    from model import create_model, create_inference_model


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _output_suffix(output_mode: str) -> str:
    mode = str(output_mode).lower().strip()
    if mode == "heads":
        return "_heads"
    if mode == "simcc_packed":
        return "_simcc"
    return ""


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
    parser.add_argument("--output_mode", type=str, default="decoded",
                        choices=["decoded", "heads", "simcc_packed"],
                        help="decoded: [coords, score_logit]; heads: export raw head tensors; simcc_packed: [score_logit, simcc_xy] with simcc_xy=[B,num_bins,8] for v1/WASM INT8 compatibility")

    # TFLite
    parser.add_argument("--representative_data", type=str, default=None)
    parser.add_argument("--representative_limit", type=int, default=100,
                        help="Max representative samples used for full-int8 calibration")

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
    output_mode = getattr(args, "output_mode", "decoded")

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

    print(f"Loaded weights, creating export model...")
    if output_mode in {"heads", "simcc_packed"}:
        export_train_model = create_model(
            alpha=alpha, fpn_ch=fpn_ch, simcc_ch=simcc_ch,
            img_size=img_size, num_bins=num_bins, tau=tau,
            backbone_weights=None,
            xnnpack_safe=True,
        )
        export_train_model.set_weights(train_model.get_weights())
        if output_mode == "heads":
            model = keras.Model(
                inputs=export_train_model.input,
                outputs=[
                    export_train_model.output["simcc_x"],
                    export_train_model.output["simcc_y"],
                    export_train_model.output["corner_heatmap"],
                    export_train_model.output["corner_offset"],
                    export_train_model.output["score_logit"],
                ],
                name="DocCornerNet_V2_HeadsExport",
            )
        else:
            simcc_x_bins = keras.layers.Permute((2, 1), name="simcc_x_bins_first")(
                export_train_model.output["simcc_x"]
            )
            simcc_y_bins = keras.layers.Permute((2, 1), name="simcc_y_bins_first")(
                export_train_model.output["simcc_y"]
            )
            simcc_xy = keras.layers.Concatenate(axis=-1, name="simcc_packed")(
                [simcc_x_bins, simcc_y_bins]
            )
            model = keras.Model(
                inputs=export_train_model.input,
                # TFLite reverses multi-output ordering in Interpreter.get_output_details().
                # Export [simcc_xy, score_logit] here so the final flatbuffer matches the
                # existing WASM/V1 contract: output 0 = score_logit [B,1], output 1 = simcc_xy [B,num_bins,8].
                outputs=[simcc_xy, export_train_model.output["score_logit"]],
                name="DocCornerNet_V2_SimCCPackedExport",
            )
    else:
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


def _normalize_image(image: np.ndarray, img_size: int) -> np.ndarray:
    image = tf.image.resize(image, [img_size, img_size], method="bilinear").numpy()
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image


def _iter_representative_images(data_path: Path, img_size: int, limit: int):
    if limit <= 0:
        raise ValueError("--representative_limit must be > 0")

    if data_path.is_file() and data_path.suffix == ".parquet":
        parquet_files = [data_path]
    elif data_path.is_dir():
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(sorted(data_path.glob(f"*{ext}")))
        if image_files:
            yielded = 0
            for image_path in image_files[:limit]:
                image = tf.io.decode_image(
                    tf.io.read_file(str(image_path)),
                    channels=3,
                    expand_animations=False,
                ).numpy()
                yield _normalize_image(image, img_size)
                yielded += 1
            if yielded == 0:
                raise ValueError(f"No usable images found in {data_path}")
            return

        parquet_files = sorted(data_path.glob("*.parquet"))
        if not parquet_files:
            for split_name in ("val", "validation", "train", "test"):
                split_dir = data_path / split_name
                if split_dir.is_dir():
                    parquet_files = sorted(split_dir.glob("*.parquet"))
                    if parquet_files:
                        break
    else:
        parquet_files = []

    if not parquet_files:
        raise ValueError(
            "Representative data path must be an image directory, parquet file, "
            "parquet split directory, or dataset root containing val/validation/train/test."
        )

    import pyarrow.parquet as pq

    yielded = 0
    for parquet_file in parquet_files:
        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(batch_size=min(limit, 32), columns=["image"]):
            for row in batch.to_pylist():
                image_feature = row["image"]
                if isinstance(image_feature, dict):
                    image_bytes = image_feature.get("bytes")
                else:
                    image_bytes = image_feature
                if not image_bytes:
                    continue
                image = tf.io.decode_image(
                    image_bytes,
                    channels=3,
                    expand_animations=False,
                ).numpy()
                yield _normalize_image(image, img_size)
                yielded += 1
                if yielded >= limit:
                    return

    if yielded == 0:
        raise ValueError(f"No representative samples found in {data_path}")


def _build_representative_dataset(representative_data_path: str, img_size: int, limit: int):
    data_path = Path(representative_data_path)

    def representative_dataset():
        for image in _iter_representative_images(data_path, img_size=img_size, limit=limit):
            yield [np.expand_dims(image.astype(np.float32), axis=0)]

    return representative_dataset


def export_tflite(model, output_path: Path, img_size: int,
                  quantize: bool = False, representative_data_path: str = None,
                  representative_limit: int = 100) -> float:
    """Export to TFLite format."""
    label = "int8" if quantize else "float32"
    print(f"\nExporting TFLite ({label}) to {output_path}...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_data_path:
            converter.representative_dataset = _build_representative_dataset(
                representative_data_path,
                img_size=img_size,
                limit=representative_limit,
            )
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            print(f"  Using full-int8 calibration from {representative_data_path} "
                  f"(limit={representative_limit})")
        else:
            print("  Warning: No representative data, using dynamic range quantization")

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB")
    return size_mb


def _make_interpreter(tflite_path: Path, use_default_delegates: bool, num_threads: int = 4):
    kwargs = {
        "model_path": str(tflite_path),
        "num_threads": num_threads,
    }
    resolver_type = getattr(tf.lite.experimental, "OpResolverType", None)
    if not use_default_delegates and resolver_type is not None:
        kwargs["experimental_op_resolver_type"] = resolver_type.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    interpreter = tf.lite.Interpreter(**kwargs)
    interpreter.allocate_tensors()
    return interpreter


def _collect_execution_plan(interpreter: tf.lite.Interpreter):
    ops = interpreter._get_ops_details()
    producer_by_tensor = {}
    for node_idx, op in enumerate(ops):
        for tensor_idx in op.get("outputs", []):
            if tensor_idx >= 0:
                producer_by_tensor[int(tensor_idx)] = node_idx

    output_tensors = [int(d["index"]) for d in interpreter.get_output_details()]
    active_nodes = set()
    queue = deque(output_tensors)

    while queue:
        tensor_idx = queue.popleft()
        producer = producer_by_tensor.get(tensor_idx)
        if producer is None or producer in active_nodes:
            continue
        active_nodes.add(producer)
        for input_idx in ops[producer].get("inputs", []):
            input_idx = int(input_idx)
            if input_idx >= 0 and input_idx in producer_by_tensor:
                queue.append(input_idx)

    return sorted(active_nodes), ops


def inspect_tflite_model(tflite_path: Path, num_threads: int = 4) -> dict:
    """Inspect TFLite model and estimate post-delegate execution plan."""
    baseline = _make_interpreter(tflite_path, use_default_delegates=False, num_threads=num_threads)
    delegated = _make_interpreter(tflite_path, use_default_delegates=True, num_threads=num_threads)

    baseline_plan, baseline_ops = _collect_execution_plan(baseline)
    delegated_plan, delegated_ops = _collect_execution_plan(delegated)

    delegated_plan_ops = [delegated_ops[idx] for idx in delegated_plan]
    non_delegated = [
        {
            "index": int(idx),
            "op_name": delegated_ops[idx]["op_name"],
        }
        for idx in delegated_plan
        if delegated_ops[idx]["op_name"] != "DELEGATE"
    ]

    all_tensor_dtypes = Counter(
        str(detail["dtype"].__name__ if hasattr(detail["dtype"], "__name__") else detail["dtype"])
        for detail in delegated.get_tensor_details()
    )
    input_details = delegated.get_input_details()
    output_details = delegated.get_output_details()

    report = {
        "nodes": int(len(delegated_ops)),
        "before_delegate_nodes": int(len(baseline_ops)),
        "execution_plan_nodes": int(len(delegated_plan)),
        "delegate_plan_nodes": int(sum(op["op_name"] == "DELEGATE" for op in delegated_plan_ops)),
        "non_delegated_builtin_ops": non_delegated,
        "fully_delegated": bool(non_delegated == [] and any(
            op["op_name"] == "DELEGATE" for op in delegated_plan_ops
        )),
        "input_details": [
            {
                "name": detail["name"],
                "index": int(detail["index"]),
                "dtype": str(detail["dtype"].__name__ if hasattr(detail["dtype"], "__name__") else detail["dtype"]),
                "shape": [int(v) for v in detail["shape"]],
                "quantization": [float(detail["quantization"][0]), int(detail["quantization"][1])],
            }
            for detail in input_details
        ],
        "output_details": [
            {
                "name": detail["name"],
                "index": int(detail["index"]),
                "dtype": str(detail["dtype"].__name__ if hasattr(detail["dtype"], "__name__") else detail["dtype"]),
                "shape": [int(v) for v in detail["shape"]],
                "quantization": [float(detail["quantization"][0]), int(detail["quantization"][1])],
            }
            for detail in output_details
        ],
        "tensor_dtype_histogram": dict(sorted(all_tensor_dtypes.items())),
    }
    return report


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
        print(f"  output shapes: {[tuple(o.shape) for o in outputs]}")
    else:
        print(f"  outputs: {type(outputs)}")

    results = {"formats": {}}

    if "savedmodel" in args.format:
        size = export_savedmodel(model, output_dir / "savedmodel", img_size)
        results["formats"]["savedmodel"] = {"size_mb": size}

    if "tflite" in args.format:
        suffix = _output_suffix(args.output_mode)
        path = output_dir / f"model_float32{suffix}.tflite"
        size = export_tflite(model, path, img_size)
        results["formats"]["tflite_float32"] = {"size_mb": size}
        bench = benchmark_tflite(path, img_size)
        results["formats"]["tflite_float32"]["benchmark"] = bench
        report = inspect_tflite_model(path)
        results["formats"]["tflite_float32"]["delegate_report"] = report
        with open(output_dir / f"model_float32{suffix}_delegate_report.json", "w") as f:
            json.dump(report, f, indent=2)

    if "tflite_int8" in args.format:
        suffix = _output_suffix(args.output_mode)
        path = output_dir / f"model_int8{suffix}.tflite"
        size = export_tflite(model, path, img_size, quantize=True,
                             representative_data_path=args.representative_data,
                             representative_limit=args.representative_limit)
        results["formats"]["tflite_int8"] = {"size_mb": size}
        bench = benchmark_tflite(path, img_size)
        results["formats"]["tflite_int8"]["benchmark"] = bench
        report = inspect_tflite_model(path)
        results["formats"]["tflite_int8"]["delegate_report"] = report
        with open(output_dir / f"model_int8{suffix}_delegate_report.json", "w") as f:
            json.dump(report, f, indent=2)

    with open(output_dir / "export_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nExport complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
