"""Integration tests for main() functions in evaluate.py, export.py, train_ultra.py.

These tests exercise the CLI main flows with synthetic data to push coverage
over the 90% threshold.
"""

import json
import sys
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from model import create_model, create_inference_model


@pytest.fixture
def trained_model_dir(tmp_path):
    """Create a directory with saved model weights and config."""
    model = create_model(backbone_weights=None)
    model.save_weights(str(tmp_path / "best_model.weights.h5"))
    config = {
        "alpha": 0.35, "fpn_ch": 32, "simcc_ch": 96,
        "img_size": 224, "num_bins": 224, "tau": 1.0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


@pytest.fixture
def mini_data_dir(tmp_path):
    """Create a minimal dataset directory."""
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "images-negative").mkdir()
    (data / "labels").mkdir()

    for i in range(4):
        img = Image.new("RGB", (50, 50), color=(i * 60, 100, 150))
        img.save(data / "images" / f"img_{i}.jpg")
        with open(data / "labels" / f"img_{i}.txt", "w") as f:
            f.write(f"0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")

    for i in range(2):
        img = Image.new("RGB", (50, 50), color=(200, 200, 200))
        img.save(data / "images-negative" / f"negative_n{i}.jpg")

    with open(data / "train.txt", "w") as f:
        names = [f"img_{i}.jpg" for i in range(3)] + ["negative_n0.jpg"]
        f.write("\n".join(names))
    with open(data / "val.txt", "w") as f:
        names = ["img_3.jpg", "negative_n1.jpg"]
        f.write("\n".join(names))

    return data


# ---------------------------------------------------------------------------
# evaluate.py main flow
# ---------------------------------------------------------------------------

class TestEvaluateMain:
    def test_main_flow(self, trained_model_dir, mini_data_dir):
        from evaluate import main as eval_main

        with patch("sys.argv", [
            "evaluate",
            "--model_path", str(trained_model_dir),
            "--data_root", str(mini_data_dir),
            "--split", "val",
            "--batch_size", "2",
            "--img_size", "64",
        ]):
            # Should run without error
            eval_main()


# ---------------------------------------------------------------------------
# export.py main flow
# ---------------------------------------------------------------------------

class TestExportMain:
    def test_main_flow(self, trained_model_dir, tmp_path):
        from export import main as export_main

        out_dir = tmp_path / "export_out"
        with patch("sys.argv", [
            "export",
            "--weights", str(trained_model_dir),
            "--output_dir", str(out_dir),
            "--format", "tflite",
            "--img_size", "224",
        ]):
            export_main()

        assert (out_dir / "model_float32.tflite").exists()
        assert (out_dir / "export_results.json").exists()


# ---------------------------------------------------------------------------
# export.py — additional code paths
# ---------------------------------------------------------------------------

class TestExportAdditionalPaths:
    def test_load_from_final_weights(self, tmp_path):
        """Load from directory with final_model.weights.h5."""
        model = create_model(backbone_weights=None)
        model.save_weights(str(tmp_path / "final_model.weights.h5"))

        import types
        from export import load_model_for_export
        args = types.SimpleNamespace(
            weights=str(tmp_path),
            alpha=0.35, fpn_ch=32, simcc_ch=96,
            img_size=224, num_bins=224, tau=1.0,
        )
        inf_model, _ = load_model_for_export(args)
        assert inf_model is not None


# ---------------------------------------------------------------------------
# train_ultra.py main flow (smoke, tiny dataset, 1 epoch)
# ---------------------------------------------------------------------------

class TestTrainMain:
    def test_main_flow(self, mini_data_dir, tmp_path):
        from train_ultra import main as train_main

        out_dir = tmp_path / "train_out"
        with patch("sys.argv", [
            "train",
            "--data_root", str(mini_data_dir),
            "--output_dir", str(out_dir),
            "--epochs", "1",
            "--batch_size", "2",
            "--img_size", "64",
            "--num_bins", "64",
            "--backbone_weights", "none",
            "--warmup_epochs", "0",
            "--num_workers", "2",
        ]):
            train_main()

        assert (out_dir / "config.json").exists()
        assert (out_dir / "final_model.weights.h5").exists()
        assert (out_dir / "training_log.csv").exists()

    def test_main_with_init_weights(self, mini_data_dir, tmp_path):
        """Test warm-start from init_weights."""
        from train_ultra import main as train_main

        # Create initial weights
        model = create_model(
            backbone_weights=None, img_size=64, num_bins=64,
        )
        init_path = tmp_path / "init.weights.h5"
        model.save_weights(str(init_path))

        out_dir = tmp_path / "train_init"
        with patch("sys.argv", [
            "train",
            "--data_root", str(mini_data_dir),
            "--output_dir", str(out_dir),
            "--epochs", "1",
            "--batch_size", "2",
            "--img_size", "64",
            "--num_bins", "64",
            "--backbone_weights", "none",
            "--warmup_epochs", "0",
            "--num_workers", "2",
            "--init_weights", str(init_path),
        ]):
            train_main()

        assert (out_dir / "final_model.weights.h5").exists()


# ---------------------------------------------------------------------------
# metrics.py — Shapely-specific code paths
# ---------------------------------------------------------------------------
from metrics import (
    SHAPELY_AVAILABLE,
    ValidationMetrics,
    coords_to_polygon,
    compute_polygon_iou,
)


@pytest.mark.skipif(not SHAPELY_AVAILABLE, reason="Shapely not installed")
class TestMetricsShapelyPaths:
    def test_coords_to_polygon_multi(self):
        """Test with a MultiPolygon result from make_valid."""
        # Bowtie that results in MultiPolygon
        coords = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
        poly = coords_to_polygon(coords)
        assert poly.area > 0

    def test_vectorized_iou_with_invalid_polygons(self):
        """Some invalid polygons should fall back to per-sample."""
        metrics = ValidationMetrics(img_size=224)
        B = 5
        # Mix of valid and degenerate
        gt = np.array([
            [0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8],  # valid
            [0.3, 0.3, 0.7, 0.3, 0.7, 0.7, 0.3, 0.7],  # valid
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # degenerate
            [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],  # valid
            [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],  # self-intersecting
        ], dtype=np.float32)
        pred = gt.copy()
        pred[:2] += 0.01
        pred = np.clip(pred, 0, 1).astype(np.float32)
        scores = np.ones(B, dtype=np.float32)
        has_doc = np.ones(B, dtype=np.float32)
        metrics.update(pred, gt, scores, has_doc)
        results = metrics.compute()
        assert results["mean_iou"] > 0
        assert results["num_with_doc"] == B

    def test_empty_polygon(self):
        """Empty/zero-area polygon."""
        c1 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        c2 = np.array([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8])
        iou = compute_polygon_iou(c1, c2)
        assert 0.0 <= iou <= 1.0
