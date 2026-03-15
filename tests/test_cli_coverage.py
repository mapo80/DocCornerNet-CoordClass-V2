"""Tests to cover CLI-facing functions in evaluate.py, export.py, train_ultra.py."""

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from model import create_model, create_inference_model


# ---------------------------------------------------------------------------
# evaluate.py — parse_args and main flow
# ---------------------------------------------------------------------------
from v2 import evaluate as evaluate_mod


class TestEvaluateParseArgs:
    def test_parse_args(self):
        with patch("sys.argv", ["evaluate", "--model_path", "/tmp/m", "--data_root", "/tmp/d"]):
            args = evaluate_mod.parse_args()
            assert args.model_path == "/tmp/m"
            assert args.data_root == "/tmp/d"
            assert args.split == "val"
            assert args.batch_size == 32

    def test_parse_args_custom(self):
        with patch("sys.argv", [
            "evaluate", "--model_path", "/tmp/m", "--data_root", "/tmp/d",
            "--split", "test", "--batch_size", "16", "--alpha", "0.5",
            "--fpn_ch", "48", "--simcc_ch", "128", "--img_size", "256",
            "--num_bins", "336", "--tau", "0.5",
        ]):
            args = evaluate_mod.parse_args()
            assert args.split == "test"
            assert args.batch_size == 16
            assert args.alpha == 0.5


# ---------------------------------------------------------------------------
# export.py — parse_args and main flow
# ---------------------------------------------------------------------------
from v2 import export as export_mod


class TestExportParseArgs:
    def test_parse_args(self):
        with patch("sys.argv", ["export", "--weights", "/tmp/w.h5"]):
            args = export_mod.parse_args()
            assert args.weights == "/tmp/w.h5"
            assert args.output_dir == "./exported_v2"

    def test_parse_args_custom(self):
        with patch("sys.argv", [
            "export", "--weights", "/tmp/w.h5",
            "--output_dir", "/tmp/out",
            "--format", "savedmodel", "tflite_int8",
            "--alpha", "0.5", "--representative_data", "/tmp/data",
        ]):
            args = export_mod.parse_args()
            assert args.output_dir == "/tmp/out"
            assert "savedmodel" in args.format
            assert "tflite_int8" in args.format
            assert args.representative_data == "/tmp/data"


class TestExportSavedModelFlow:
    def test_full_savedmodel_export(self, tmp_path):
        model = create_model(backbone_weights=None)
        inf = create_inference_model(model)
        out = tmp_path / "sm"
        size = export_mod.export_savedmodel(inf, out, 224)
        assert size > 0


class TestExportBenchmarkInt8:
    def test_benchmark_int8_tflite(self, tmp_path):
        model = create_model(backbone_weights=None)
        inf = create_inference_model(model)
        path = tmp_path / "int8.tflite"
        export_mod.export_tflite(inf, path, 224, quantize=True)
        results = export_mod.benchmark_tflite(path, 224, num_runs=3)
        assert results["mean_ms"] > 0


# ---------------------------------------------------------------------------
# train_ultra.py — parse_args and more
# ---------------------------------------------------------------------------
from v2 import train_ultra as train_mod


class TestTrainParseArgs:
    def test_parse_args(self):
        with patch("sys.argv", ["train", "--data_root", "/tmp/d"]):
            args = train_mod.parse_args()
            assert args.data_root == "/tmp/d"
            assert args.epochs == 100
            assert args.batch_size == 32
            assert args.aug_factor == 1
            assert args.hard_selector_file is None
            assert args.augment_selector_only is False
            assert args.selector_weight_file is None
            assert args.source_balance_power == 0.0
            assert args.source_weight_sampling is False

    def test_parse_args_custom(self):
        with patch("sys.argv", [
            "train", "--data_root", "/tmp/d",
            "--epochs", "5", "--batch_size", "16",
            "--learning_rate", "1e-3", "--sigma_px", "3.0",
            "--loss_tau", "0.1", "--w_coord", "0.5",
            "--init_weights", "/tmp/w.h5",
            "--backbone_weights", "none",
            "--augment",
            "--aug_factor", "3",
            "--hard_selector_file", "/tmp/hard.txt",
            "--hard_source_mix_weight", "0.25",
            "--report_val_hard",
            "--augment_selector_only",
            "--selector_weight_file", "/tmp/weights.txt",
            "--source_balance_power", "0.5",
            "--source_weight_sampling",
        ]):
            args = train_mod.parse_args()
            assert args.epochs == 5
            assert args.sigma_px == 3.0
            assert args.init_weights == "/tmp/w.h5"
            assert args.aug_factor == 3
            assert args.hard_selector_file == "/tmp/hard.txt"
            assert args.hard_selector_mix_weight == 0.25
            assert args.report_val_hard is True
            assert args.augment_selector_only is True
            assert args.selector_weight_file == "/tmp/weights.txt"
            assert args.source_balance_power == 0.5
            assert args.source_weight_sampling is True


class TestTrainSetupPlatform:
    def test_cpu_path(self):
        # Force CPU by mocking GPU list
        with patch.object(tf.config, "list_physical_devices", return_value=[]):
            with patch("sys.platform", "linux"):
                platform = train_mod.setup_platform()
                assert platform == "cpu"


class TestTrainLoadSingleImageMore:
    def test_bad_image_file(self, tmp_path):
        """Corrupted image file should return None."""
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        # Write garbage to an "image" file
        (tmp_path / "images" / "bad.jpg").write_bytes(b"not an image")
        result = train_mod._load_single_image(("bad.jpg", str(tmp_path), 64))
        assert result is None

    def test_image_with_bad_label(self, tmp_path):
        """Image with unparseable label → has_doc=0."""
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        img = Image.new("RGB", (50, 50))
        img.save(tmp_path / "images" / "test.jpg")
        # Write invalid label
        (tmp_path / "labels" / "test.txt").write_text("corrupted data\n")
        result = train_mod._load_single_image(("test.jpg", str(tmp_path), 64))
        assert result is not None
        _, _, has_doc = result
        assert has_doc == 0.0


class TestMakeTfDatasetImageNet:
    def test_imagenet_normalization_values(self):
        """Verify ImageNet normalization is applied correctly."""
        N = 2
        # All zeros
        images = np.zeros((N, 32, 32, 3), dtype=np.uint8)
        coords = np.zeros((N, 8), dtype=np.float32)
        has_doc = np.ones(N, dtype=np.float32)
        ds = train_mod.make_tf_dataset(images, coords, has_doc, batch_size=N,
                                       shuffle=False, image_norm="imagenet")
        for batch_images, _ in ds.take(1):
            # With all-zero input: (0/255 - mean) / std
            pixel = batch_images.numpy()[0, 0, 0]
            expected_r = (0.0 - 0.485) / 0.229
            np.testing.assert_allclose(pixel[0], expected_r, atol=1e-4)
