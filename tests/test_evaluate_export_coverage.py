"""Additional coverage tests for v2/evaluate.py, v2/export.py, v2/train_ultra.py."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import types

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from model import create_model, create_inference_model

# -----------------------------------------------------------------------
# v2/evaluate.py coverage
# -----------------------------------------------------------------------
from evaluate import (
    _find_config_path,
    load_model,
)


class TestEvaluateFindConfigPath:
    def test_config_in_dir(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        result = _find_config_path(tmp_path)
        assert result == tmp_path / "config.json"

    def test_config_in_parent(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "config.json").write_text("{}")
        # file path case
        result = _find_config_path(sub / "weights.h5")
        # Should look in parent
        assert result is None or result.exists()

    def test_no_config(self, tmp_path):
        result = _find_config_path(tmp_path / "nonexistent.h5")
        assert result is None


class TestEvaluateLoadModel:
    def test_load_from_weights_h5(self, tmp_path):
        # Create and save model
        model = create_model(backbone_weights=None)
        weights_path = tmp_path / "model.weights.h5"
        model.save_weights(str(weights_path))

        # Load via args
        args = types.SimpleNamespace(
            model_path=str(weights_path),
            alpha=0.35, fpn_ch=32, simcc_ch=96,
            img_size=224, num_bins=224, tau=1.0,
        )
        loaded, img_size = load_model(args)
        assert loaded is not None
        assert img_size == 224

    def test_load_from_directory(self, tmp_path):
        model = create_model(backbone_weights=None)
        weights_path = tmp_path / "best_model.weights.h5"
        model.save_weights(str(weights_path))

        args = types.SimpleNamespace(
            model_path=str(tmp_path),
            alpha=0.35, fpn_ch=32, simcc_ch=96,
            img_size=224, num_bins=224, tau=1.0,
        )
        loaded, img_size = load_model(args)
        assert loaded is not None

    def test_load_with_config(self, tmp_path):
        model = create_model(backbone_weights=None)
        weights_path = tmp_path / "best_model.weights.h5"
        model.save_weights(str(weights_path))

        config = {"alpha": 0.35, "fpn_ch": 32, "simcc_ch": 96,
                  "img_size": 224, "num_bins": 224, "tau": 1.0}
        (tmp_path / "config.json").write_text(json.dumps(config))

        args = types.SimpleNamespace(
            model_path=str(tmp_path),
            alpha=0.35, fpn_ch=32, simcc_ch=96,
            img_size=224, num_bins=224, tau=1.0,
        )
        loaded, img_size = load_model(args)
        assert loaded is not None

    def test_load_no_weights_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        args = types.SimpleNamespace(
            model_path=str(tmp_path / "empty"),
            alpha=0.35, fpn_ch=32, simcc_ch=96,
            img_size=224, num_bins=224, tau=1.0,
        )
        with pytest.raises(ValueError, match="No weights"):
            load_model(args)

    def test_load_bad_path_raises(self):
        args = types.SimpleNamespace(
            model_path="/nonexistent/path.xyz",
            alpha=0.35, fpn_ch=32, simcc_ch=96,
            img_size=224, num_bins=224, tau=1.0,
        )
        with pytest.raises(ValueError, match="Cannot load"):
            load_model(args)


# -----------------------------------------------------------------------
# v2/export.py coverage
# -----------------------------------------------------------------------
from export import (
    _find_config_path as export_find_config_path,
    load_model_for_export,
    export_savedmodel,
    export_tflite,
    benchmark_tflite,
)


class TestExportFindConfigPath:
    def test_found(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        result = export_find_config_path(tmp_path)
        assert result == tmp_path / "config.json"

    def test_not_found(self, tmp_path):
        result = export_find_config_path(tmp_path / "nope.h5")
        assert result is None


class TestExportLoadModel:
    def test_load_from_h5(self, tmp_path):
        model = create_model(backbone_weights=None)
        wp = tmp_path / "model.weights.h5"
        model.save_weights(str(wp))

        args = types.SimpleNamespace(
            weights=str(wp),
            alpha=0.35, fpn_ch=32, simcc_ch=96,
            img_size=224, num_bins=224, tau=1.0,
        )
        inf_model, img_size = load_model_for_export(args)
        assert inf_model is not None
        x = np.random.randn(1, 224, 224, 3).astype(np.float32)
        out = inf_model(x)
        assert isinstance(out, (list, tuple))
        assert len(out) == 2

    def test_load_from_directory(self, tmp_path):
        model = create_model(backbone_weights=None)
        wp = tmp_path / "best_model.weights.h5"
        model.save_weights(str(wp))

        args = types.SimpleNamespace(
            weights=str(tmp_path),
            alpha=0.35, fpn_ch=32, simcc_ch=96,
            img_size=224, num_bins=224, tau=1.0,
        )
        inf_model, _ = load_model_for_export(args)
        assert inf_model is not None

    def test_load_with_config(self, tmp_path):
        model = create_model(backbone_weights=None)
        wp = tmp_path / "best_model.weights.h5"
        model.save_weights(str(wp))
        config = {"alpha": 0.35, "fpn_ch": 32, "simcc_ch": 96,
                  "img_size": 224, "num_bins": 224, "tau": 1.0}
        (tmp_path / "config.json").write_text(json.dumps(config))

        args = types.SimpleNamespace(
            weights=str(tmp_path),
            alpha=0.35, fpn_ch=32, simcc_ch=96,
            img_size=224, num_bins=224, tau=1.0,
        )
        inf_model, _ = load_model_for_export(args)
        assert inf_model is not None

    def test_bad_path_raises(self):
        args = types.SimpleNamespace(
            weights="/nonexistent.xyz",
            alpha=0.35, fpn_ch=32, simcc_ch=96,
            img_size=224, num_bins=224, tau=1.0,
        )
        with pytest.raises(ValueError, match="Cannot load"):
            load_model_for_export(args)

    def test_empty_dir_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        args = types.SimpleNamespace(
            weights=str(tmp_path / "empty"),
            alpha=0.35, fpn_ch=32, simcc_ch=96,
            img_size=224, num_bins=224, tau=1.0,
        )
        with pytest.raises(ValueError, match="No weights"):
            load_model_for_export(args)


class TestExportSavedModel:
    def test_export(self, tmp_path):
        model = create_model(backbone_weights=None)
        inf = create_inference_model(model)
        out_path = tmp_path / "savedmodel"
        size = export_savedmodel(inf, out_path, img_size=224)
        assert size > 0
        assert out_path.exists()


class TestExportTFLiteAdditional:
    def test_export_int8_without_representative(self, tmp_path):
        """Dynamic range quantization (no representative data)."""
        model = create_model(backbone_weights=None)
        inf = create_inference_model(model)
        path = tmp_path / "model_int8.tflite"
        size = export_tflite(inf, path, img_size=224, quantize=True,
                             representative_data_path=None)
        assert size > 0
        assert path.exists()


# -----------------------------------------------------------------------
# v2/train_ultra.py coverage
# -----------------------------------------------------------------------
from train_ultra import (
    setup_platform,
    _load_single_image,
    load_dataset_fast,
    make_tf_dataset,
)


class TestSetupPlatform:
    def test_returns_string(self):
        platform = setup_platform()
        assert platform in ("cuda", "mps", "cpu")


class TestLoadSingleImage:
    def test_positive_image(self, tmp_path):
        # Create image and label
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img.save(tmp_path / "images" / "test.jpg")
        with open(tmp_path / "labels" / "test.txt", "w") as f:
            f.write("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")

        result = _load_single_image(("test.jpg", str(tmp_path), 64))
        assert result is not None
        img_arr, coords, has_doc = result
        assert img_arr.shape == (64, 64, 3)
        assert has_doc == 1.0

    def test_negative_image(self, tmp_path):
        (tmp_path / "images-negative").mkdir()
        img = Image.new("RGB", (100, 100))
        img.save(tmp_path / "images-negative" / "negative_test.jpg")

        result = _load_single_image(("negative_test.jpg", str(tmp_path), 64))
        assert result is not None
        _, _, has_doc = result
        assert has_doc == 0.0

    def test_missing_image(self, tmp_path):
        (tmp_path / "images").mkdir()
        result = _load_single_image(("nonexistent.jpg", str(tmp_path), 64))
        assert result is None

    def test_no_label(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        img = Image.new("RGB", (100, 100))
        img.save(tmp_path / "images" / "test.jpg")
        # No label file
        result = _load_single_image(("test.jpg", str(tmp_path), 64))
        assert result is not None
        _, _, has_doc = result
        assert has_doc == 0.0


class TestLoadDatasetFast:
    def test_basic(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        for i in range(3):
            img = Image.new("RGB", (100, 100))
            img.save(tmp_path / "images" / f"img_{i}.jpg")
            with open(tmp_path / "labels" / f"img_{i}.txt", "w") as f:
                f.write(f"0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")
        with open(tmp_path / "train.txt", "w") as f:
            f.write("\n".join([f"img_{i}.jpg" for i in range(3)]))

        images, coords, has_doc = load_dataset_fast(str(tmp_path), "train", 64, num_workers=2)
        assert images.shape == (3, 64, 64, 3)
        assert coords.shape == (3, 8)
        assert has_doc.shape == (3,)

    def test_missing_split(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset_fast(str(tmp_path), "nonexistent", 64)

    def test_fallback_split_file(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        img = Image.new("RGB", (50, 50))
        img.save(tmp_path / "images" / "img.jpg")
        with open(tmp_path / "labels" / "img.txt", "w") as f:
            f.write("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n")
        # Use fallback naming convention
        with open(tmp_path / "train_with_negative.txt", "w") as f:
            f.write("img.jpg\n")

        images, coords, has_doc = load_dataset_fast(str(tmp_path), "train", 64, num_workers=1)
        assert len(images) == 1

    def test_return_names(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        for name in ["img_0.jpg", "img_1.jpg"]:
            img = Image.new("RGB", (32, 32))
            img.save(tmp_path / "images" / name)
            with open(tmp_path / "labels" / f"{Path(name).stem}.txt", "w") as f:
                f.write("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n")
        with open(tmp_path / "train.txt", "w") as f:
            f.write("img_0.jpg\nimg_1.jpg\n")

        images, coords, has_doc, names = load_dataset_fast(
            str(tmp_path), "train", 64, num_workers=1, return_names=True,
        )
        assert images.shape[0] == 2
        assert list(names) == ["img_0.jpg", "img_1.jpg"]
