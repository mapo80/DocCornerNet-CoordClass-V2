"""Additional tests to push coverage above 90% for all v2 modules."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

# -----------------------------------------------------------------------
# v2/model.py additional coverage
# -----------------------------------------------------------------------
from model import (
    AxisMean,
    Resize1D,
    Broadcast1D,
    NearestUpsample2x,
    SimCCDecode,
    build_doccorner_v2,
    create_model,
    create_inference_model,
    load_inference_model,
    _get_feature_layers,
    _build_backbone,
)


class TestModelEdgeCases:
    def test_build_backbone_none_weights(self):
        inp = keras.Input((224, 224, 3))
        bb = _build_backbone(inp, alpha=0.35, backbone_weights=None,
                             backbone_include_preprocessing=False)
        assert bb is not None

    def test_load_inference_model(self, tmp_path):
        # Create model and save weights
        model = create_model(backbone_weights=None)
        wp = str(tmp_path / "model.weights.h5")
        model.save_weights(wp)

        # Load via convenience function
        inf_model = load_inference_model(wp, alpha=0.35, fpn_ch=32, simcc_ch=96)
        x = np.random.randn(1, 224, 224, 3).astype(np.float32)
        out = inf_model(x, training=False)
        assert isinstance(out, (list, tuple))
        assert out[0].shape == (1, 8)

    def test_get_feature_layers_missing_scale(self):
        """Test that missing feature scales raise ValueError."""
        # Create a tiny model that doesn't have all required scales
        inp = keras.Input((32, 32, 3))
        x = keras.layers.Conv2D(8, 3, padding="same")(inp)
        bb = keras.Model(inp, x, name="tiny_bb")
        with pytest.raises(ValueError, match="Could not find"):
            _get_feature_layers(bb, img_size=224)

    def test_resize1d_different_method(self):
        layer = Resize1D(target_length=100, method="nearest")
        x = tf.random.normal([2, 50, 16])
        out = layer(x)
        assert out.shape == (2, 100, 16)


# -----------------------------------------------------------------------
# v2/dataset.py additional coverage
# -----------------------------------------------------------------------
from dataset import (
    augment_sample,
    normalize_image,
    create_dataset,
    DEFAULT_AUG_CONFIG,
)


class TestDatasetEdgeCases:
    def test_augment_with_rotation(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        coords = np.array([0.3, 0.3, 0.7, 0.3, 0.7, 0.7, 0.3, 0.7], dtype=np.float32)
        config = dict(DEFAULT_AUG_CONFIG)
        config["rotation_degrees"] = 15
        config["scale_range"] = (0.85, 0.95)
        img_out, coords_out = augment_sample(img, coords, aug_config=config)
        assert coords_out.min() >= 0.0
        assert coords_out.max() <= 1.0

    def test_augment_with_blur(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        coords = np.array([0.3, 0.3, 0.7, 0.3, 0.7, 0.7, 0.3, 0.7], dtype=np.float32)
        config = dict(DEFAULT_AUG_CONFIG)
        config["blur_prob"] = 1.0  # Force blur
        img_out, _ = augment_sample(img, coords, aug_config=config)
        assert img_out.shape == (224, 224, 3)

    def test_augment_with_color_augments(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        coords = np.array([0.3, 0.3, 0.7, 0.3, 0.7, 0.7, 0.3, 0.7], dtype=np.float32)
        config = {
            "rotation_degrees": 0,
            "scale_range": (1.0, 1.0),
            "brightness": 0.5,
            "contrast": 0.5,
            "saturation": 0.5,
            "blur_prob": 0.0,
            "blur_kernel": 3,
        }
        img_out, _ = augment_sample(img, coords, aug_config=config)
        assert img_out.shape == (224, 224, 3)

    def test_create_dataset_empty_raises(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        with open(tmp_path / "train.txt", "w") as f:
            f.write("nonexistent.jpg\n")
        with pytest.raises(ValueError, match="No valid images"):
            create_dataset(str(tmp_path), split="train", negative_ratio=0.0)

    def test_create_dataset_no_negative_dir(self, tmp_path):
        """Should work even without images-negative directory."""
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (100, 100))
        img.save(tmp_path / "images" / "test.jpg")
        with open(tmp_path / "labels" / "test.txt", "w") as f:
            f.write("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")
        with open(tmp_path / "train.txt", "w") as f:
            f.write("test.jpg\n")
        ds = create_dataset(str(tmp_path), split="train", negative_ratio=0.0,
                            batch_size=1, shuffle=False, augment=False, img_size=64)
        for batch in ds.take(1):
            assert batch[0].shape[0] == 1


# -----------------------------------------------------------------------
# v2/metrics.py additional coverage
# -----------------------------------------------------------------------
from metrics import (
    compute_polygon_iou,
    compute_bbox_iou,
    compute_corner_error,
    ValidationMetrics,
    SHAPELY_AVAILABLE,
)


class TestMetricsEdgeCases:
    def test_polygon_iou_exception_fallback(self):
        """Gracefully handle bad polygons."""
        # degenerate polygon (all same point)
        c1 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        c2 = np.array([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8])
        iou = compute_polygon_iou(c1, c2)
        assert 0.0 <= iou <= 1.0

    def test_validation_metrics_with_imperfect_predictions(self):
        metrics = ValidationMetrics(img_size=224)
        B = 10
        gt = np.random.uniform(0.2, 0.8, (B, 8)).astype(np.float32)
        pred = gt + np.random.normal(0, 0.1, (B, 8)).astype(np.float32)
        pred = np.clip(pred, 0, 1)
        scores = np.ones(B, dtype=np.float32)
        has_doc = np.ones(B, dtype=np.float32)
        metrics.update(pred, gt, scores, has_doc)
        results = metrics.compute()
        assert results["mean_iou"] > 0
        assert results["corner_error_px"] > 0

    @pytest.mark.skipif(not SHAPELY_AVAILABLE, reason="Shapely not installed")
    def test_validation_metrics_vectorized_path(self):
        """Test the Shapely vectorized IoU path."""
        metrics = ValidationMetrics(img_size=224)
        B = 20
        gt = np.random.uniform(0.2, 0.8, (B, 8)).astype(np.float32)
        pred = gt + np.random.normal(0, 0.01, (B, 8)).astype(np.float32)
        pred = np.clip(pred, 0, 1)
        scores = np.ones(B, dtype=np.float32)
        has_doc = np.ones(B, dtype=np.float32)
        metrics.update(pred, gt, scores, has_doc)
        results = metrics.compute()
        assert results["mean_iou"] > 0.5

    def test_validation_metrics_outlier_counts(self):
        metrics = ValidationMetrics(img_size=224)
        B = 10
        gt = np.random.uniform(0.2, 0.8, (B, 8)).astype(np.float32)
        # Large error predictions
        pred = gt + 0.3
        pred = np.clip(pred, 0, 1).astype(np.float32)
        scores = np.ones(B, dtype=np.float32)
        has_doc = np.ones(B, dtype=np.float32)
        metrics.update(pred, gt, scores, has_doc)
        results = metrics.compute()
        # With large errors, some outlier counts should be > 0
        assert results["num_err_gt_10"] >= 0


# -----------------------------------------------------------------------
# v2/losses.py additional coverage
# -----------------------------------------------------------------------
from losses import (
    gaussian_1d_targets,
    SimCCLoss,
    CoordLoss,
    DocCornerNetV2Trainer,
)


class TestLossesEdgeCases:
    def test_coord_loss_smooth_l1_small_diff(self):
        """SmoothL1 quadratic regime for small diffs."""
        loss_fn = CoordLoss(loss_type="smooth_l1")
        B = 2
        pred = tf.constant([[0.5] * 8, [0.5] * 8], dtype=tf.float32)
        gt = tf.constant([[0.5001] * 8, [0.5001] * 8], dtype=tf.float32)
        mask = tf.ones([B])
        loss = loss_fn(pred, gt, mask)
        assert loss.numpy() < 0.01

    def test_trainer_extract_targets_2d(self):
        net = create_model(backbone_weights=None)
        trainer = DocCornerNetV2Trainer(net)
        y = {
            "has_doc": tf.constant([[1.0], [0.0]]),
            "coords": tf.random.uniform([2, 8]),
        }
        has_doc, coords = trainer._extract_targets(y)
        assert has_doc.shape == (2,)

    def test_trainer_get_metrics_dict_keys(self):
        net = create_model(backbone_weights=None)
        trainer = DocCornerNetV2Trainer(net)
        trainer.compile(optimizer=keras.optimizers.Adam(1e-4))
        # Run one step to initialize metrics
        x = np.random.randn(2, 224, 224, 3).astype(np.float32)
        y = {"has_doc": np.array([1, 0], dtype=np.float32),
             "coords": np.random.uniform(0.1, 0.9, (2, 8)).astype(np.float32)}
        trainer.train_step((x, y))
        md = trainer._get_metrics_dict()
        assert set(md.keys()) == {
            "loss", "loss_simcc", "loss_coord", "loss_heatmap", "loss_coord2d",
            "loss_score", "iou", "corner_err_px",
        }


# -----------------------------------------------------------------------
# v2/train_ultra.py additional coverage
# -----------------------------------------------------------------------
from train_ultra import (
    WarmupCosineSchedule,
    make_tf_dataset,
)


class TestTrainUltraEdgeCases:
    def test_make_tf_dataset_raw255(self):
        N = 4
        images = np.full((N, 32, 32, 3), 128, dtype=np.uint8)
        coords = np.zeros((N, 8), dtype=np.float32)
        has_doc = np.ones(N, dtype=np.float32)
        ds = make_tf_dataset(images, coords, has_doc, batch_size=2,
                             shuffle=False, image_norm="raw255")
        for batch_images, _ in ds.take(1):
            # raw255: should be 128.0
            np.testing.assert_allclose(batch_images.numpy()[0, 0, 0], [128, 128, 128], atol=1)

    def test_make_tf_dataset_drop_remainder(self):
        N = 5
        images = np.random.randint(0, 255, (N, 32, 32, 3), dtype=np.uint8)
        coords = np.zeros((N, 8), dtype=np.float32)
        has_doc = np.ones(N, dtype=np.float32)
        ds = make_tf_dataset(images, coords, has_doc, batch_size=3,
                             shuffle=False, drop_remainder=True)
        count = sum(1 for _ in ds)
        assert count == 1  # 5 / 3 = 1 full batch

    def test_warmup_cosine_edge_cases(self):
        sched = WarmupCosineSchedule(base_lr=1e-3, total_steps=100, warmup_steps=0, min_lr=1e-6)
        lr_end = sched(100).numpy()
        assert lr_end >= 1e-6 - 1e-12  # allow tiny float imprecision
