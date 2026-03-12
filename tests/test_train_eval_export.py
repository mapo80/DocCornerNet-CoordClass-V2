"""Tests for v2/train_ultra.py, v2/evaluate.py, v2/export.py — smoke tests."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from model import create_model, create_inference_model
from losses import DocCornerNetV2Trainer
from metrics import ValidationMetrics
from train_ultra import (
    WarmupCosineSchedule,
    make_tf_dataset,
)
from export import (
    export_tflite,
    benchmark_tflite,
)


# ---------------------------------------------------------------------------
# Training smoke tests
# ---------------------------------------------------------------------------

class TestWarmupCosineSchedule:
    def test_warmup_phase(self):
        sched = WarmupCosineSchedule(base_lr=1e-3, total_steps=1000, warmup_steps=100)
        lr_0 = sched(0).numpy()
        lr_50 = sched(50).numpy()
        lr_100 = sched(100).numpy()
        # During warmup, LR should increase
        assert lr_0 < lr_50 < lr_100

    def test_cosine_phase(self):
        sched = WarmupCosineSchedule(base_lr=1e-3, total_steps=1000, warmup_steps=100)
        lr_100 = sched(100).numpy()
        lr_500 = sched(500).numpy()
        lr_999 = sched(999).numpy()
        # During cosine phase, LR should decrease
        assert lr_100 > lr_500 > lr_999

    def test_no_warmup(self):
        sched = WarmupCosineSchedule(base_lr=1e-3, total_steps=1000, warmup_steps=0)
        lr_0 = sched(0).numpy()
        assert lr_0 == pytest.approx(1e-3, rel=0.1)

    def test_get_config(self):
        sched = WarmupCosineSchedule(base_lr=1e-3, total_steps=1000, warmup_steps=100)
        cfg = sched.get_config()
        assert cfg["base_lr"] == 1e-3
        assert cfg["total_steps"] == 1000
        assert cfg["warmup_steps"] == 100


class TestMakeTfDataset:
    def test_basic(self):
        N = 16
        images = np.random.randint(0, 255, (N, 224, 224, 3), dtype=np.uint8)
        coords = np.random.uniform(0.1, 0.9, (N, 8)).astype(np.float32)
        has_doc = np.ones(N, dtype=np.float32)

        ds = make_tf_dataset(
            images, coords, has_doc,
            batch_size=4, shuffle=False, augment=False,
        )
        for batch_images, batch_targets in ds.take(1):
            assert batch_images.shape == (4, 224, 224, 3)
            assert batch_targets["coords"].shape == (4, 8)
            assert batch_targets["has_doc"].shape == (4,)

    def test_shuffle(self):
        N = 16
        images = np.random.randint(0, 255, (N, 224, 224, 3), dtype=np.uint8)
        coords = np.random.uniform(0.1, 0.9, (N, 8)).astype(np.float32)
        has_doc = np.ones(N, dtype=np.float32)

        ds = make_tf_dataset(
            images, coords, has_doc,
            batch_size=4, shuffle=True, augment=False,
        )
        assert ds is not None

    def test_normalization_zero_one(self):
        N = 4
        images = np.full((N, 32, 32, 3), 255, dtype=np.uint8)
        coords = np.zeros((N, 8), dtype=np.float32)
        has_doc = np.ones(N, dtype=np.float32)

        ds = make_tf_dataset(
            images, coords, has_doc,
            batch_size=4, shuffle=False, augment=False,
            image_norm="zero_one",
        )
        for batch_images, _ in ds.take(1):
            np.testing.assert_allclose(batch_images.numpy(), 1.0, atol=1e-5)


class TestTrainingSmoke:
    def test_full_training_step(self):
        """End-to-end: build model, create trainer, run one training step."""
        net = create_model(backbone_weights=None)
        trainer = DocCornerNetV2Trainer(net, bins=224, sigma_px=2.0, tau=0.5)
        trainer.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=2e-4, weight_decay=1e-4)
        )

        B = 2
        x = np.random.randn(B, 224, 224, 3).astype(np.float32)
        y = {
            "has_doc": np.array([1.0, 0.0], dtype=np.float32),
            "coords": np.random.uniform(0.1, 0.9, (B, 8)).astype(np.float32),
        }
        metrics = trainer.train_step((x, y))
        assert float(metrics["loss"]) > 0


# ---------------------------------------------------------------------------
# Evaluator smoke tests
# ---------------------------------------------------------------------------

class TestEvaluatorSmoke:
    def test_model_evaluation_loop(self):
        """Simulate evaluation loop on synthetic data."""
        net = create_model(backbone_weights=None)

        val_metrics = ValidationMetrics(img_size=224)
        B = 4
        x = np.random.randn(B, 224, 224, 3).astype(np.float32)
        outputs = net(x, training=False)
        coords_pred = outputs["coords"].numpy()
        score_logit = outputs["score_logit"].numpy()
        score_pred = 1.0 / (1.0 + np.exp(-np.clip(score_logit, -60, 60)))

        gt_coords = np.random.uniform(0.2, 0.8, (B, 8)).astype(np.float32)
        has_doc = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)

        val_metrics.update(coords_pred, gt_coords, score_pred, has_doc)
        results = val_metrics.compute()
        assert "mean_iou" in results
        assert results["num_samples"] == B


# ---------------------------------------------------------------------------
# Export smoke tests
# ---------------------------------------------------------------------------

class TestExportSmoke:
    def test_inference_model_creation(self):
        train_model = create_model(backbone_weights=None)
        inf_model = create_inference_model(train_model)
        x = np.random.randn(1, 224, 224, 3).astype(np.float32)
        out = inf_model(x, training=False)
        assert isinstance(out, (list, tuple))
        assert len(out) == 2

    def test_tflite_export(self, tmp_path):
        """Export to TFLite float32 and verify outputs."""
        train_model = create_model(backbone_weights=None)
        inf_model = create_inference_model(train_model)

        tflite_path = tmp_path / "test.tflite"
        size_mb = export_tflite(inf_model, tflite_path, img_size=224, quantize=False)
        assert size_mb > 0
        assert tflite_path.exists()

        # Verify TFLite model runs
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        dummy = np.random.randn(1, 224, 224, 3).astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()

        # Should have 2 outputs: coords and score_logit
        assert len(output_details) == 2

    def test_tflite_benchmark(self, tmp_path):
        """Benchmark TFLite model."""
        train_model = create_model(backbone_weights=None)
        inf_model = create_inference_model(train_model)

        tflite_path = tmp_path / "bench.tflite"
        export_tflite(inf_model, tflite_path, img_size=224)
        results = benchmark_tflite(tflite_path, img_size=224, num_runs=5)
        assert "mean_ms" in results
        assert results["mean_ms"] > 0

    def test_save_load_weights_roundtrip(self, tmp_path):
        """Full roundtrip: save weights, reload, verify output consistency."""
        model1 = create_model(backbone_weights=None)
        x = np.random.randn(1, 224, 224, 3).astype(np.float32)
        out1 = model1(x, training=False)

        weights_path = str(tmp_path / "model.weights.h5")
        model1.save_weights(weights_path)

        # Save config
        config = {
            "alpha": 0.35, "fpn_ch": 32, "simcc_ch": 96,
            "img_size": 224, "num_bins": 224, "tau": 1.0,
        }
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config, f)

        # Reload
        model2 = create_model(backbone_weights=None)
        model2.load_weights(weights_path)
        out2 = model2(x, training=False)

        np.testing.assert_allclose(
            out1["coords"].numpy(), out2["coords"].numpy(), atol=1e-5,
        )
        np.testing.assert_allclose(
            out1["score_logit"].numpy(), out2["score_logit"].numpy(), atol=1e-5,
        )
