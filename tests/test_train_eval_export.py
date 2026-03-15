"""Tests for v2/train_ultra.py, v2/evaluate.py, v2/export.py — smoke tests."""

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

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
    build_hard_selector_mask,
    extract_source_name,
    format_hard_selector_summary,
    load_hard_selector_file,
    make_tf_dataset,
    resolve_effective_aug_factor,
    resolve_hard_selector_config,
    should_activate_augmentation,
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
            batch_size=4, shuffle=False,         )
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
            batch_size=4, shuffle=True,         )
        assert ds is not None

    def test_normalization_zero_one(self):
        N = 4
        images = np.full((N, 32, 32, 3), 255, dtype=np.uint8)
        coords = np.zeros((N, 8), dtype=np.float32)
        has_doc = np.ones(N, dtype=np.float32)

        ds = make_tf_dataset(
            images, coords, has_doc,
            batch_size=4, shuffle=False,             image_norm="zero_one",
        )
        for batch_images, _ in ds.take(1):
            np.testing.assert_allclose(batch_images.numpy(), 1.0, atol=1e-5)

    def test_repeat_forever_reuses_samples(self):
        N = 4
        images = np.arange(N * 8 * 8 * 3, dtype=np.uint8).reshape(N, 8, 8, 3)
        coords = np.zeros((N, 8), dtype=np.float32)
        has_doc = np.ones(N, dtype=np.float32)

        ds = make_tf_dataset(
            images, coords, has_doc,
            batch_size=2, shuffle=False,
            repeat_forever=True,
        )
        batches = list(ds.take(3))
        assert len(batches) == 3
        np.testing.assert_allclose(
            batches[0][0].numpy(),
            batches[2][0].numpy(),
            atol=1e-6,
        )


class TestResolveEffectiveAugFactor:
    def test_default_factor(self):
        args = SimpleNamespace(
            augment=False,
            aug_factor=1,
            aug_start_epoch=1,
            aug_min_iou=0.0,
        )
        assert resolve_effective_aug_factor(args) == 1

    def test_requires_augment(self):
        args = SimpleNamespace(
            augment=False,
            aug_factor=2,
            aug_start_epoch=1,
            aug_min_iou=0.0,
        )
        with pytest.raises(ValueError, match="requires --augment"):
            resolve_effective_aug_factor(args)

    def test_requires_immediate_start(self):
        args = SimpleNamespace(
            augment=True,
            aug_factor=2,
            aug_start_epoch=3,
            aug_min_iou=0.0,
        )
        with pytest.raises(ValueError, match="requires --aug_start_epoch 1"):
            resolve_effective_aug_factor(args)

    def test_requires_zero_min_iou(self):
        args = SimpleNamespace(
            augment=True,
            aug_factor=2,
            aug_start_epoch=1,
            aug_min_iou=0.5,
        )
        with pytest.raises(ValueError, match="requires --aug_min_iou 0.0"):
            resolve_effective_aug_factor(args)

    def test_accepts_valid_factor(self):
        args = SimpleNamespace(
            augment=True,
            aug_factor=3,
            aug_start_epoch=1,
            aug_min_iou=0.0,
        )
        assert resolve_effective_aug_factor(args) == 3


class TestHardSelectorHelpers:
    def test_extract_source_name(self):
        assert extract_source_name("idcard_jj_004845.jpg") == "idcard_jj"
        assert extract_source_name("nested/path/receipt_occam_000052.jpg") == "receipt_occam"

    def test_load_hard_selector_file(self, tmp_path):
        selector_file = tmp_path / "hard.txt"
        selector_file.write_text(
            "\n".join([
                "# comment",
                "idcard_jj",
                "source:receipt_occam",
                "sample:receipt_occam_000052.jpg",
            ]),
            encoding="utf-8",
        )
        selectors = load_hard_selector_file(selector_file)
        assert selectors["sources"] == {"idcard_jj", "receipt_occam"}
        assert selectors["samples"] == {"receipt_occam_000052.jpg"}

    def test_build_hard_selector_mask(self):
        names = np.array([
            "idcard_jj_000001.jpg",
            "midv500_000123.jpg",
            "receipts_segmentation_000002.jpg",
            "negative_example_000003.jpg",
            "custom_case_000004.jpg",
        ], dtype=object)
        has_doc = np.array([1.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)
        selectors = {
            "sources": {"idcard_jj", "receipts_segmentation"},
            "samples": {"custom_case_000004.jpg"},
        }
        mask = build_hard_selector_mask(names, has_doc, selectors)
        np.testing.assert_array_equal(mask, np.array([True, False, True, False, True]))

    def test_format_hard_selector_summary(self):
        selectors = {"sources": {"idcard_jj"}, "samples": {"sample_a.jpg", "sample_b.jpg"}}
        assert format_hard_selector_summary(selectors) == "sources=1, samples=2"

    def test_resolve_hard_selector_config_valid(self, tmp_path):
        selector_file = tmp_path / "hard.txt"
        selector_file.write_text("idcard_jj\nsample:foo.jpg\n", encoding="utf-8")
        args = SimpleNamespace(
            hard_selector_file=str(selector_file),
            hard_selector_mix_weight=0.25,
            report_val_hard=True,
        )
        selectors = resolve_hard_selector_config(args)
        assert "idcard_jj" in selectors["sources"]
        assert "foo.jpg" in selectors["samples"]

    def test_resolve_hard_selector_config_requires_file(self):
        args = SimpleNamespace(
            hard_selector_file=None,
            hard_selector_mix_weight=0.25,
            report_val_hard=False,
        )
        with pytest.raises(ValueError, match="requires --hard_selector_file"):
            resolve_hard_selector_config(args)

    def test_resolve_hard_selector_config_requires_file_for_val_report(self):
        args = SimpleNamespace(
            hard_selector_file=None,
            hard_selector_mix_weight=0.0,
            report_val_hard=True,
        )
        with pytest.raises(ValueError, match="requires --hard_selector_file"):
            resolve_hard_selector_config(args)

    def test_resolve_hard_selector_config_rejects_empty_file(self, tmp_path):
        selector_file = tmp_path / "empty.txt"
        selector_file.write_text("# only comments\n", encoding="utf-8")
        args = SimpleNamespace(
            hard_selector_file=str(selector_file),
            hard_selector_mix_weight=0.0,
            report_val_hard=True,
            augment=False,
            augment_selector_only=False,
        )
        with pytest.raises(ValueError, match="did not define any selectors"):
            resolve_hard_selector_config(args)

    def test_resolve_hard_selector_config_requires_file_for_selector_only(self):
        args = SimpleNamespace(
            hard_selector_file=None,
            hard_selector_mix_weight=0.0,
            report_val_hard=False,
            augment=True,
            augment_selector_only=True,
        )
        with pytest.raises(ValueError, match="requires --hard_selector_file"):
            resolve_hard_selector_config(args)

    def test_resolve_hard_selector_config_requires_augment_for_selector_only(self, tmp_path):
        selector_file = tmp_path / "hard.txt"
        selector_file.write_text("idcard_jj\n", encoding="utf-8")
        args = SimpleNamespace(
            hard_selector_file=str(selector_file),
            hard_selector_mix_weight=0.0,
            report_val_hard=False,
            augment=False,
            augment_selector_only=True,
        )
        with pytest.raises(ValueError, match="requires --augment"):
            resolve_hard_selector_config(args)


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

    def test_training_step_with_augmentation(self):
        """§11.9: training loop uses augmentation when active."""
        from dataset import tf_augment_batch, tf_augment_color_only

        net = create_model(backbone_weights=None)
        trainer = DocCornerNetV2Trainer(net, bins=224, sigma_px=2.0, tau=0.5)
        trainer.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=2e-4, weight_decay=1e-4)
        )

        B = 4
        images = np.random.randn(B, 224, 224, 3).astype(np.float32)
        coords = np.random.uniform(0.1, 0.9, (B, 8)).astype(np.float32)
        has_doc = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)

        # Apply augmentation then train — simulates the training loop
        import tensorflow as tf
        images_t = tf.constant(images)
        coords_t = tf.constant(coords)
        has_doc_t = tf.constant(has_doc)

        aug_images, aug_coords = tf_augment_batch(
            images_t, coords_t, has_doc_t,
            rotation_range=5.0, scale_range=0.0,
        )
        targets = {"coords": aug_coords, "has_doc": has_doc_t}
        metrics = trainer.train_step((aug_images, targets))
        assert float(metrics["loss"]) > 0

    def test_validation_without_augmentation(self):
        """§11.10: validation remains without augmentation."""
        net = create_model(backbone_weights=None)
        trainer = DocCornerNetV2Trainer(net, bins=224, sigma_px=2.0, tau=0.5)
        trainer.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=2e-4, weight_decay=1e-4)
        )

        B = 2
        images = np.random.randn(B, 224, 224, 3).astype(np.float32)
        coords = np.random.uniform(0.1, 0.9, (B, 8)).astype(np.float32)
        has_doc = np.array([1.0, 0.0], dtype=np.float32)

        # Validation path: no augmentation, just test_step
        import tensorflow as tf
        targets = {"coords": tf.constant(coords), "has_doc": tf.constant(has_doc)}
        result, _, _ = trainer.test_step((tf.constant(images), targets))
        assert float(result["loss"]) > 0
        # Verify coords pass through unmodified
        np.testing.assert_allclose(
            targets["coords"].numpy(), coords, atol=1e-6)

    def test_validation_targets_can_carry_hard_flag(self):
        N = 4
        images = np.random.randint(0, 255, (N, 32, 32, 3), dtype=np.uint8)
        coords = np.zeros((N, 8), dtype=np.float32)
        has_doc = np.ones(N, dtype=np.float32)
        is_hard = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        ds = make_tf_dataset(
            images, coords, has_doc,
            batch_size=2, shuffle=False,
            is_hard_source=is_hard,
        )
        for _, targets in ds.take(1):
            assert "is_hard_source" in targets
            assert targets["is_hard_source"].shape == (2,)


# ---------------------------------------------------------------------------
# Delayed augmentation activation tests
# ---------------------------------------------------------------------------

class TestDelayedAugmentation:
    """Tests for --aug_start_epoch and --aug_min_iou logic."""

    def test_defaults_immediate_activation(self):
        """aug_start_epoch=1, aug_min_iou=0.0 -> active from epoch 1."""
        assert should_activate_augmentation(epoch=1, best_iou=0.0,
                                            aug_start_epoch=1, aug_min_iou=0.0)

    def test_aug_start_epoch_delays_activation(self):
        """Augmentation inactive before start epoch, active at start epoch."""
        assert not should_activate_augmentation(epoch=1, best_iou=0.0,
                                                aug_start_epoch=3, aug_min_iou=0.0)
        assert not should_activate_augmentation(epoch=2, best_iou=0.5,
                                                aug_start_epoch=3, aug_min_iou=0.0)
        assert should_activate_augmentation(epoch=3, best_iou=0.5,
                                            aug_start_epoch=3, aug_min_iou=0.0)
        assert should_activate_augmentation(epoch=10, best_iou=0.9,
                                            aug_start_epoch=3, aug_min_iou=0.0)

    def test_aug_min_iou_delays_activation(self):
        """Augmentation inactive until best_iou reaches threshold."""
        assert not should_activate_augmentation(epoch=1, best_iou=0.0,
                                                aug_start_epoch=1, aug_min_iou=0.5)
        assert not should_activate_augmentation(epoch=5, best_iou=0.49,
                                                aug_start_epoch=1, aug_min_iou=0.5)
        assert should_activate_augmentation(epoch=5, best_iou=0.5,
                                            aug_start_epoch=1, aug_min_iou=0.5)
        assert should_activate_augmentation(epoch=10, best_iou=0.9,
                                            aug_start_epoch=1, aug_min_iou=0.5)

    def test_and_logic_both_conditions_required(self):
        """Both epoch AND IoU conditions must be met (AND logic)."""
        # Epoch met but IoU not met
        assert not should_activate_augmentation(epoch=5, best_iou=0.3,
                                                aug_start_epoch=3, aug_min_iou=0.5)
        # IoU met but epoch not met
        assert not should_activate_augmentation(epoch=2, best_iou=0.8,
                                                aug_start_epoch=3, aug_min_iou=0.5)
        # Both met
        assert should_activate_augmentation(epoch=5, best_iou=0.8,
                                            aug_start_epoch=3, aug_min_iou=0.5)

    def test_exact_boundary_values(self):
        """Activation at exact boundary (>=, not >)."""
        assert should_activate_augmentation(epoch=3, best_iou=0.5,
                                            aug_start_epoch=3, aug_min_iou=0.5)

    def test_latch_behavior_simulation(self):
        """Once activated, aug_active stays True even if IoU drops.

        This tests the latch pattern used in train_ultra.py:
        aug_active is only set to True, never back to False.
        """
        aug_active = False
        iou_sequence = [0.0, 0.3, 0.6, 0.4, 0.35]  # IoU drops after epoch 3

        for epoch, iou in enumerate(iou_sequence, start=1):
            if not aug_active:
                if should_activate_augmentation(epoch, iou,
                                                aug_start_epoch=1, aug_min_iou=0.5):
                    aug_active = True

        # Should have activated at epoch 3 (iou=0.6 >= 0.5) and stayed active
        assert aug_active

    def test_never_activates_if_iou_stays_low(self):
        """If IoU never reaches threshold, augmentation never activates."""
        aug_active = False
        for epoch in range(1, 11):
            if not aug_active:
                if should_activate_augmentation(epoch, best_iou=0.1,
                                                aug_start_epoch=1, aug_min_iou=0.5):
                    aug_active = True
        assert not aug_active

    def test_weak_aug_interaction(self):
        """Delayed start + weak aug: no_aug -> full_aug -> weak_aug sequence.

        Simulates: epochs=10, aug_start_epoch=3, aug_weak_epochs=2
        Expected: epoch 1-2 no aug, 3-8 full aug, 9-10 weak aug.
        """
        total_epochs = 10
        aug_start_epoch = 3
        aug_weak_epochs = 2
        aug_min_iou = 0.0

        aug_active = False
        phases = []

        for epoch in range(1, total_epochs + 1):
            # Check activation (happens after validation in real code,
            # but for first epoch with aug_min_iou=0 it activates immediately)
            if not aug_active:
                if should_activate_augmentation(epoch, best_iou=0.0,
                                                aug_start_epoch=aug_start_epoch,
                                                aug_min_iou=aug_min_iou):
                    aug_active = True

            use_weak = (aug_active and aug_weak_epochs > 0
                        and epoch >= total_epochs - aug_weak_epochs + 1)

            if not aug_active:
                phases.append("none")
            elif use_weak:
                phases.append("weak")
            else:
                phases.append("full")

        assert phases == [
            "none", "none",                         # epoch 1-2: before start
            "full", "full", "full", "full",         # epoch 3-8: full aug
            "full", "full",
            "weak", "weak",                         # epoch 9-10: weak aug
        ]


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
