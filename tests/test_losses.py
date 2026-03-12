"""Tests for v2/losses.py — loss functions and trainer."""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from model import create_model
from losses import (
    gaussian_1d_targets,
    SimCCLoss,
    CoordLoss,
    DocCornerNetV2Trainer,
)


class TestGaussian1DTargets:
    def test_shape(self):
        coords = tf.constant([[0.2, 0.5, 0.8, 0.3]], dtype=tf.float32)
        targets = gaussian_1d_targets(coords, bins=224, sigma_px=2.0)
        assert targets.shape == (1, 4, 224)

    def test_normalized(self):
        coords = tf.constant([[0.2, 0.5, 0.8, 0.3]], dtype=tf.float32)
        targets = gaussian_1d_targets(coords, bins=224, sigma_px=2.0)
        sums = tf.reduce_sum(targets, axis=-1).numpy()
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_peak_location(self):
        coords = tf.constant([[0.5, 0.5, 0.5, 0.5]], dtype=tf.float32)
        targets = gaussian_1d_targets(coords, bins=224, sigma_px=2.0)
        # Peak should be near bin 111-112 (center)
        peak_bin = tf.argmax(targets[0, 0]).numpy()
        expected = int(0.5 * 223)
        assert abs(peak_bin - expected) <= 1

    def test_batch(self):
        coords = tf.random.uniform([8, 4], 0.1, 0.9)
        targets = gaussian_1d_targets(coords, bins=224, sigma_px=2.0)
        assert targets.shape == (8, 4, 224)

    def test_label_smoothing(self):
        coords = tf.constant([[0.5, 0.5, 0.5, 0.5]], dtype=tf.float32)
        t_no_smooth = gaussian_1d_targets(coords, bins=224, sigma_px=2.0, label_smoothing=0.0)
        t_smooth = gaussian_1d_targets(coords, bins=224, sigma_px=2.0, label_smoothing=0.1)
        # Smoothed should be more uniform
        assert t_smooth.numpy().min() > t_no_smooth.numpy().min()

    def test_different_bins(self):
        coords = tf.constant([[0.5, 0.5, 0.5, 0.5]], dtype=tf.float32)
        t336 = gaussian_1d_targets(coords, bins=336, sigma_px=2.0)
        assert t336.shape == (1, 4, 336)

    def test_different_sigma(self):
        coords = tf.constant([[0.5, 0.5, 0.5, 0.5]], dtype=tf.float32)
        t_narrow = gaussian_1d_targets(coords, bins=224, sigma_px=1.0)
        t_wide = gaussian_1d_targets(coords, bins=224, sigma_px=4.0)
        # Narrow sigma → more peaked
        assert t_narrow.numpy().max() > t_wide.numpy().max()


class TestSimCCLoss:
    def test_basic(self):
        loss_fn = SimCCLoss(bins=224, sigma_px=2.0, tau=1.0)
        B = 4
        sx = tf.random.normal([B, 4, 224])
        sy = tf.random.normal([B, 4, 224])
        gt = tf.random.uniform([B, 8], 0.1, 0.9)
        mask = tf.ones([B])
        loss = loss_fn(sx, sy, gt, mask)
        assert loss.shape == ()
        assert loss.numpy() > 0

    def test_masked(self):
        loss_fn = SimCCLoss(bins=224, sigma_px=2.0, tau=1.0)
        B = 4
        sx = tf.random.normal([B, 4, 224])
        sy = tf.random.normal([B, 4, 224])
        gt = tf.random.uniform([B, 8], 0.1, 0.9)

        mask_all = tf.ones([B])
        mask_half = tf.constant([1.0, 1.0, 0.0, 0.0])
        loss_all = loss_fn(sx, sy, gt, mask_all)
        loss_half = loss_fn(sx, sy, gt, mask_half)
        # Different masks → different losses
        assert loss_all.numpy() != loss_half.numpy()

    def test_zero_mask(self):
        loss_fn = SimCCLoss(bins=224, sigma_px=2.0, tau=1.0)
        B = 2
        sx = tf.random.normal([B, 4, 224])
        sy = tf.random.normal([B, 4, 224])
        gt = tf.random.uniform([B, 8], 0.1, 0.9)
        mask = tf.zeros([B])
        loss = loss_fn(sx, sy, gt, mask)
        assert loss.numpy() == pytest.approx(0.0, abs=1e-6)

    def test_tau_effect(self):
        loss_fn_low = SimCCLoss(bins=224, sigma_px=2.0, tau=0.1)
        loss_fn_high = SimCCLoss(bins=224, sigma_px=2.0, tau=1.0)
        B = 4
        sx = tf.random.normal([B, 4, 224])
        sy = tf.random.normal([B, 4, 224])
        gt = tf.random.uniform([B, 8], 0.1, 0.9)
        mask = tf.ones([B])
        loss_low = loss_fn_low(sx, sy, gt, mask)
        loss_high = loss_fn_high(sx, sy, gt, mask)
        # Different tau → different loss values
        assert loss_low.numpy() != loss_high.numpy()


class TestCoordLoss:
    def test_l1(self):
        loss_fn = CoordLoss(loss_type="l1")
        B = 4
        pred = tf.random.uniform([B, 8], 0.1, 0.9)
        gt = tf.random.uniform([B, 8], 0.1, 0.9)
        mask = tf.ones([B])
        loss = loss_fn(pred, gt, mask)
        assert loss.shape == ()
        assert loss.numpy() > 0

    def test_smooth_l1(self):
        loss_fn = CoordLoss(loss_type="smooth_l1")
        B = 4
        pred = tf.random.uniform([B, 8], 0.1, 0.9)
        gt = tf.random.uniform([B, 8], 0.1, 0.9)
        mask = tf.ones([B])
        loss = loss_fn(pred, gt, mask)
        assert loss.shape == ()
        assert loss.numpy() > 0

    def test_perfect_prediction(self):
        loss_fn = CoordLoss(loss_type="l1")
        B = 4
        gt = tf.random.uniform([B, 8], 0.1, 0.9)
        mask = tf.ones([B])
        loss = loss_fn(gt, gt, mask)
        assert loss.numpy() == pytest.approx(0.0, abs=1e-6)

    def test_masked(self):
        loss_fn = CoordLoss(loss_type="l1")
        B = 4
        pred = tf.constant(np.full((B, 8), 0.0, dtype=np.float32))
        gt = tf.constant(np.full((B, 8), 1.0, dtype=np.float32))
        mask = tf.constant([1.0, 0.0, 0.0, 0.0])
        loss = loss_fn(pred, gt, mask)
        # Only 1 sample contributes, all coords are 1.0 diff
        assert loss.numpy() == pytest.approx(1.0, abs=1e-5)


class TestDocCornerNetV2Trainer:
    @pytest.fixture
    def trainer(self):
        net = create_model(backbone_weights=None)
        t = DocCornerNetV2Trainer(net, bins=224, sigma_px=2.0, tau=0.5)
        t.compile(optimizer=keras.optimizers.AdamW(learning_rate=2e-4, weight_decay=1e-4))
        return t

    def _make_batch(self, B=4):
        x = np.random.randn(B, 224, 224, 3).astype(np.float32)
        y = {
            "has_doc": np.array([1, 1, 0, 1][:B], dtype=np.float32),
            "coords": np.random.uniform(0.1, 0.9, (B, 8)).astype(np.float32),
        }
        return x, y

    def test_train_step(self, trainer):
        x, y = self._make_batch()
        metrics = trainer.train_step((x, y))
        assert "loss" in metrics
        assert "loss_simcc" in metrics
        assert "loss_coord" in metrics
        assert "loss_score" in metrics
        assert "iou" in metrics
        assert "corner_err_px" in metrics

    def test_test_step(self, trainer):
        x, y = self._make_batch()
        metrics, out_coords, out_score_logit = trainer.test_step((x, y))
        assert "loss" in metrics
        assert out_coords.shape[-1] == 8
        assert out_score_logit.shape[-1] == 1

    def test_call(self, trainer):
        x = np.random.randn(2, 224, 224, 3).astype(np.float32)
        out = trainer(x, training=False)
        assert "coords" in out

    def test_metrics_property(self, trainer):
        assert len(trainer.metrics) == 6

    def test_all_negative_samples(self, trainer):
        B = 4
        x = np.random.randn(B, 224, 224, 3).astype(np.float32)
        y = {
            "has_doc": np.zeros(B, dtype=np.float32),
            "coords": np.zeros((B, 8), dtype=np.float32),
        }
        metrics, _, _ = trainer.test_step((x, y))
        assert "loss" in metrics

    def test_has_doc_2d(self, trainer):
        """has_doc can be [B, 1] instead of [B]."""
        B = 4
        x = np.random.randn(B, 224, 224, 3).astype(np.float32)
        y = {
            "has_doc": np.array([[1], [1], [0], [1]], dtype=np.float32),
            "coords": np.random.uniform(0.1, 0.9, (B, 8)).astype(np.float32),
        }
        metrics, _, _ = trainer.test_step((x, y))
        assert "loss" in metrics

    def test_geometry_metrics_computation(self, trainer):
        """Verify _compute_geometry_metrics works."""
        pred = tf.constant([[0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8]], dtype=tf.float32)
        gt = pred  # Perfect match
        mask = tf.constant([1.0])
        iou, err = trainer._compute_geometry_metrics(pred, gt, mask)
        assert iou.numpy() == pytest.approx(1.0, abs=1e-5)
        assert err.numpy() == pytest.approx(0.0, abs=1e-5)
