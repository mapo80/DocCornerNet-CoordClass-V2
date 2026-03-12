"""
Loss functions for DocCornerNet V2.

Contains:
- gaussian_1d_targets: Generate 1D Gaussian target distributions
- SimCCLoss: Cross-entropy loss with soft Gaussian targets
- CoordLoss: Direct coordinate supervision (L1/SmoothL1)
- DocCornerNetV2Trainer: Custom training wrapper with proper loss masking
"""

import tensorflow as tf
from tensorflow import keras


def gaussian_1d_targets(coords_01, bins=224, sigma_px=2.0, label_smoothing=0.0):
    """
    Generate 1D Gaussian target distributions for SimCC.

    Args:
        coords_01: [B, 4] coordinates in [0, 1] for one axis
        bins: Number of bins
        sigma_px: Gaussian sigma in pixel space
        label_smoothing: Blend factor with uniform distribution

    Returns:
        targets: [B, 4, bins] normalized Gaussian distributions
    """
    coords_px = coords_01 * tf.cast(bins - 1, tf.float32)
    bin_indices = tf.cast(tf.range(bins), tf.float32)
    diff = bin_indices[None, None, :] - coords_px[:, :, None]
    gauss = tf.exp(-0.5 * tf.square(diff) / (sigma_px * sigma_px))
    gauss = gauss / (tf.reduce_sum(gauss, axis=-1, keepdims=True) + 1e-9)

    if label_smoothing > 0.0:
        uniform = tf.ones_like(gauss) / tf.cast(bins, tf.float32)
        gauss = (1.0 - label_smoothing) * gauss + label_smoothing * uniform

    return gauss


class SimCCLoss(keras.layers.Layer):
    """Cross-entropy loss with soft Gaussian targets for SimCC logits."""

    def __init__(self, bins=224, sigma_px=2.0, tau=1.0, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.bins = bins
        self.sigma_px = sigma_px
        self.tau = tau
        self.label_smoothing = label_smoothing

    def call(self, simcc_x, simcc_y, gt_coords, mask):
        """
        Args:
            simcc_x: [B, 4, bins] predicted X logits
            simcc_y: [B, 4, bins] predicted Y logits
            gt_coords: [B, 8] ground truth (x0,y0,x1,y1,...)
            mask: [B] float, 1=positive, 0=negative
        Returns:
            loss: scalar
        """
        gt_4x2 = tf.reshape(gt_coords, [-1, 4, 2])
        gt_x = gt_4x2[:, :, 0]
        gt_y = gt_4x2[:, :, 1]

        target_x = gaussian_1d_targets(gt_x, self.bins, self.sigma_px, self.label_smoothing)
        target_y = gaussian_1d_targets(gt_y, self.bins, self.sigma_px, self.label_smoothing)

        log_pred_x = tf.nn.log_softmax(simcc_x / self.tau, axis=-1)
        log_pred_y = tf.nn.log_softmax(simcc_y / self.tau, axis=-1)

        ce_x = -tf.reduce_sum(target_x * log_pred_x, axis=-1)  # [B, 4]
        ce_y = -tf.reduce_sum(target_y * log_pred_y, axis=-1)

        ce = tf.reduce_mean(ce_x + ce_y, axis=-1)  # [B]
        loss = tf.reduce_sum(ce * mask) / (tf.reduce_sum(mask) + 1e-9)
        return loss


class CoordLoss(keras.layers.Layer):
    """Direct coordinate loss (L1 or SmoothL1)."""

    def __init__(self, loss_type="l1", **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type

    def call(self, pred_coords, gt_coords, mask):
        """
        Args:
            pred_coords: [B, 8]
            gt_coords: [B, 8]
            mask: [B] float
        Returns:
            loss: scalar
        """
        if self.loss_type == "smooth_l1":
            diff = tf.abs(pred_coords - gt_coords)
            beta = 0.01
            loss_per_coord = tf.where(
                diff < beta,
                0.5 * tf.square(diff) / beta,
                diff - 0.5 * beta,
            )
        else:
            loss_per_coord = tf.abs(pred_coords - gt_coords)

        loss_per_sample = tf.reduce_mean(loss_per_coord, axis=-1)
        loss = tf.reduce_sum(loss_per_sample * mask) / (tf.reduce_sum(mask) + 1e-9)
        return loss


class DocCornerNetV2Trainer(keras.Model):
    """
    Custom training wrapper for DocCornerNet V2.

    Handles:
    - SimCC loss on X/Y marginals (only positive samples)
    - Coordinate loss (only positive samples)
    - Score BCE loss (all samples)

    Input format:
    - x: images [B, H, W, 3]
    - y: dict with "has_doc" [B, 1] or [B] and "coords" [B, 8]
    """

    def __init__(
        self,
        net,
        bins=224,
        sigma_px=2.0,
        tau=1.0,
        w_simcc=1.0,
        w_coord=0.2,
        w_score=1.0,
        label_smoothing=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.net = net
        self.bins = bins
        self.sigma_px = sigma_px
        self.tau = tau
        self.w_simcc = w_simcc
        self.w_coord = w_coord
        self.w_score = w_score
        self.label_smoothing = label_smoothing

        self.simcc_loss_fn = SimCCLoss(
            bins=bins, sigma_px=sigma_px, tau=tau, label_smoothing=label_smoothing,
        )
        self.coord_loss_fn = CoordLoss(loss_type="l1")

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.simcc_loss_tracker = keras.metrics.Mean(name="loss_simcc")
        self.coord_loss_tracker = keras.metrics.Mean(name="loss_coord")
        self.score_loss_tracker = keras.metrics.Mean(name="loss_score")
        self.iou_tracker = keras.metrics.Mean(name="iou")
        self.corner_err_tracker = keras.metrics.Mean(name="corner_err_px")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.simcc_loss_tracker,
            self.coord_loss_tracker,
            self.score_loss_tracker,
            self.iou_tracker,
            self.corner_err_tracker,
        ]

    def call(self, inputs, training=False):
        return self.net(inputs, training=training)

    def _extract_targets(self, y):
        has_doc = tf.cast(y["has_doc"], tf.float32)
        if len(has_doc.shape) == 2:
            has_doc = tf.squeeze(has_doc, axis=-1)
        coords_gt = tf.cast(y["coords"], tf.float32)
        return has_doc, coords_gt

    def _compute_losses(self, outputs, has_doc, coords_gt):
        simcc_x = outputs["simcc_x"]
        simcc_y = outputs["simcc_y"]
        score_logit = outputs["score_logit"]
        coords_pred = outputs["coords"]

        loss_score = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=has_doc[:, None], logits=score_logit,
        )
        loss_score = tf.reduce_mean(loss_score)

        loss_simcc = self.simcc_loss_fn(simcc_x, simcc_y, coords_gt, has_doc)
        loss_coord = self.coord_loss_fn(coords_pred, coords_gt, has_doc)

        loss = (
            self.w_simcc * loss_simcc
            + self.w_coord * loss_coord
            + self.w_score * loss_score
        )
        return loss, loss_simcc, loss_coord, loss_score, coords_pred

    def _update_metrics(self, loss, loss_simcc, loss_coord, loss_score, coords_pred, coords_gt, has_doc):
        self.loss_tracker.update_state(loss)
        self.simcc_loss_tracker.update_state(loss_simcc)
        self.coord_loss_tracker.update_state(loss_coord)
        self.score_loss_tracker.update_state(loss_score)

        iou, corner_err = self._compute_geometry_metrics(coords_pred, coords_gt, has_doc)
        self.iou_tracker.update_state(iou)
        self.corner_err_tracker.update_state(corner_err)

    def _get_metrics_dict(self):
        return {
            "loss": self.loss_tracker.result(),
            "loss_simcc": self.simcc_loss_tracker.result(),
            "loss_coord": self.coord_loss_tracker.result(),
            "loss_score": self.score_loss_tracker.result(),
            "iou": self.iou_tracker.result(),
            "corner_err_px": self.corner_err_tracker.result(),
        }

    def train_step(self, data):
        x, y = data
        has_doc, coords_gt = self._extract_targets(y)

        with tf.GradientTape() as tape:
            outputs = self.net(x, training=True)
            loss, loss_simcc, loss_coord, loss_score, coords_pred = self._compute_losses(
                outputs, has_doc, coords_gt,
            )

        grads = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))

        self._update_metrics(loss, loss_simcc, loss_coord, loss_score, coords_pred, coords_gt, has_doc)
        return self._get_metrics_dict()

    def test_step(self, data):
        x, y = data
        has_doc, coords_gt = self._extract_targets(y)

        outputs = self.net(x, training=False)
        loss, loss_simcc, loss_coord, loss_score, coords_pred = self._compute_losses(
            outputs, has_doc, coords_gt,
        )

        self._update_metrics(loss, loss_simcc, loss_coord, loss_score, coords_pred, coords_gt, has_doc)
        return self._get_metrics_dict()

    def _compute_geometry_metrics(self, pred_coords, gt_coords, mask):
        """Simplified bbox IoU and corner error for positive samples."""
        img_size = 224.0
        mask_bool = tf.cast(mask, tf.bool)
        pred_pos = tf.boolean_mask(pred_coords, mask_bool)
        gt_pos = tf.boolean_mask(gt_coords, mask_bool)
        n_pos = tf.shape(pred_pos)[0]

        def compute_metrics():
            diff = tf.abs(pred_pos - gt_pos) * img_size
            corner_err = tf.reduce_mean(diff)

            pred_xy = tf.reshape(pred_pos, [-1, 4, 2])
            gt_xy = tf.reshape(gt_pos, [-1, 4, 2])

            pred_min = tf.reduce_min(pred_xy, axis=1)
            pred_max = tf.reduce_max(pred_xy, axis=1)
            gt_min = tf.reduce_min(gt_xy, axis=1)
            gt_max = tf.reduce_max(gt_xy, axis=1)

            inter_min = tf.maximum(pred_min, gt_min)
            inter_max = tf.minimum(pred_max, gt_max)
            inter_wh = tf.maximum(inter_max - inter_min, 0.0)
            inter_area = inter_wh[:, 0] * inter_wh[:, 1]

            pred_area = (pred_max - pred_min)[:, 0] * (pred_max - pred_min)[:, 1]
            gt_area = (gt_max - gt_min)[:, 0] * (gt_max - gt_min)[:, 1]

            union_area = pred_area + gt_area - inter_area + 1e-9
            iou = inter_area / union_area
            mean_iou = tf.reduce_mean(iou)
            return mean_iou, corner_err

        def zero_metrics():
            return tf.constant(0.0), tf.constant(0.0)

        return tf.cond(n_pos > 0, compute_metrics, zero_metrics)
