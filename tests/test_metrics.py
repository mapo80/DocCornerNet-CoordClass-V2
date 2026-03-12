"""Tests for v2/metrics.py — evaluation metrics."""

import numpy as np
import pytest

from metrics import (
    ValidationMetrics,
    compute_bbox_iou,
    compute_corner_error,
    compute_polygon_iou,
    coords_to_polygon,
    SHAPELY_AVAILABLE,
)


class TestComputeBboxIoU:
    def test_perfect_match(self):
        coords = np.array([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8])
        iou = compute_bbox_iou(coords, coords)
        assert iou == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap(self):
        c1 = np.array([0.0, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1])
        c2 = np.array([0.5, 0.5, 0.6, 0.5, 0.6, 0.6, 0.5, 0.6])
        iou = compute_bbox_iou(c1, c2)
        assert iou == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self):
        c1 = np.array([0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5])
        c2 = np.array([0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75])
        iou = compute_bbox_iou(c1, c2)
        assert 0.0 < iou < 1.0

    def test_zero_area(self):
        c1 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        c2 = np.array([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8])
        iou = compute_bbox_iou(c1, c2)
        assert iou == 0.0


class TestComputePolygonIoU:
    def test_perfect_match(self):
        coords = np.array([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8])
        iou = compute_polygon_iou(coords, coords)
        assert iou == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self):
        c1 = np.array([0.0, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1])
        c2 = np.array([0.5, 0.5, 0.6, 0.5, 0.6, 0.6, 0.5, 0.6])
        iou = compute_polygon_iou(c1, c2)
        assert iou == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self):
        c1 = np.array([0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5])
        c2 = np.array([0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75])
        iou = compute_polygon_iou(c1, c2)
        assert 0.0 < iou < 1.0


@pytest.mark.skipif(not SHAPELY_AVAILABLE, reason="Shapely not installed")
class TestCoordsToPolygon:
    def test_valid_polygon(self):
        coords = np.array([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8])
        poly = coords_to_polygon(coords)
        assert poly.is_valid
        assert poly.area > 0

    def test_self_intersecting_polygon(self):
        # Bowtie shape (self-intersecting)
        coords = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
        poly = coords_to_polygon(coords)
        assert poly.area > 0  # Should be made valid


class TestComputeCornerError:
    def test_zero_error(self):
        coords = np.array([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8])
        mean_err, per_corner = compute_corner_error(coords, coords, img_size=224)
        assert mean_err == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(per_corner, 0.0, atol=1e-6)

    def test_known_error(self):
        c1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        c2 = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        mean_err, per_corner = compute_corner_error(c1, c2, img_size=224)
        # Each corner: sqrt((224)^2 + 0^2) = 224
        assert mean_err == pytest.approx(224.0, abs=0.01)

    def test_per_corner_shape(self):
        c1 = np.random.uniform(0, 1, 8)
        c2 = np.random.uniform(0, 1, 8)
        mean_err, per_corner = compute_corner_error(c1, c2)
        assert per_corner.shape == (4,)


class TestValidationMetrics:
    def test_perfect_predictions(self):
        metrics = ValidationMetrics(img_size=224)
        B = 10
        coords = np.random.uniform(0.2, 0.8, (B, 8)).astype(np.float32)
        scores = np.ones(B, dtype=np.float32)
        has_doc = np.ones(B, dtype=np.float32)
        metrics.update(coords, coords, scores, has_doc)
        results = metrics.compute()

        assert results["mean_iou"] == pytest.approx(1.0, abs=1e-4)
        assert results["corner_error_px"] == pytest.approx(0.0, abs=1e-4)
        assert results["recall_99"] == pytest.approx(1.0, abs=1e-4)
        assert results["num_with_doc"] == B

    def test_classification_metrics(self):
        metrics = ValidationMetrics(img_size=224)
        B = 4
        coords = np.random.uniform(0.2, 0.8, (B, 8)).astype(np.float32)
        scores = np.array([0.9, 0.8, 0.1, 0.2], dtype=np.float32)
        has_doc = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        metrics.update(coords, coords, scores, has_doc)
        results = metrics.compute()

        assert results["cls_accuracy"] == pytest.approx(1.0)
        assert results["cls_precision"] == pytest.approx(1.0)
        assert results["cls_recall"] == pytest.approx(1.0)
        assert results["cls_f1"] == pytest.approx(1.0)

    def test_no_positive_samples(self):
        metrics = ValidationMetrics(img_size=224)
        B = 4
        coords = np.zeros((B, 8), dtype=np.float32)
        scores = np.array([0.1, 0.2, 0.1, 0.2], dtype=np.float32)
        has_doc = np.zeros(B, dtype=np.float32)
        metrics.update(coords, coords, scores, has_doc)
        results = metrics.compute()
        assert results["mean_iou"] == 0.0
        assert results["num_with_doc"] == 0

    def test_multiple_updates(self):
        metrics = ValidationMetrics(img_size=224)
        for _ in range(3):
            B = 5
            coords = np.random.uniform(0.2, 0.8, (B, 8)).astype(np.float32)
            scores = np.ones(B, dtype=np.float32)
            has_doc = np.ones(B, dtype=np.float32)
            metrics.update(coords, coords, scores, has_doc)
        results = metrics.compute()
        assert results["num_samples"] == 15
        assert results["num_with_doc"] == 15

    def test_reset(self):
        metrics = ValidationMetrics(img_size=224)
        coords = np.random.uniform(0.2, 0.8, (5, 8)).astype(np.float32)
        scores = np.ones(5, dtype=np.float32)
        has_doc = np.ones(5, dtype=np.float32)
        metrics.update(coords, coords, scores, has_doc)
        metrics.reset()
        metrics.update(coords[:2], coords[:2], scores[:2], has_doc[:2])
        results = metrics.compute()
        assert results["num_samples"] == 2

    def test_2d_scores_and_has_doc(self):
        """Handles [B, 1] shaped inputs."""
        metrics = ValidationMetrics(img_size=224)
        B = 4
        coords = np.random.uniform(0.2, 0.8, (B, 8)).astype(np.float32)
        scores = np.ones((B, 1), dtype=np.float32)
        has_doc = np.ones((B, 1), dtype=np.float32)
        metrics.update(coords, coords, scores, has_doc)
        results = metrics.compute()
        assert results["num_with_doc"] == B

    def test_all_metric_keys_present(self):
        metrics = ValidationMetrics(img_size=224)
        B = 4
        coords = np.random.uniform(0.2, 0.8, (B, 8)).astype(np.float32)
        scores = np.ones(B, dtype=np.float32)
        has_doc = np.ones(B, dtype=np.float32)
        metrics.update(coords, coords, scores, has_doc)
        results = metrics.compute()

        expected_keys = [
            "mean_iou", "median_iou",
            "corner_error_px", "corner_error_p95_px",
            "corner_error_min_px", "corner_error_max_px",
            "recall_50", "recall_75", "recall_90", "recall_95", "recall_99",
            "num_iou_lt_90", "num_iou_lt_95", "num_iou_lt_99",
            "num_err_gt_10", "num_err_gt_20", "num_err_gt_50",
            "cls_accuracy", "cls_precision", "cls_recall", "cls_f1",
            "num_samples", "num_with_doc",
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
