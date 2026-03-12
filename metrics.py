"""
Evaluation metrics for DocCornerNet V2.

Contains:
- compute_polygon_iou: True polygon IoU using Shapely
- compute_bbox_iou: Axis-aligned bounding box IoU fallback
- compute_corner_error: Corner error in pixels
- ValidationMetrics: Accumulator for epoch-level metrics
"""

import numpy as np

try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


def coords_to_polygon(coords: np.ndarray) -> "Polygon":
    """Convert 8-value coords [x0,y0,...,x3,y3] (TL,TR,BR,BL) to Shapely Polygon."""
    points = [
        (coords[0], coords[1]),
        (coords[2], coords[3]),
        (coords[4], coords[5]),
        (coords[6], coords[7]),
    ]
    poly = Polygon(points)
    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.geom_type == "GeometryCollection":
            for geom in poly.geoms:
                if geom.geom_type == "Polygon":
                    return geom
            return Polygon(points).convex_hull
        elif poly.geom_type == "MultiPolygon":
            return max(poly.geoms, key=lambda p: p.area)
    return poly


def compute_polygon_iou(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """Compute IoU between predicted and ground truth quadrilaterals."""
    if not SHAPELY_AVAILABLE:
        return compute_bbox_iou(pred_coords, gt_coords)
    try:
        pred_poly = coords_to_polygon(pred_coords)
        gt_poly = coords_to_polygon(gt_coords)
        if pred_poly.is_empty or gt_poly.is_empty:
            return 0.0
        intersection = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area
        if union == 0:
            return 0.0
        return intersection / union
    except Exception:
        return compute_bbox_iou(pred_coords, gt_coords)


def compute_bbox_iou(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """Axis-aligned bounding box IoU fallback."""
    pred_x = pred_coords[0::2]
    pred_y = pred_coords[1::2]
    gt_x = gt_coords[0::2]
    gt_y = gt_coords[1::2]

    pred_bbox = [pred_x.min(), pred_y.min(), pred_x.max(), pred_y.max()]
    gt_bbox = [gt_x.min(), gt_y.min(), gt_x.max(), gt_y.max()]

    x1 = max(pred_bbox[0], gt_bbox[0])
    y1 = max(pred_bbox[1], gt_bbox[1])
    x2 = min(pred_bbox[2], gt_bbox[2])
    y2 = min(pred_bbox[3], gt_bbox[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    union = pred_area + gt_area - intersection

    if union == 0:
        return 0.0
    return intersection / union


def compute_corner_error(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    img_size: int = 224,
) -> tuple:
    """
    Compute corner error in pixels.

    Returns:
        (mean_error_px, per_corner_error_px [4])
    """
    pred_px = pred_coords * img_size
    gt_px = gt_coords * img_size
    pred_corners = pred_px.reshape(4, 2)
    gt_corners = gt_px.reshape(4, 2)
    distances = np.sqrt(((pred_corners - gt_corners) ** 2).sum(axis=1))
    return float(distances.mean()), distances


class ValidationMetrics:
    """Accumulates predictions and computes aggregate metrics."""

    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        self.reset()

    def reset(self):
        self.pred_coords_list = []
        self.gt_coords_list = []
        self.pred_scores_list = []
        self.has_doc_list = []

    def update(
        self,
        pred_coords: np.ndarray,
        gt_coords: np.ndarray,
        pred_scores: np.ndarray,
        has_doc: np.ndarray,
    ):
        if len(pred_scores.shape) == 2:
            pred_scores = pred_scores.squeeze(-1)
        if len(has_doc.shape) == 2:
            has_doc = has_doc.squeeze(-1)
        self.pred_coords_list.append(pred_coords)
        self.gt_coords_list.append(gt_coords)
        self.pred_scores_list.append(pred_scores)
        self.has_doc_list.append(has_doc)

    def compute(self) -> dict:
        pred_coords = np.concatenate(self.pred_coords_list, axis=0)
        gt_coords = np.concatenate(self.gt_coords_list, axis=0)
        pred_scores = np.concatenate(self.pred_scores_list, axis=0)
        has_doc = np.concatenate(self.has_doc_list, axis=0)

        num_samples = len(pred_coords)
        mask = has_doc == 1
        num_with_doc = int(mask.sum())

        results = {
            "mean_iou": 0.0,
            "median_iou": 0.0,
            "corner_error_px": 0.0,
            "corner_error_p95_px": 0.0,
            "corner_error_min_px": 0.0,
            "corner_error_max_px": 0.0,
            "recall_50": 0.0,
            "recall_75": 0.0,
            "recall_90": 0.0,
            "recall_95": 0.0,
            "recall_99": 0.0,
            "num_iou_lt_90": 0,
            "num_iou_lt_95": 0,
            "num_iou_lt_99": 0,
            "num_err_gt_10": 0,
            "num_err_gt_20": 0,
            "num_err_gt_50": 0,
            "cls_accuracy": 0.0,
            "cls_precision": 0.0,
            "cls_recall": 0.0,
            "cls_f1": 0.0,
            "num_samples": num_samples,
            "num_with_doc": num_with_doc,
        }

        # Classification metrics
        pred_labels = (pred_scores > 0.5).astype(int)
        tp = int(((pred_labels == 1) & (has_doc == 1)).sum())
        fp = int(((pred_labels == 1) & (has_doc == 0)).sum())
        tn = int(((pred_labels == 0) & (has_doc == 0)).sum())
        fn = int(((pred_labels == 0) & (has_doc == 1)).sum())

        if num_samples > 0:
            results["cls_accuracy"] = (tp + tn) / num_samples
        if tp + fp > 0:
            results["cls_precision"] = tp / (tp + fp)
        if tp + fn > 0:
            results["cls_recall"] = tp / (tp + fn)
        if results["cls_precision"] + results["cls_recall"] > 0:
            results["cls_f1"] = (
                2 * results["cls_precision"] * results["cls_recall"]
                / (results["cls_precision"] + results["cls_recall"])
            )

        if num_with_doc == 0:
            return results

        pred_coords_pos = pred_coords[mask]
        gt_coords_pos = gt_coords[mask]

        # IoU per sample
        ious = np.zeros(num_with_doc, dtype=np.float64)
        if SHAPELY_AVAILABLE:
            try:
                from shapely import area, intersection, is_empty, is_valid, polygons, union

                pred_pts = pred_coords_pos.reshape(-1, 4, 2).astype(np.float64)
                gt_pts = gt_coords_pos.reshape(-1, 4, 2).astype(np.float64)
                pred_polys = polygons(pred_pts)
                gt_polys = polygons(gt_pts)
                valid = is_valid(pred_polys) & is_valid(gt_polys) & (~is_empty(pred_polys)) & (~is_empty(gt_polys))
                if bool(valid.any()):
                    inter = area(intersection(pred_polys[valid], gt_polys[valid]))
                    uni = area(union(pred_polys[valid], gt_polys[valid]))
                    ious[valid] = np.where(uni > 0, inter / uni, 0.0)
                invalid_idx = np.where(~valid)[0]
                if invalid_idx.size:
                    for i in invalid_idx.tolist():
                        ious[i] = float(compute_polygon_iou(pred_coords_pos[i], gt_coords_pos[i]))
            except Exception:
                for i in range(num_with_doc):
                    ious[i] = float(compute_polygon_iou(pred_coords_pos[i], gt_coords_pos[i]))
        else:
            # Vectorized bbox IoU
            pred_xy = pred_coords_pos.reshape(-1, 4, 2)
            gt_xy = gt_coords_pos.reshape(-1, 4, 2)
            pxmin, pymin = pred_xy[:, :, 0].min(axis=1), pred_xy[:, :, 1].min(axis=1)
            pxmax, pymax = pred_xy[:, :, 0].max(axis=1), pred_xy[:, :, 1].max(axis=1)
            gxmin, gymin = gt_xy[:, :, 0].min(axis=1), gt_xy[:, :, 1].min(axis=1)
            gxmax, gymax = gt_xy[:, :, 0].max(axis=1), gt_xy[:, :, 1].max(axis=1)
            ix1, iy1 = np.maximum(pxmin, gxmin), np.maximum(pymin, gymin)
            ix2, iy2 = np.minimum(pxmax, gxmax), np.minimum(pymax, gymax)
            inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
            area_p = np.maximum(0.0, pxmax - pxmin) * np.maximum(0.0, pymax - pymin)
            area_g = np.maximum(0.0, gxmax - gxmin) * np.maximum(0.0, gymax - gymin)
            uni = area_p + area_g - inter
            ious = np.where(uni > 0, inter / uni, 0.0)

        # Corner error
        pred_px = pred_coords_pos.reshape(-1, 4, 2) * float(self.img_size)
        gt_px = gt_coords_pos.reshape(-1, 4, 2) * float(self.img_size)
        per_corner = np.sqrt(((pred_px - gt_px) ** 2).sum(axis=2))
        mean_corner_errors = per_corner.mean(axis=1)
        all_corner_errors = per_corner.reshape(-1)

        results["mean_iou"] = float(np.mean(ious))
        results["median_iou"] = float(np.median(ious))
        results["corner_error_px"] = float(np.mean(mean_corner_errors))
        results["corner_error_p95_px"] = float(np.percentile(all_corner_errors, 95))
        results["corner_error_min_px"] = float(np.min(all_corner_errors))
        results["corner_error_max_px"] = float(np.max(all_corner_errors))

        results["recall_50"] = float((ious >= 0.50).sum() / num_with_doc)
        results["recall_75"] = float((ious >= 0.75).sum() / num_with_doc)
        results["recall_90"] = float((ious >= 0.90).sum() / num_with_doc)
        results["recall_95"] = float((ious >= 0.95).sum() / num_with_doc)
        results["recall_99"] = float((ious >= 0.99).sum() / num_with_doc)

        results["num_iou_lt_90"] = int((ious < 0.90).sum())
        results["num_iou_lt_95"] = int((ious < 0.95).sum())
        results["num_iou_lt_99"] = int((ious < 0.99).sum())
        results["num_err_gt_10"] = int((mean_corner_errors > 10.0).sum())
        results["num_err_gt_20"] = int((mean_corner_errors > 20.0).sum())
        results["num_err_gt_50"] = int((mean_corner_errors > 50.0).sum())

        return results
