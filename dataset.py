"""
TensorFlow Dataset for DocCornerNet V2.

Loads images and labels from:
- images/ directory (positive samples with documents)
- images-negative/ directory (negative samples without documents)
- labels/ directory (YOLO OBB format .txt files)
- Split files (train.txt, val.txt, test.txt)

Output format per sample:
- image: [H, W, 3] float32 (normalized)
- coords: [8] float32 in [0,1]  (x0,y0,x1,y1,x2,y2,x3,y3)
- has_doc: [1] float32  (1.0=positive, 0.0=negative)
"""

import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter, ImageEnhance

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default augmentation config
DEFAULT_AUG_CONFIG = {
    "rotation_degrees": 5,
    "scale_range": (0.9, 1.0),
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.1,
    "blur_prob": 0.1,
    "blur_kernel": 3,
    "translate": 0.0,
    "perspective": (0.0, 0.03),
}


def load_split_file(split_path: str) -> List[str]:
    """Load image names from split file."""
    with open(split_path) as f:
        content = f.read().strip()
    if not content:
        return []
    # Try line-based first
    names = [line.strip() for line in content.split("\n") if line.strip()]
    if len(names) == 1 and ";" in names[0]:
        names = [n.strip() for n in names[0].split(";") if n.strip()]
    return names


def load_label_yolo_obb(label_path: str) -> np.ndarray:
    """Load YOLO OBB label → [8] array (x0,y0,...,x3,y3) normalized."""
    with open(label_path) as f:
        line = f.readline().strip()
    if not line:
        return np.zeros(8, dtype=np.float32)
    parts = line.split()
    if len(parts) >= 9:
        coords = np.array([float(x) for x in parts[1:9]], dtype=np.float32)
        return coords
    return np.zeros(8, dtype=np.float32)


def load_image(image_path: str, img_size: int = 224) -> np.ndarray:
    """Load and resize image to [img_size, img_size, 3] uint8."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((img_size, img_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8).copy()


def augment_sample(
    image: np.ndarray,
    coords: np.ndarray,
    aug_config: Dict = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply augmentation to image and coordinates.

    Args:
        image: [H, W, 3] uint8
        coords: [8] normalized
        aug_config: augmentation config dict

    Returns:
        (augmented_image, augmented_coords)
    """
    if aug_config is None:
        aug_config = DEFAULT_AUG_CONFIG

    img = Image.fromarray(image)

    # Color augmentations
    brightness = aug_config.get("brightness", 0.0)
    if brightness > 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        img = ImageEnhance.Brightness(img).enhance(factor)

    contrast = aug_config.get("contrast", 0.0)
    if contrast > 0:
        factor = 1.0 + random.uniform(-contrast, contrast)
        img = ImageEnhance.Contrast(img).enhance(factor)

    saturation = aug_config.get("saturation", 0.0)
    if saturation > 0:
        factor = 1.0 + random.uniform(-saturation, saturation)
        img = ImageEnhance.Color(img).enhance(factor)

    blur_prob = aug_config.get("blur_prob", 0.0)
    if blur_prob > 0 and random.random() < blur_prob:
        kernel = aug_config.get("blur_kernel", 3)
        img = img.filter(ImageFilter.GaussianBlur(radius=kernel / 2))

    image_out = np.asarray(img, dtype=np.uint8).copy()

    # Geometric augmentation on coordinates
    coords_out = coords.copy()
    rotation_deg = aug_config.get("rotation_degrees", 0)
    if rotation_deg > 0:
        angle = random.uniform(-rotation_deg, rotation_deg)
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        pts = coords_out.reshape(4, 2)
        cx, cy = 0.5, 0.5
        pts_c = pts - [cx, cy]
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        pts_c = pts_c @ rot.T
        pts = pts_c + [cx, cy]
        coords_out = np.clip(pts.reshape(8), 0.0, 1.0).astype(np.float32)

    scale_range = aug_config.get("scale_range", (1.0, 1.0))
    if scale_range != (1.0, 1.0):
        s = random.uniform(scale_range[0], scale_range[1])
        pts = coords_out.reshape(4, 2)
        pts = 0.5 + (pts - 0.5) * s
        coords_out = np.clip(pts.reshape(8), 0.0, 1.0).astype(np.float32)

    return image_out, coords_out


def normalize_image(image: np.ndarray, method: str = "imagenet") -> np.ndarray:
    """Normalize uint8 image to float32."""
    img = image.astype(np.float32)
    if method == "imagenet":
        img = img / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
    elif method == "zero_one":
        img = img / 255.0
    elif method == "raw255":
        pass
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return img.astype(np.float32)


def _load_from_parquet(split_dir, img_size):
    """Load images and labels from parquet files (HuggingFace format)."""
    import io
    import pyarrow.parquet as pq

    table = pq.read_table(split_dir)
    n = table.num_rows

    img_col = table.column("image")
    is_neg_col = table.column("is_negative")
    coord_cols = ["corner_tl_x", "corner_tl_y", "corner_tr_x", "corner_tr_y",
                  "corner_br_x", "corner_br_y", "corner_bl_x", "corner_bl_y"]

    all_images = []
    all_coords = []
    all_has_doc = []

    for i in range(n):
        try:
            img_bytes = img_col[i].as_py()["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((img_size, img_size), Image.BILINEAR)
            all_images.append(np.asarray(img, dtype=np.uint8).copy())

            is_neg = is_neg_col[i].as_py()
            if is_neg:
                all_coords.append(np.zeros(8, dtype=np.float32))
                all_has_doc.append(0.0)
            else:
                c = np.array([table.column(cn)[i].as_py() for cn in coord_cols], dtype=np.float32)
                all_coords.append(c)
                all_has_doc.append(1.0 if c.sum() > 0 else 0.0)
        except Exception:
            continue

    return all_images, all_coords, all_has_doc


def _load_from_files(data_root, split, img_size, negative_ratio):
    """Load images and labels from file-based directory structure."""
    image_dir = data_root / "images"
    negative_dir = data_root / "images-negative"
    label_dir = data_root / "labels"

    split_file = data_root / f"{split}.txt"
    if not split_file.exists():
        for suffix in ["_with_negative_v2", "_with_negative"]:
            candidate = data_root / f"{split}{suffix}.txt"
            if candidate.exists():
                split_file = candidate
                break
    if not split_file.exists():
        raise FileNotFoundError(f"No split file for '{split}' in {data_root}")

    image_names = load_split_file(str(split_file))

    positive_names = []
    negative_names = []
    for name in image_names:
        if name.startswith("negative_"):
            negative_names.append(name)
        else:
            positive_names.append(name)

    all_images = []
    all_coords = []
    all_has_doc = []

    for name in positive_names:
        img_path = image_dir / name
        if not img_path.exists():
            continue
        label_path = label_dir / f"{Path(name).stem}.txt"
        coords = np.zeros(8, dtype=np.float32)
        has_doc = 0.0
        if label_path.exists():
            coords = load_label_yolo_obb(str(label_path))
            if coords.sum() > 0:
                has_doc = 1.0
        img = load_image(str(img_path), img_size)
        all_images.append(img)
        all_coords.append(coords)
        all_has_doc.append(has_doc)

    if negative_ratio > 0 and negative_dir.exists():
        n_neg_target = int(len(positive_names) * negative_ratio / (1.0 - negative_ratio))
        neg_available = negative_names
        if not neg_available:
            neg_available = [f.name for f in negative_dir.iterdir() if f.suffix in (".jpg", ".jpeg", ".png")]
        if neg_available:
            if len(neg_available) > n_neg_target:
                neg_available = random.sample(neg_available, n_neg_target)
            for name in neg_available:
                img_path = negative_dir / name
                if not img_path.exists():
                    continue
                img = load_image(str(img_path), img_size)
                all_images.append(img)
                all_coords.append(np.zeros(8, dtype=np.float32))
                all_has_doc.append(0.0)

    return all_images, all_coords, all_has_doc


def create_dataset(
    data_root: str,
    split: str = "train",
    img_size: int = 224,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = True,
    drop_remainder: bool = False,
    image_norm: str = "imagenet",
    negative_ratio: float = 0.3,
    num_workers: int = 4,
) -> tf.data.Dataset:
    """
    Create tf.data.Dataset from local directory.

    Args:
        data_root: Root dataset directory
        split: Split name (train, val, test)
        img_size: Target image size
        batch_size: Batch size
        shuffle: Whether to shuffle
        augment: Apply augmentation (training only)
        drop_remainder: Drop incomplete final batch
        image_norm: Normalization ('imagenet', 'zero_one', 'raw255')
        negative_ratio: Fraction of negative samples (0 to disable)
        num_workers: Number of parallel workers

    Returns:
        tf.data.Dataset yielding (images, {"coords": ..., "has_doc": ...})
    """
    data_root = Path(data_root)

    # Check for parquet format first
    split_dir = data_root / split
    if split_dir.is_dir() and any(split_dir.glob("*.parquet")):
        all_images, all_coords, all_has_doc = _load_from_parquet(
            split_dir, img_size,
        )
    else:
        all_images, all_coords, all_has_doc = _load_from_files(
            data_root, split, img_size, negative_ratio,
        )

    if not all_images:
        raise ValueError(f"No valid images found for split '{split}' in {data_root}")

    images_np = np.stack(all_images)
    coords_np = np.stack(all_coords)
    has_doc_np = np.array(all_has_doc, dtype=np.float32)

    # Apply augmentation and normalization
    def _process_sample(idx):
        img = images_np[idx]
        coords = coords_np[idx]
        if augment and has_doc_np[idx] > 0:
            img, coords = augment_sample(img, coords)
        img = normalize_image(img, image_norm)
        return img, coords, has_doc_np[idx]

    # Pre-process all
    processed_images = np.empty((len(all_images), img_size, img_size, 3), dtype=np.float32)
    processed_coords = coords_np.copy()
    for i in range(len(all_images)):
        img, coords, _ = _process_sample(i)
        processed_images[i] = img
        processed_coords[i] = coords

    # Build tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        processed_images,
        {"coords": processed_coords, "has_doc": has_doc_np},
    ))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(all_images), 10000))

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
