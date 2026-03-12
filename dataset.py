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

# Default augmentation config (photometric only — geometric handled by tf_augment_batch)
DEFAULT_AUG_CONFIG = {
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.1,
    "blur_prob": 0.1,
    "blur_kernel": 3,
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

    # Geometric augmentation handled by tf_augment_batch() in training loop
    return image_out, coords.copy()


# ---------------------------------------------------------------------------
# TensorFlow batch augmentation (GPU-accelerated)
# ---------------------------------------------------------------------------

def _tf_rotate_batch(images, coords, has_doc, rotation_range):
    """Apply random rotation to a batch using TF ops.

    Args:
        images: [B, H, W, 3] float32
        coords: [B, 8] float32 normalized [0,1]
        has_doc: [B] float32
        rotation_range: max rotation in degrees

    Returns:
        rotated_images, rotated_coords
    """
    batch_size = tf.shape(images)[0]
    h = tf.cast(tf.shape(images)[1], tf.float32)
    w = tf.cast(tf.shape(images)[2], tf.float32)

    # Random angles in radians
    angles = tf.random.uniform([batch_size], -rotation_range, rotation_range) * (3.14159265 / 180.0)
    cos_a = tf.cos(angles)
    sin_a = tf.sin(angles)

    # Center of image (aligned with normalized coord convention)
    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    # Inverse affine transform for ImageProjectiveTransformV3
    tx = cx - cos_a * cx - sin_a * cy
    ty = cy + sin_a * cx - cos_a * cy

    transforms = tf.stack([
        cos_a, sin_a, tx,
        -sin_a, cos_a, ty,
        tf.zeros([batch_size]), tf.zeros([batch_size])
    ], axis=1)

    images = tf.raw_ops.ImageProjectiveTransformV3(
        images=images, transforms=transforms,
        output_shape=tf.shape(images)[1:3],
        interpolation="BILINEAR", fill_mode="NEAREST", fill_value=0.0)

    # Forward transform for coordinates (normalized [0,1])
    coords_4x2 = tf.reshape(coords, [-1, 4, 2])
    x = coords_4x2[:, :, 0] - 0.5
    y = coords_4x2[:, :, 1] - 0.5

    new_x = cos_a[:, None] * x - sin_a[:, None] * y + 0.5
    new_y = sin_a[:, None] * x + cos_a[:, None] * y + 0.5

    new_coords = tf.stack([new_x, new_y], axis=-1)
    new_coords = tf.reshape(new_coords, [-1, 8])

    # Only rotate positive samples' coords
    has_doc_mask = tf.cast(tf.reshape(has_doc > 0.5, [-1, 1]), tf.float32)
    coords = coords * (1.0 - has_doc_mask) + new_coords * has_doc_mask
    coords = tf.clip_by_value(coords, 0.0, 1.0)

    return images, coords


def _tf_scale_batch(images, coords, has_doc, scale_range):
    """Apply random scale augmentation to batch.

    Scale range 0.15 means uniform scale in [0.85, 1.15].
    Zoom-in (>1): center crop + resize. Zoom-out (<1): shrink + pad.
    Coordinates: coords_new = 0.5 + (coords - 0.5) / scale.
    Samples where any coord goes outside [0,1] are left unchanged.
    """
    batch_size = tf.shape(images)[0]
    h = tf.cast(tf.shape(images)[1], tf.float32)
    w = tf.cast(tf.shape(images)[2], tf.float32)

    # Random scale per sample
    scales = tf.random.uniform([batch_size], 1.0 - scale_range, 1.0 + scale_range)

    has_doc_1d = tf.cast(tf.reshape(has_doc > 0.5, [batch_size]), tf.float32)

    # Transform coordinates: zoom around center
    coords_4x2 = tf.reshape(coords, [-1, 4, 2])
    new_coords_4x2 = 0.5 + (coords_4x2 - 0.5) / scales[:, None, None]
    new_coords = tf.reshape(new_coords_4x2, [-1, 8])

    # Check if any coord goes OOB — if so, skip transform for that sample
    coords_min = tf.reduce_min(new_coords_4x2, axis=[1, 2])
    coords_max = tf.reduce_max(new_coords_4x2, axis=[1, 2])
    valid = tf.cast((coords_min >= 0.0) & (coords_max <= 1.0), tf.float32)
    apply_mask = valid * has_doc_1d

    # Transform images using crop-and-resize
    half_h = 0.5 / scales
    half_w = 0.5 / scales
    y1 = 0.5 - half_h
    x1 = 0.5 - half_w
    y2 = 0.5 + half_h
    x2 = 0.5 + half_w
    boxes = tf.stack([y1, x1, y2, x2], axis=1)
    box_indices = tf.range(batch_size)
    crop_size = tf.cast([h, w], tf.int32)

    images_scaled = tf.image.crop_and_resize(
        images, boxes, box_indices, crop_size, method='bilinear')

    # Apply mask: use scaled version only for valid samples
    apply_mask_img = tf.reshape(apply_mask, [batch_size, 1, 1, 1])
    images = images * (1.0 - apply_mask_img) + images_scaled * apply_mask_img

    apply_mask_coord = tf.reshape(apply_mask, [batch_size, 1])
    coords = coords * (1.0 - apply_mask_coord) + new_coords * apply_mask_coord
    coords = tf.clip_by_value(coords, 0.0, 1.0)

    return images, coords


def tf_augment_batch(images, coords, has_doc, img_size=224,
                     image_norm="imagenet", rotation_range=5.0,
                     scale_range=0.0):
    """Apply augmentation to a batch using TensorFlow ops (GPU-accelerated).

    Args:
        images: [B, H, W, 3] float32 tensor (already normalized)
        coords: [B, 8] float32 tensor, normalized [0,1]
        has_doc: [B] float32 tensor
        img_size: Image size
        image_norm: Normalization mode ("imagenet", "zero_one", "raw255")
        rotation_range: Max rotation in degrees (0.0 = disabled)
        scale_range: Scale range (0.15 means 0.85x-1.15x, 0.0 = disabled)

    Returns:
        augmented_images: [B, H, W, 3] float32
        augmented_coords: [B, 8] float32
    """
    batch_size = tf.shape(images)[0]

    norm_mode = (image_norm or "imagenet").lower().strip()
    if norm_mode in {"zero_one", "0_1", "01"}:
        clip_min, clip_max = 0.0, 1.0
        brightness_scale = 1.0
    elif norm_mode in {"raw255", "0_255", "0255"}:
        clip_min, clip_max = 0.0, 255.0
        brightness_scale = 255.0
    else:
        # "imagenet" (or unknown): assume roughly standardized inputs
        clip_min, clip_max = -3.0, 3.0
        brightness_scale = 1.0

    # Augmentation strengths (fixed for MVP — no outlier support)
    brightness_range = tf.fill([batch_size], 0.2)
    contrast_delta = tf.fill([batch_size], 0.2)
    sat_delta = tf.fill([batch_size], 0.15)

    # --- Photometric augmentations (all samples) ---

    # Brightness
    brightness_delta = tf.random.uniform([batch_size], -1.0, 1.0) * brightness_range * brightness_scale
    brightness_delta = tf.reshape(brightness_delta, [batch_size, 1, 1, 1])
    images = images + brightness_delta

    # Contrast
    contrast_min = 1.0 - contrast_delta
    contrast_max = 1.0 + contrast_delta
    contrast_factor = tf.random.uniform([batch_size]) * (contrast_max - contrast_min) + contrast_min
    contrast_factor = tf.reshape(contrast_factor, [batch_size, 1, 1, 1])
    mean = tf.reduce_mean(images, axis=[1, 2, 3], keepdims=True)
    images = (images - mean) * contrast_factor + mean

    # Saturation
    sat_min = 1.0 - sat_delta
    sat_max = 1.0 + sat_delta
    sat_factor = tf.random.uniform([batch_size]) * (sat_max - sat_min) + sat_min
    sat_factor = tf.reshape(sat_factor, [batch_size, 1, 1, 1])
    gray = tf.reduce_mean(images, axis=-1, keepdims=True)
    gray = tf.tile(gray, [1, 1, 1, 3])
    images = gray + sat_factor * (images - gray)

    # --- Horizontal flip (50%) ---
    flip_mask = tf.random.uniform([batch_size]) > 0.5
    flip_mask_img = tf.reshape(flip_mask, [batch_size, 1, 1, 1])

    images_flipped = tf.reverse(images, axis=[2])
    images = tf.where(flip_mask_img, images_flipped, images)

    # Flip coordinates: TL<->TR, BL<->BR, x -> 1-x (positive samples only)
    flip_mask_coord = tf.cast(tf.reshape(flip_mask, [batch_size, 1]), tf.float32)
    has_doc_mask = tf.cast(tf.reshape(has_doc > 0.5, [batch_size, 1]), tf.float32)
    should_flip_coords = flip_mask_coord * has_doc_mask

    x0, y0, x1, y1, x2, y2, x3, y3 = tf.unstack(coords, axis=1)
    coords_flipped = tf.stack([
        1.0 - x1, y1,  # new TL = old TR with flipped x
        1.0 - x0, y0,  # new TR = old TL with flipped x
        1.0 - x3, y3,  # new BR = old BL with flipped x
        1.0 - x2, y2,  # new BL = old BR with flipped x
    ], axis=1)

    coords = coords * (1.0 - should_flip_coords) + coords_flipped * should_flip_coords

    # Clip values
    images = tf.clip_by_value(images, clip_min, clip_max)
    coords = tf.clip_by_value(coords, 0.0, 1.0)

    # --- Geometric augmentations ---
    if rotation_range > 0:
        images, coords = _tf_rotate_batch(images, coords, has_doc, rotation_range)
    if scale_range > 0:
        images, coords = _tf_scale_batch(images, coords, has_doc, scale_range)

    return images, coords


def tf_augment_color_only(images, image_norm="imagenet"):
    """Apply color-only augmentation (no geometric transforms).

    Args:
        images: [B, H, W, 3] float32 tensor (normalized)
        image_norm: Normalization mode for clipping

    Returns:
        augmented_images: [B, H, W, 3] float32
    """
    images = tf.image.random_brightness(images, max_delta=0.15)
    images = tf.image.random_contrast(images, lower=0.85, upper=1.15)
    images = tf.image.random_saturation(images, lower=0.85, upper=1.15)

    norm_mode = (image_norm or "imagenet").lower().strip()
    if norm_mode in {"zero_one", "0_1", "01"}:
        images = tf.clip_by_value(images, 0.0, 1.0)
    elif norm_mode in {"raw255", "0_255", "0255"}:
        images = tf.clip_by_value(images, 0.0, 255.0)
    else:
        images = tf.clip_by_value(images, -3.0, 3.0)

    return images


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
