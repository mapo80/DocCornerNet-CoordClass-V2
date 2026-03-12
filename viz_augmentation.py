"""Visualize geometric augmentation on dataset samples.

Creates a collage showing original vs augmented images with corner overlays.

Usage:
    cd v2
    python viz_augmentation.py \
        --data_root ../dataset/DocCornerDataset_small \
        --split val \
        --num_samples 16 \
        --num_aug 3 \
        --rotation_range 5.0 \
        --scale_range 0.15 \
        --output augmentation_collage.png
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    tf_augment_batch,
    _tf_rotate_batch,
    _tf_scale_batch,
)

# Corner colors: TL=red, TR=green, BR=blue, BL=yellow
CORNER_COLORS = ["red", "lime", "blue", "yellow"]
CORNER_NAMES = ["TL", "TR", "BR", "BL"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize geometric augmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_samples", type=int, default=16,
                   help="Number of samples to show")
    p.add_argument("--num_aug", type=int, default=3,
                   help="Number of augmentation variants per sample")
    p.add_argument("--rotation_range", type=float, default=5.0)
    p.add_argument("--scale_range", type=float, default=0.15)
    p.add_argument("--output", type=str, default="augmentation_collage.png")
    p.add_argument("--cell_size", type=int, default=256,
                   help="Size of each cell in the collage")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    return p.parse_args()


def load_positive_samples(data_root, split, img_size, max_samples=100):
    """Load positive-only samples from dataset (parquet or file-based)."""
    import io
    data_root = Path(data_root)
    split_dir = data_root / split

    # Try parquet format first
    parquet_files = sorted(split_dir.glob("*.parquet")) if split_dir.is_dir() else []
    if parquet_files:
        return _load_from_parquet(split_dir, img_size, max_samples)

    # Fallback to file-based
    return _load_from_files(data_root, split, img_size, max_samples)


def _load_from_parquet(split_dir, img_size, max_samples):
    """Load positive samples from parquet files."""
    import io
    import pyarrow.parquet as pq

    table = pq.read_table(split_dir)
    img_col = table.column("image")
    is_neg_col = table.column("is_negative")
    coord_cols = ["corner_tl_x", "corner_tl_y", "corner_tr_x", "corner_tr_y",
                  "corner_br_x", "corner_br_y", "corner_bl_x", "corner_bl_y"]

    images = []
    coords_list = []
    names = []

    for i in range(table.num_rows):
        if len(images) >= max_samples:
            break
        if is_neg_col[i].as_py():
            continue

        img_bytes = img_col[i].as_py()["bytes"]
        coord_values = [table.column(cn)[i].as_py() for cn in coord_cols]
        c = np.array(coord_values, dtype=np.float32)
        if c.sum() == 0:
            continue

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8).copy()

        images.append(arr)
        coords_list.append(c)

        # Try to get filename
        if "filename" in table.column_names:
            names.append(table.column("filename")[i].as_py())
        else:
            names.append(f"sample_{i}")

    return np.array(images), np.array(coords_list), names


def _load_from_files(data_root, split, img_size, max_samples):
    """Load positive samples from file-based dataset."""
    split_file = None
    for name in [f"{split}.txt", f"{split}_with_negative_v2.txt",
                 f"{split}_with_negative.txt"]:
        candidate = data_root / name
        if candidate.exists():
            split_file = candidate
            break
    if split_file is None:
        raise FileNotFoundError(f"No split file for '{split}' in {data_root}")

    with open(split_file) as f:
        filenames = [line.strip().split(";")[0].strip() for line in f if line.strip()]

    filenames = [fn for fn in filenames if not fn.startswith("negative_")]

    images = []
    coords_list = []
    names = []

    for fn in filenames:
        if len(images) >= max_samples:
            break

        img_path = data_root / "images" / fn
        label_path = data_root / "labels" / f"{Path(fn).stem}.txt"

        if not img_path.exists() or not label_path.exists():
            continue

        with open(label_path) as f:
            line = f.readline().strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        c = np.array([float(x) for x in parts[1:9]], dtype=np.float32)

        with Image.open(img_path) as img:
            img = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8).copy()

        images.append(arr)
        coords_list.append(c)
        names.append(fn)

    return np.array(images), np.array(coords_list), names


def draw_corners(pil_img, coords_norm, img_size, color_quad="white",
                 width=2, radius=4):
    """Draw quadrilateral and corner dots on a PIL image."""
    draw = ImageDraw.Draw(pil_img)
    pts = coords_norm.reshape(4, 2) * img_size

    # Draw quadrilateral
    polygon_pts = [tuple(pt) for pt in pts] + [tuple(pts[0])]
    for i in range(4):
        x0, y0 = polygon_pts[i]
        x1, y1 = polygon_pts[i + 1]
        draw.line([(x0, y0), (x1, y1)], fill=color_quad, width=width)

    # Draw corner dots
    for i, (x, y) in enumerate(pts):
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=CORNER_COLORS[i],
            outline="black",
            width=1,
        )
    return pil_img


def denormalize_image(img_float, image_norm="imagenet"):
    """Convert normalized float32 tensor back to uint8 for display."""
    img = img_float.numpy()
    if image_norm == "imagenet":
        img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    elif image_norm == "zero_one":
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    else:
        img = img.clip(0, 255).astype(np.uint8)
    return img


def augment_geometric_only(images_f, coords, has_doc, rotation_range, scale_range):
    """Apply ONLY geometric augmentations (no photometric) for clean visualization."""
    # Horizontal flip 50%
    batch_size = tf.shape(images_f)[0]
    flip_mask = tf.random.uniform([batch_size]) > 0.5
    flip_mask_img = tf.reshape(flip_mask, [batch_size, 1, 1, 1])

    images_out = tf.where(flip_mask_img, tf.reverse(images_f, axis=[2]), images_f)

    flip_mask_coord = tf.cast(tf.reshape(flip_mask, [batch_size, 1]), tf.float32)
    has_doc_mask = tf.cast(tf.reshape(has_doc > 0.5, [batch_size, 1]), tf.float32)
    should_flip = flip_mask_coord * has_doc_mask

    x0, y0, x1, y1, x2, y2, x3, y3 = tf.unstack(coords, axis=1)
    coords_flipped = tf.stack([
        1.0 - x1, y1, 1.0 - x0, y0,
        1.0 - x3, y3, 1.0 - x2, y2,
    ], axis=1)
    coords_out = coords * (1.0 - should_flip) + coords_flipped * should_flip
    coords_out = tf.clip_by_value(coords_out, 0.0, 1.0)

    # Rotation
    if rotation_range > 0:
        images_out, coords_out = _tf_rotate_batch(
            images_out, coords_out, has_doc, rotation_range)

    # Scale
    if scale_range > 0:
        images_out, coords_out = _tf_scale_batch(
            images_out, coords_out, has_doc, scale_range)

    return images_out, coords_out


def apply_flip_only(images_f, coords, has_doc):
    """Apply ONLY horizontal flip (forced, not random) for debug."""
    batch_size = tf.shape(images_f)[0]
    images_out = tf.reverse(images_f, axis=[2])

    has_doc_mask = tf.cast(tf.reshape(has_doc > 0.5, [batch_size, 1]), tf.float32)
    x0, y0, x1, y1, x2, y2, x3, y3 = tf.unstack(coords, axis=1)
    coords_flipped = tf.stack([
        1.0 - x1, y1, 1.0 - x0, y0,
        1.0 - x3, y3, 1.0 - x2, y2,
    ], axis=1)
    coords_out = coords * (1.0 - has_doc_mask) + coords_flipped * has_doc_mask
    coords_out = tf.clip_by_value(coords_out, 0.0, 1.0)
    return images_out, coords_out


def apply_rotation_only(images_f, coords, has_doc, rotation_range):
    """Apply ONLY rotation (fixed angle = rotation_range) for debug."""
    batch_size = tf.shape(images_f)[0]
    h = tf.cast(tf.shape(images_f)[1], tf.float32)
    w = tf.cast(tf.shape(images_f)[2], tf.float32)

    # Fixed angle (not random) for deterministic debug
    angle_deg = rotation_range
    angle_rad = angle_deg * (3.14159265 / 180.0)
    angles = tf.fill([batch_size], angle_rad)
    cos_a = tf.cos(angles)
    sin_a = tf.sin(angles)

    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    tx = cx - cos_a * cx - sin_a * cy
    ty = cy + sin_a * cx - cos_a * cy

    transforms = tf.stack([
        cos_a, sin_a, tx,
        -sin_a, cos_a, ty,
        tf.zeros([batch_size]), tf.zeros([batch_size])
    ], axis=1)

    images_out = tf.raw_ops.ImageProjectiveTransformV3(
        images=images_f, transforms=transforms,
        output_shape=tf.shape(images_f)[1:3],
        interpolation="BILINEAR", fill_mode="NEAREST", fill_value=0.0)

    # Forward transform for coordinates
    coords_4x2 = tf.reshape(coords, [-1, 4, 2])
    x = coords_4x2[:, :, 0] - 0.5
    y = coords_4x2[:, :, 1] - 0.5
    new_x = cos_a[:, None] * x - sin_a[:, None] * y + 0.5
    new_y = sin_a[:, None] * x + cos_a[:, None] * y + 0.5
    new_coords = tf.stack([new_x, new_y], axis=-1)
    new_coords = tf.reshape(new_coords, [-1, 8])

    has_doc_mask = tf.cast(tf.reshape(has_doc > 0.5, [-1, 1]), tf.float32)
    coords_out = coords * (1.0 - has_doc_mask) + new_coords * has_doc_mask
    coords_out = tf.clip_by_value(coords_out, 0.0, 1.0)
    return images_out, coords_out


def apply_scale_only(images_f, coords, has_doc, scale_factor):
    """Apply ONLY scale (fixed factor) for debug."""
    batch_size = tf.shape(images_f)[0]
    h = tf.cast(tf.shape(images_f)[1], tf.float32)
    w = tf.cast(tf.shape(images_f)[2], tf.float32)

    scales = tf.fill([batch_size], scale_factor)
    has_doc_1d = tf.cast(tf.reshape(has_doc > 0.5, [batch_size]), tf.float32)

    coords_4x2 = tf.reshape(coords, [-1, 4, 2])
    new_coords_4x2 = 0.5 + (coords_4x2 - 0.5) * scales[:, None, None]
    new_coords = tf.reshape(new_coords_4x2, [-1, 8])

    coords_min = tf.reduce_min(new_coords_4x2, axis=[1, 2])
    coords_max = tf.reduce_max(new_coords_4x2, axis=[1, 2])
    valid = tf.cast((coords_min >= 0.0) & (coords_max <= 1.0), tf.float32)
    apply_mask = valid * has_doc_1d

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
        images_f, boxes, box_indices, crop_size, method='bilinear')

    apply_mask_img = tf.reshape(apply_mask, [batch_size, 1, 1, 1])
    images_out = images_f * (1.0 - apply_mask_img) + images_scaled * apply_mask_img

    apply_mask_coord = tf.reshape(apply_mask, [batch_size, 1])
    coords_out = coords * (1.0 - apply_mask_coord) + new_coords * apply_mask_coord
    coords_out = tf.clip_by_value(coords_out, 0.0, 1.0)
    return images_out, coords_out


def make_collage(args):
    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)

    print(f"Loading samples from {args.data_root}/{args.split}...", flush=True)
    images_uint8, coords_np, names = load_positive_samples(
        args.data_root, args.split, args.img_size,
        max_samples=args.num_samples * 3,
    )
    print(f"  Loaded {len(images_uint8)} positive samples")

    n = min(args.num_samples, len(images_uint8))
    if len(images_uint8) > n:
        idx = np.random.choice(len(images_uint8), n, replace=False)
        images_uint8 = images_uint8[idx]
        coords_np = coords_np[idx]
        names = [names[i] for i in idx]

    # Normalize for augmentation
    images_f = tf.cast(images_uint8, tf.float32) / 255.0
    images_f = (images_f - IMAGENET_MEAN) / IMAGENET_STD

    # Layout: 5 columns — Original | Flip | Rot +N° | Rot -N° | Scale
    col_labels = [
        "Original",
        "Flip",
        f"Rot +{args.rotation_range}°",
        f"Rot -{args.rotation_range}°",
        f"Scale {1.0 - args.scale_range:.2f}x",
    ]
    cols = len(col_labels)
    rows = n
    cell = args.cell_size
    margin = 2
    header_h = 20

    collage_w = cols * (cell + margin) + margin
    collage_h = rows * (cell + margin + header_h) + margin
    collage = Image.new("RGB", (collage_w, collage_h), color=(40, 40, 40))

    for row_i in range(n):
        img_batch = images_f[row_i:row_i + 1]
        coord_batch = tf.constant(coords_np[row_i:row_i + 1])
        has_doc_batch = tf.constant([1.0])

        # Generate all 5 variants
        variants = []

        # 0: Original
        variants.append((images_uint8[row_i], coords_np[row_i]))

        # 1: Flip only
        flip_img, flip_coords = apply_flip_only(img_batch, coord_batch, has_doc_batch)
        variants.append((denormalize_image(flip_img[0]), flip_coords[0].numpy()))

        # 2: Rotation +N degrees
        rot_img, rot_coords = apply_rotation_only(
            img_batch, coord_batch, has_doc_batch, args.rotation_range)
        variants.append((denormalize_image(rot_img[0]), rot_coords[0].numpy()))

        # 3: Rotation -N degrees
        rot_img2, rot_coords2 = apply_rotation_only(
            img_batch, coord_batch, has_doc_batch, -args.rotation_range)
        variants.append((denormalize_image(rot_img2[0]), rot_coords2[0].numpy()))

        # 4: Scale (zoom in)
        scale_img, scale_coords = apply_scale_only(
            img_batch, coord_batch, has_doc_batch, 1.0 - args.scale_range)
        variants.append((denormalize_image(scale_img[0]), scale_coords[0].numpy()))

        y = margin + row_i * (cell + margin + header_h) + header_h

        # Filename header
        draw_c = ImageDraw.Draw(collage)
        draw_c.text((margin + 4, y - header_h + 2),
                    f"{names[row_i][:60]}", fill=(200, 200, 200))

        for col_i, (img_data, coord_data) in enumerate(variants):
            if isinstance(img_data, np.ndarray) and img_data.dtype == np.uint8:
                pil_img = Image.fromarray(img_data)
            else:
                pil_img = Image.fromarray(img_data)
            pil_img = pil_img.resize((cell, cell), Image.BILINEAR)

            quad_color = "white" if col_i == 0 else "cyan"
            draw_corners(pil_img, coord_data, cell, color_quad=quad_color)

            draw_a = ImageDraw.Draw(pil_img)
            draw_a.text((4, 4), col_labels[col_i],
                        fill="white" if col_i == 0 else "cyan")

            x = margin + col_i * (cell + margin)
            collage.paste(pil_img, (x, y))

    collage.save(args.output, quality=95)
    print(f"\nCollage saved: {args.output} ({collage_w}x{collage_h})")
    print(f"  {n} samples x {cols} columns: {' | '.join(col_labels)}")


def main():
    args = parse_args()
    make_collage(args)


if __name__ == "__main__":
    main()
