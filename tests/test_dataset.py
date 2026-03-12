"""Tests for v2/dataset.py — data loading, augmentation, normalization."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dataset import (
    DEFAULT_AUG_CONFIG,
    IMAGENET_MEAN,
    IMAGENET_STD,
    augment_sample,
    create_dataset,
    load_image,
    load_label_yolo_obb,
    load_split_file,
    normalize_image,
)


@pytest.fixture
def mini_dataset(tmp_path):
    """Create a minimal dataset directory for testing."""
    # Create directory structure
    (tmp_path / "images").mkdir()
    (tmp_path / "images-negative").mkdir()
    (tmp_path / "labels").mkdir()

    # Create positive images and labels
    for i in range(5):
        img = Image.new("RGB", (100, 100), color=(i * 50, 100, 150))
        img.save(tmp_path / "images" / f"img_{i:03d}.jpg")
        # YOLO OBB label: class x0 y0 x1 y1 x2 y2 x3 y3
        with open(tmp_path / "labels" / f"img_{i:03d}.txt", "w") as f:
            f.write(f"0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")

    # Create negative images
    for i in range(2):
        img = Image.new("RGB", (100, 100), color=(200, 200, 200))
        img.save(tmp_path / "images-negative" / f"negative_neg_{i:03d}.jpg")

    # Create split files
    names = [f"img_{i:03d}.jpg" for i in range(5)]
    neg_names = [f"negative_neg_{i:03d}.jpg" for i in range(2)]
    with open(tmp_path / "train.txt", "w") as f:
        f.write("\n".join(names[:3] + neg_names[:1]))
    with open(tmp_path / "val.txt", "w") as f:
        f.write("\n".join(names[3:] + neg_names[1:]))

    return tmp_path


class TestLoadSplitFile:
    def test_basic(self, tmp_path):
        p = tmp_path / "split.txt"
        p.write_text("img_000.jpg\nimg_001.jpg\nimg_002.jpg\n")
        names = load_split_file(str(p))
        assert len(names) == 3
        assert names[0] == "img_000.jpg"

    def test_semicolon_format(self, tmp_path):
        p = tmp_path / "split.txt"
        p.write_text("img_000.jpg;img_001.jpg;img_002.jpg")
        names = load_split_file(str(p))
        assert len(names) == 3

    def test_empty(self, tmp_path):
        p = tmp_path / "split.txt"
        p.write_text("")
        names = load_split_file(str(p))
        assert len(names) == 0


class TestLoadLabelYoloObb:
    def test_valid_label(self, tmp_path):
        p = tmp_path / "label.txt"
        p.write_text("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")
        coords = load_label_yolo_obb(str(p))
        assert coords.shape == (8,)
        np.testing.assert_allclose(coords, [0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8])

    def test_empty_label(self, tmp_path):
        p = tmp_path / "label.txt"
        p.write_text("")
        coords = load_label_yolo_obb(str(p))
        assert coords.shape == (8,)
        np.testing.assert_allclose(coords, 0.0)

    def test_short_label(self, tmp_path):
        p = tmp_path / "label.txt"
        p.write_text("0 0.2 0.3")
        coords = load_label_yolo_obb(str(p))
        np.testing.assert_allclose(coords, 0.0)


class TestLoadImage:
    def test_basic(self, tmp_path):
        img = Image.new("RGB", (200, 200), color=(128, 64, 32))
        p = tmp_path / "test.jpg"
        img.save(p)
        result = load_image(str(p), img_size=224)
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.uint8

    def test_different_size(self, tmp_path):
        img = Image.new("RGB", (100, 100))
        p = tmp_path / "test.jpg"
        img.save(p)
        result = load_image(str(p), img_size=128)
        assert result.shape == (128, 128, 3)


class TestNormalizeImage:
    def test_imagenet(self):
        img = np.full((224, 224, 3), 128, dtype=np.uint8)
        result = normalize_image(img, "imagenet")
        assert result.dtype == np.float32
        assert result.shape == (224, 224, 3)

    def test_zero_one(self):
        img = np.full((224, 224, 3), 255, dtype=np.uint8)
        result = normalize_image(img, "zero_one")
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_raw255(self):
        img = np.full((224, 224, 3), 128, dtype=np.uint8)
        result = normalize_image(img, "raw255")
        np.testing.assert_allclose(result, 128.0)

    def test_unknown(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown"):
            normalize_image(img, "unknown_method")


class TestAugmentSample:
    def test_basic(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        coords = np.array([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8], dtype=np.float32)
        img_out, coords_out = augment_sample(img, coords)
        assert img_out.shape == (224, 224, 3)
        assert coords_out.shape == (8,)
        # Option B: augment_sample no longer modifies coordinates
        np.testing.assert_allclose(coords_out, coords, atol=1e-5)

    def test_no_augment_config(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        coords = np.array([0.3, 0.3, 0.7, 0.3, 0.7, 0.7, 0.3, 0.7], dtype=np.float32)
        config = {
            "brightness": 0.0,
            "contrast": 0.0,
            "saturation": 0.0,
            "blur_prob": 0.0,
            "blur_kernel": 3,
        }
        _, coords_out = augment_sample(img, coords, aug_config=config)
        np.testing.assert_allclose(coords_out, coords, atol=1e-5)


class TestCreateDataset:
    def test_creates_dataset(self, mini_dataset):
        ds = create_dataset(
            str(mini_dataset), split="train", img_size=64,
            batch_size=2, shuffle=False, augment=False,
            negative_ratio=0.0,
        )
        assert ds is not None
        for images, targets in ds.take(1):
            assert images.shape[1:] == (64, 64, 3)
            assert "coords" in targets
            assert "has_doc" in targets

    def test_with_negatives(self, mini_dataset):
        ds = create_dataset(
            str(mini_dataset), split="train", img_size=64,
            batch_size=4, shuffle=False, augment=False,
            negative_ratio=0.3,
        )
        all_has_doc = []
        for _, targets in ds:
            all_has_doc.extend(targets["has_doc"].numpy().tolist())
        # Should have at least some samples
        assert len(all_has_doc) > 0

    def test_missing_split(self, mini_dataset):
        with pytest.raises(FileNotFoundError):
            create_dataset(str(mini_dataset), split="nonexistent")

    def test_augmented(self, mini_dataset):
        ds = create_dataset(
            str(mini_dataset), split="train", img_size=64,
            batch_size=2, shuffle=False, augment=True,
            negative_ratio=0.0,
        )
        for images, _ in ds.take(1):
            assert images.shape[1:] == (64, 64, 3)
