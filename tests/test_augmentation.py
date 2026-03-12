"""Tests for TF batch augmentation functions (§11 of AUGMENTATION_PROPOSAL.md)."""

import numpy as np
import pytest
import tensorflow as tf

from dataset import (
    _tf_rotate_batch,
    _tf_scale_batch,
    augment_sample,
    tf_augment_batch,
    tf_augment_color_only,
)


# ---------------------------------------------------------------------------
# §11.1 — tf_augment_batch preserves shape and dtype
# ---------------------------------------------------------------------------

class TestTfAugmentBatchShapeDtype:
    def test_preserves_shape_dtype(self):
        images = tf.random.uniform([4, 224, 224, 3], -2.0, 2.0)
        coords = tf.random.uniform([4, 8], 0.2, 0.8)
        has_doc = tf.constant([1.0, 1.0, 0.0, 1.0])

        out_img, out_coords = tf_augment_batch(images, coords, has_doc)

        assert out_img.shape == (4, 224, 224, 3)
        assert out_coords.shape == (4, 8)
        assert out_img.dtype == tf.float32
        assert out_coords.dtype == tf.float32

    def test_single_sample(self):
        images = tf.random.uniform([1, 224, 224, 3], -2.0, 2.0)
        coords = tf.constant([[0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8]])
        has_doc = tf.constant([1.0])

        out_img, out_coords = tf_augment_batch(images, coords, has_doc)

        assert out_img.shape == (1, 224, 224, 3)
        assert out_coords.shape == (1, 8)


# ---------------------------------------------------------------------------
# §11.2 — coordinates stay in [0, 1]
# ---------------------------------------------------------------------------

class TestCoordsInRange:
    def test_coords_clipped_with_rotation_and_scale(self):
        images = tf.random.uniform([8, 224, 224, 3], -2.0, 2.0)
        coords = tf.random.uniform([8, 8], 0.0, 1.0)
        has_doc = tf.ones([8])

        _, out_coords = tf_augment_batch(
            images, coords, has_doc,
            rotation_range=10.0, scale_range=0.15,
        )

        assert tf.reduce_min(out_coords).numpy() >= 0.0
        assert tf.reduce_max(out_coords).numpy() <= 1.0

    def test_coords_clipped_edge_values(self):
        """Coords near edges should still be clipped after augmentation."""
        images = tf.random.uniform([4, 224, 224, 3], -2.0, 2.0)
        coords = tf.constant([
            [0.01, 0.01, 0.99, 0.01, 0.99, 0.99, 0.01, 0.99],
            [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ])
        has_doc = tf.ones([4])

        _, out_coords = tf_augment_batch(
            images, coords, has_doc, rotation_range=5.0)

        assert tf.reduce_min(out_coords).numpy() >= 0.0
        assert tf.reduce_max(out_coords).numpy() <= 1.0


# ---------------------------------------------------------------------------
# §11.3 — negative samples don't get geometric transforms on coordinates
# ---------------------------------------------------------------------------

class TestNegativeSamples:
    def test_negative_samples_coords_unchanged(self):
        images = tf.random.uniform([4, 224, 224, 3])
        coords = tf.zeros([4, 8])
        has_doc = tf.zeros([4])

        _, out_coords = tf_augment_batch(
            images, coords, has_doc,
            rotation_range=5.0, scale_range=0.1,
        )

        np.testing.assert_allclose(out_coords.numpy(), 0.0, atol=1e-6)

    def test_mixed_batch_negative_preserved(self):
        """In a mixed batch, negative sample coords must stay zero."""
        images = tf.random.uniform([4, 224, 224, 3], -2.0, 2.0)
        coords = tf.constant([
            [0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # negative
            [0.3, 0.3, 0.7, 0.3, 0.7, 0.7, 0.3, 0.7],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # negative
        ])
        has_doc = tf.constant([1.0, 0.0, 1.0, 0.0])

        _, out_coords = tf_augment_batch(
            images, coords, has_doc,
            rotation_range=5.0, scale_range=0.1,
        )

        # Negative samples (index 1, 3) must have coords unchanged
        np.testing.assert_allclose(out_coords[1].numpy(), 0.0, atol=1e-6)
        np.testing.assert_allclose(out_coords[3].numpy(), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# §11.4 — horizontal flip remaps TL, TR, BR, BL correctly
# ---------------------------------------------------------------------------

class TestHorizontalFlip:
    def test_flip_remap_corners(self):
        """When all samples are flipped, coords should be TL<->TR, BL<->BR with x->1-x."""
        # Asymmetric coords: TL=(0.1,0.2), TR=(0.7,0.3), BR=(0.65,0.8), BL=(0.15,0.75)
        coords_in = np.array([[0.1, 0.2, 0.7, 0.3, 0.65, 0.8, 0.15, 0.75]], dtype=np.float32)
        # After flip: new_TL=(1-x1,y1)=(0.3,0.3), new_TR=(1-x0,y0)=(0.9,0.2),
        #             new_BR=(1-x3,y3)=(0.85,0.75), new_BL=(1-x2,y2)=(0.35,0.8)
        expected_flipped = np.array([
            [1.0 - 0.7, 0.3, 1.0 - 0.1, 0.2, 1.0 - 0.15, 0.75, 1.0 - 0.65, 0.8]
        ], dtype=np.float32)

        # Run many times to get some flips (50% chance each time)
        flipped_count = 0
        not_flipped_count = 0
        n_trials = 50
        for _ in range(n_trials):
            images = tf.zeros([1, 64, 64, 3])
            coords = tf.constant(coords_in)
            has_doc = tf.constant([1.0])

            out_img, out_coords = tf_augment_batch(
                images, coords, has_doc,
                rotation_range=0.0, scale_range=0.0,
            )

            oc = out_coords.numpy()
            # Detect flip: original x0=0.1, flipped x0=0.3 — clearly different
            if abs(oc[0, 0] - coords_in[0, 0]) < 0.05:
                not_flipped_count += 1
            else:
                flipped_count += 1
                np.testing.assert_allclose(oc, expected_flipped, atol=0.05)

        # With 50 trials, we should get both flipped and non-flipped
        assert flipped_count > 0, "No flips detected in 50 trials"
        assert not_flipped_count > 0, "All samples were flipped — expected ~50%"


# ---------------------------------------------------------------------------
# §11.5 — rotation_range=0 is identity
# ---------------------------------------------------------------------------

class TestRotationIdentity:
    def test_rotation_zero_identity(self):
        coords = tf.constant([[0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8]])
        has_doc = tf.constant([1.0])
        images = tf.zeros([1, 224, 224, 3])

        out_img, out_coords = _tf_rotate_batch(images, coords, has_doc, rotation_range=0.0)

        np.testing.assert_allclose(out_coords.numpy(), coords.numpy(), atol=1e-5)

    def test_rotation_zero_preserves_image(self):
        images = tf.random.uniform([2, 64, 64, 3])
        coords = tf.random.uniform([2, 8], 0.2, 0.8)
        has_doc = tf.ones([2])

        out_img, _ = _tf_rotate_batch(images, coords, has_doc, rotation_range=0.0)

        np.testing.assert_allclose(out_img.numpy(), images.numpy(), atol=1e-5)


# ---------------------------------------------------------------------------
# §11.6 — scale_range=0 is identity
# ---------------------------------------------------------------------------

class TestScaleIdentity:
    def test_scale_zero_identity(self):
        coords = tf.constant([[0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8]])
        has_doc = tf.constant([1.0])
        images = tf.zeros([1, 224, 224, 3])

        out_img, out_coords = _tf_scale_batch(images, coords, has_doc, scale_range=0.0)

        np.testing.assert_allclose(out_coords.numpy(), coords.numpy(), atol=1e-5)

    def test_scale_zero_preserves_image(self):
        images = tf.random.uniform([2, 64, 64, 3])
        coords = tf.random.uniform([2, 8], 0.2, 0.8)
        has_doc = tf.ones([2])

        out_img, _ = _tf_scale_batch(images, coords, has_doc, scale_range=0.0)

        np.testing.assert_allclose(out_img.numpy(), images.numpy(), atol=1e-4)


# ---------------------------------------------------------------------------
# §11.7 — tf_augment_color_only does not modify coordinates
# ---------------------------------------------------------------------------

class TestColorOnly:
    def test_no_coord_change(self):
        """tf_augment_color_only takes no coords — by design it can't modify them."""
        images = tf.random.uniform([4, 224, 224, 3], -2.0, 2.0)
        out = tf_augment_color_only(images)
        assert out.shape == images.shape
        assert out.dtype == tf.float32

    def test_respects_imagenet_norm(self):
        images = tf.random.uniform([2, 64, 64, 3], -2.5, 2.5)
        out = tf_augment_color_only(images, image_norm="imagenet")
        assert tf.reduce_min(out).numpy() >= -3.0
        assert tf.reduce_max(out).numpy() <= 3.0

    def test_respects_zero_one_norm(self):
        images = tf.random.uniform([2, 64, 64, 3], 0.0, 1.0)
        out = tf_augment_color_only(images, image_norm="zero_one")
        assert tf.reduce_min(out).numpy() >= 0.0
        assert tf.reduce_max(out).numpy() <= 1.0

    def test_respects_raw255_norm(self):
        images = tf.random.uniform([2, 64, 64, 3], 0.0, 255.0)
        out = tf_augment_color_only(images, image_norm="raw255")
        assert tf.reduce_min(out).numpy() >= 0.0
        assert tf.reduce_max(out).numpy() <= 255.0


# ---------------------------------------------------------------------------
# §11.8 — augment_sample can't produce incoherent geometry
# ---------------------------------------------------------------------------

class TestAugmentSampleNoGeometry:
    def test_coords_unchanged(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        coords = np.array([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8], dtype=np.float32)

        _, coords_out = augment_sample(img, coords)

        np.testing.assert_allclose(coords_out, coords)

    def test_coords_unchanged_with_default_config(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        coords = np.array([0.1, 0.15, 0.9, 0.12, 0.88, 0.85, 0.12, 0.9], dtype=np.float32)

        _, coords_out = augment_sample(img, coords)

        np.testing.assert_allclose(coords_out, coords)


# ---------------------------------------------------------------------------
# §11 — image_norm variants for tf_augment_batch
# ---------------------------------------------------------------------------

class TestImageNormVariants:
    def test_imagenet_clipping(self):
        images = tf.random.uniform([2, 64, 64, 3], -2.0, 2.0)
        coords = tf.random.uniform([2, 8], 0.2, 0.8)
        has_doc = tf.ones([2])

        out_img, _ = tf_augment_batch(
            images, coords, has_doc, image_norm="imagenet",
            rotation_range=0.0, scale_range=0.0)

        assert tf.reduce_min(out_img).numpy() >= -3.0
        assert tf.reduce_max(out_img).numpy() <= 3.0

    def test_zero_one_clipping(self):
        images = tf.random.uniform([2, 64, 64, 3], 0.0, 1.0)
        coords = tf.random.uniform([2, 8], 0.2, 0.8)
        has_doc = tf.ones([2])

        out_img, _ = tf_augment_batch(
            images, coords, has_doc, image_norm="zero_one",
            rotation_range=0.0, scale_range=0.0)

        assert tf.reduce_min(out_img).numpy() >= 0.0
        assert tf.reduce_max(out_img).numpy() <= 1.0
