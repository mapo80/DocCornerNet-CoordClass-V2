"""Tests for v2/model.py — construction, shapes, layers, serialization."""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from model import (
    AxisMean,
    Broadcast1D,
    NearestUpsample2x,
    Resize1D,
    SimCCDecode,
    build_doccorner_v2,
    create_inference_model,
    create_model,
    load_inference_model,
    _get_feature_layers,
    _build_backbone,
    _separable_conv_block,
)


# ---------------------------------------------------------------------------
# Custom layer unit tests
# ---------------------------------------------------------------------------

class TestAxisMean:
    def test_axis1_shape(self):
        layer = AxisMean(axis=1)
        x = tf.random.normal([2, 56, 56, 32])
        out = layer(x)
        assert out.shape == (2, 56, 32)

    def test_axis2_shape(self):
        layer = AxisMean(axis=2)
        x = tf.random.normal([2, 56, 56, 32])
        out = layer(x)
        assert out.shape == (2, 56, 32)

    def test_axis1_values(self):
        layer = AxisMean(axis=1)
        x = tf.ones([1, 4, 3, 2])
        out = layer(x)
        np.testing.assert_allclose(out.numpy(), 1.0)

    def test_axis2_values(self):
        layer = AxisMean(axis=2)
        x = tf.ones([1, 4, 3, 2])
        out = layer(x)
        np.testing.assert_allclose(out.numpy(), 1.0)

    def test_invalid_axis(self):
        with pytest.raises(ValueError, match="axis must be 1 or 2"):
            AxisMean(axis=0)

    def test_get_config(self):
        layer = AxisMean(axis=1, name="test_am")
        cfg = layer.get_config()
        assert cfg["axis"] == 1
        assert cfg["name"] == "test_am"


class TestResize1D:
    def test_upsample(self):
        layer = Resize1D(target_length=224)
        x = tf.random.normal([2, 56, 32])
        out = layer(x)
        assert out.shape == (2, 224, 32)

    def test_downsample(self):
        layer = Resize1D(target_length=56)
        x = tf.random.normal([2, 224, 32])
        out = layer(x)
        assert out.shape == (2, 56, 32)

    def test_identity(self):
        layer = Resize1D(target_length=10)
        x = tf.random.normal([2, 10, 5])
        out = layer(x)
        assert out.shape == (2, 10, 5)

    def test_get_config(self):
        layer = Resize1D(target_length=128, method="bilinear")
        cfg = layer.get_config()
        assert cfg["target_length"] == 128
        assert cfg["method"] == "bilinear"


class TestBroadcast1D:
    def test_shape(self):
        layer = Broadcast1D(target_length=224)
        x = tf.random.normal([2, 48])
        out = layer(x)
        assert out.shape == (2, 224, 48)

    def test_values(self):
        layer = Broadcast1D(target_length=3)
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        out = layer(x).numpy()
        np.testing.assert_allclose(out[0, 0], [1.0, 2.0])
        np.testing.assert_allclose(out[0, 1], [1.0, 2.0])
        np.testing.assert_allclose(out[0, 2], [1.0, 2.0])

    def test_get_config(self):
        layer = Broadcast1D(target_length=100)
        cfg = layer.get_config()
        assert cfg["target_length"] == 100


class TestNearestUpsample2x:
    def test_shape(self):
        layer = NearestUpsample2x()
        x = tf.random.normal([2, 14, 14, 32])
        out = layer(x)
        assert out.shape == (2, 28, 28, 32)

    def test_values(self):
        """Each pixel should be duplicated in 2x2 block."""
        layer = NearestUpsample2x()
        x = tf.constant([[[[1.0], [2.0]], [[3.0], [4.0]]]])  # [1, 2, 2, 1]
        out = layer(x).numpy()
        assert out.shape == (1, 4, 4, 1)
        np.testing.assert_allclose(out[0, 0, 0, 0], 1.0)
        np.testing.assert_allclose(out[0, 0, 1, 0], 1.0)
        np.testing.assert_allclose(out[0, 1, 0, 0], 1.0)
        np.testing.assert_allclose(out[0, 1, 1, 0], 1.0)
        np.testing.assert_allclose(out[0, 2, 2, 0], 4.0)

    def test_not_built_state(self):
        layer = NearestUpsample2x()
        assert layer._h is None
        # After calling, auto-builds
        out = layer(tf.random.normal([2, 14, 14, 32]))
        assert layer._h == 14

    def test_get_config(self):
        layer = NearestUpsample2x(name="up2x")
        cfg = layer.get_config()
        assert cfg["name"] == "up2x"


class TestSimCCDecode:
    def test_shape(self):
        layer = SimCCDecode(num_bins=224, tau=1.0)
        sx = tf.random.normal([2, 4, 224])
        sy = tf.random.normal([2, 4, 224])
        out = layer([sx, sy])
        assert out.shape == (2, 8)

    def test_range(self):
        layer = SimCCDecode(num_bins=224, tau=1.0)
        sx = tf.random.normal([4, 4, 224])
        sy = tf.random.normal([4, 4, 224])
        out = layer([sx, sy]).numpy()
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_peaked_distribution(self):
        """When logits peak at center, coords should be near 0.5."""
        layer = SimCCDecode(num_bins=224, tau=0.01)
        # Create logits peaked at center bin
        logits = np.full((1, 4, 224), -100.0, dtype=np.float32)
        logits[:, :, 112] = 100.0  # peak at center
        sx = tf.constant(logits)
        sy = tf.constant(logits)
        out = layer([sx, sy]).numpy()
        np.testing.assert_allclose(out, 0.5, atol=0.01)

    def test_not_built_state(self):
        layer = SimCCDecode(num_bins=224, tau=1.0)
        assert layer._centers_col is None
        # After call, auto-builds
        out = layer([tf.zeros([1, 4, 224]), tf.zeros([1, 4, 224])])
        assert layer._centers_col is not None

    def test_get_config(self):
        layer = SimCCDecode(num_bins=336, tau=0.5)
        cfg = layer.get_config()
        assert cfg["num_bins"] == 336
        assert cfg["tau"] == 0.5


# ---------------------------------------------------------------------------
# Model construction tests
# ---------------------------------------------------------------------------

class TestModelConstruction:
    @pytest.fixture(scope="class")
    def model(self):
        return create_model(backbone_weights=None)

    def test_model_builds(self, model):
        assert model is not None

    def test_model_name(self, model):
        assert "V2" in model.name or "v2" in model.name.lower()

    def test_param_count_reasonable(self, model):
        """~496K params per proposal (495K base + ~1.5K attention)."""
        params = model.count_params()
        assert 400_000 < params < 600_000, f"Got {params:,}"

    def test_output_keys(self, model):
        x = np.random.randn(1, 224, 224, 3).astype(np.float32)
        out = model(x, training=False)
        assert "simcc_x" in out
        assert "simcc_y" in out
        assert "score_logit" in out
        assert "coords" in out

    def test_output_shapes(self, model):
        x = np.random.randn(2, 224, 224, 3).astype(np.float32)
        out = model(x, training=False)
        assert out["simcc_x"].shape == (2, 4, 224)
        assert out["simcc_y"].shape == (2, 4, 224)
        assert out["score_logit"].shape == (2, 1)
        assert out["coords"].shape == (2, 8)

    def test_coords_in_range(self, model):
        x = np.random.randn(4, 224, 224, 3).astype(np.float32)
        out = model(x, training=False)
        coords = out["coords"].numpy()
        assert coords.min() >= 0.0
        assert coords.max() <= 1.0

    def test_training_mode(self, model):
        x = np.random.randn(2, 224, 224, 3).astype(np.float32)
        out_train = model(x, training=True)
        out_eval = model(x, training=False)
        # Both should produce valid outputs
        assert out_train["coords"].shape == (2, 8)
        assert out_eval["coords"].shape == (2, 8)


class TestModelConfig:
    def test_custom_bins(self):
        model = create_model(num_bins=336, backbone_weights=None)
        x = np.random.randn(1, 224, 224, 3).astype(np.float32)
        out = model(x, training=False)
        assert out["simcc_x"].shape == (1, 4, 336)

    def test_custom_fpn_ch(self):
        model = create_model(fpn_ch=48, backbone_weights=None)
        assert model is not None
        assert model.count_params() > 0

    def test_custom_simcc_ch(self):
        model = create_model(simcc_ch=128, backbone_weights=None)
        assert model is not None

    def test_custom_alpha(self):
        model = create_model(alpha=0.5, backbone_weights=None)
        assert model is not None

    def test_include_preprocessing(self):
        model = create_model(backbone_include_preprocessing=True, backbone_weights=None)
        x = np.random.randint(0, 255, (1, 224, 224, 3)).astype(np.float32)
        out = model(x, training=False)
        assert out["coords"].shape == (1, 8)


class TestInferenceModel:
    def test_create_inference_model(self):
        train_model = create_model(backbone_weights=None)
        inf_model = create_inference_model(train_model)
        x = np.random.randn(2, 224, 224, 3).astype(np.float32)
        out = inf_model(x, training=False)
        assert isinstance(out, (list, tuple))
        assert len(out) == 2
        assert out[0].shape == (2, 8)   # coords
        assert out[1].shape == (2, 1)   # score_logit


class TestModelSerialization:
    def test_save_load_weights(self, tmp_path):
        model1 = create_model(backbone_weights=None)
        x = np.random.randn(1, 224, 224, 3).astype(np.float32)
        out1 = model1(x, training=False)

        weights_path = str(tmp_path / "test.weights.h5")
        model1.save_weights(weights_path)

        model2 = create_model(backbone_weights=None)
        model2.load_weights(weights_path)
        out2 = model2(x, training=False)

        np.testing.assert_allclose(
            out1["coords"].numpy(), out2["coords"].numpy(), atol=1e-5
        )

    def test_custom_layer_serialization(self):
        """All custom layers must be serializable (get_config roundtrip)."""
        layers_to_test = [
            AxisMean(axis=1),
            AxisMean(axis=2),
            Resize1D(target_length=224),
            Broadcast1D(target_length=224),
            NearestUpsample2x(),
            SimCCDecode(num_bins=224, tau=1.0),
        ]
        for layer in layers_to_test:
            cfg = layer.get_config()
            assert isinstance(cfg, dict), f"{layer.__class__.__name__} get_config failed"
            # Verify it can be reconstructed
            cls = layer.__class__
            restored = cls.from_config(cfg)
            cfg2 = restored.get_config()
            # Check key params match
            for key in cfg:
                if key != "name" and key != "dtype":
                    assert cfg[key] == cfg2[key], f"{cls.__name__}: {key} mismatch"


# ---------------------------------------------------------------------------
# Backbone / FPN helper tests
# ---------------------------------------------------------------------------

class TestBackboneHelpers:
    def test_build_backbone(self):
        inp = keras.Input((224, 224, 3))
        bb = _build_backbone(inp, alpha=0.35, backbone_weights=None,
                             backbone_include_preprocessing=False)
        assert bb is not None

    def test_build_backbone_with_preprocessing(self):
        inp = keras.Input((224, 224, 3))
        bb = _build_backbone(inp, alpha=0.35, backbone_weights=None,
                             backbone_include_preprocessing=True)
        assert bb is not None

    def test_get_feature_layers(self):
        inp = keras.Input((224, 224, 3))
        bb = _build_backbone(inp, alpha=0.35, backbone_weights=None,
                             backbone_include_preprocessing=False)
        c2, c3, c4, c5 = _get_feature_layers(bb, img_size=224)
        assert c2.shape[1] == 56
        assert c3.shape[1] == 28
        assert c4.shape[1] == 14
        assert c5.shape[1] == 7

    def test_separable_conv_block(self):
        x = keras.Input((56, 56, 32))
        out = _separable_conv_block(x, 32, "test")
        model = keras.Model(x, out)
        result = model(np.random.randn(1, 56, 56, 32).astype(np.float32))
        assert result.shape == (1, 56, 56, 32)


# ---------------------------------------------------------------------------
# Corner attention tests
# ---------------------------------------------------------------------------

class TestCornerAttention:
    def test_four_attention_heads_exist(self):
        model = create_model(backbone_weights=None)
        layer_names = [l.name for l in model.layers]
        for cn in ["tl", "tr", "br", "bl"]:
            assert any(f"att_{cn}_conv" in n for n in layer_names), f"Missing att_{cn}_conv"
            assert any(f"att_{cn}_sigmoid" in n for n in layer_names), f"Missing att_{cn}_sigmoid"
            assert any(f"att_{cn}_mul" in n for n in layer_names), f"Missing att_{cn}_mul"

    def test_shared_simcc_weights(self):
        """SimCC Conv1D layers should be shared (same name, called 4 times)."""
        model = create_model(backbone_weights=None)
        layer_names = [l.name for l in model.layers]
        # The shared layers should appear only once by name
        assert layer_names.count("simcc_x_conv1") == 1
        assert layer_names.count("simcc_y_conv1") == 1
