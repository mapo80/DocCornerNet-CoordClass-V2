"""
DocCornerNet V2: Document Corner Detection with Corner-Specific Spatial Attention.

Architecture:
- Backbone: MobileNetV2 alpha=0.35 (224x224 input)
- Neck: Mini-FPN top-down (C2/C3/C4 → P2 fused at 56x56)
- Corner Attention: 4 lightweight spatial attention heads on shared precursor
- Head: Direct per-corner SimCC coarse path + 2D heatmap/offset refinement
- Score: Document presence from C5 global pool
- Decode: SimCC coarse coords + heatmap-weighted local offsets → coords [B, 8]

Output dict keys:
  simcc_x:        [B, 4, num_bins]   X marginal logits per corner
  simcc_y:        [B, 4, num_bins]   Y marginal logits per corner
  corner_heatmap: [B, 56, 56, 4]     Corner heatmap logits
  corner_offset:  [B, 56, 56, 8]     Corner local offsets (dx,dy per corner)
  coords_2d:      [B, 8]             Direct 2D decode from heatmap+offset branch
  score_logit:    [B, 1]             Document presence logit
  coords:         [B, 8]             Decoded normalized coordinates
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable


# ---------------------------------------------------------------------------
# Custom serializable layers
# ---------------------------------------------------------------------------

@register_keras_serializable(package="doccorner_v2")
class AxisMean(layers.Layer):
    """Reduce a spatial axis of an NHWC tensor."""

    def __init__(self, axis: int, impl: str = "avgpool", **kwargs):
        super().__init__(**kwargs)
        if axis not in (1, 2):
            raise ValueError(f"AxisMean axis must be 1 or 2, got {axis}")
        self.axis = int(axis)
        self.impl = str(impl).lower().strip()
        self._h = None
        self._w = None
        self._c = None
        self._filter_full = None

    def build(self, input_shape):
        h, w, c = input_shape[1], input_shape[2], input_shape[3]
        if h is None or w is None or c is None:
            raise ValueError("AxisMean requires static H/W/C.")
        self._h = int(h)
        self._w = int(w)
        self._c = int(c)
        if self.impl == "dwconv_full":
            if self.axis == 1:
                k = np.ones((self._h, 1, self._c, 1), dtype=np.float32) / float(self._h)
            else:
                k = np.ones((1, self._w, self._c, 1), dtype=np.float32) / float(self._w)
            self._filter_full = tf.constant(k, dtype=tf.float32, name=f"{self.name}_dwfilter_full")
        super().build(input_shape)

    def call(self, inputs):
        if self._h is None or self._w is None or self._c is None:
            raise RuntimeError("AxisMean is not built.")
        if self.impl == "dwconv_full":
            if self._filter_full is None:
                raise RuntimeError("AxisMean dwconv_full filter missing.")
            x = tf.nn.depthwise_conv2d(inputs, self._filter_full, strides=[1, 1, 1, 1], padding="VALID")
            if self.axis == 1:
                return tf.reshape(x, [-1, self._w, self._c])
            return tf.reshape(x, [-1, self._h, self._c])
        if self.axis == 1:
            x = tf.nn.avg_pool2d(inputs, ksize=(self._h, 1), strides=(1, 1), padding="VALID")
            return tf.reshape(x, [-1, self._w, self._c])
        x = tf.nn.avg_pool2d(inputs, ksize=(1, self._w), strides=(1, 1), padding="VALID")
        return tf.reshape(x, [-1, self._h, self._c])

    def get_config(self):
        config = super().get_config()
        config["axis"] = self.axis
        config["impl"] = self.impl
        return config


@register_keras_serializable(package="doccorner_v2")
class Resize1D(layers.Layer):
    """Resize a 1D sequence [B, L, C] → [B, target_length, C] via bilinear interpolation."""

    def __init__(self, target_length: int, method: str = "bilinear", **kwargs):
        super().__init__(**kwargs)
        self.target_length = int(target_length)
        self.method = str(method)

    def call(self, inputs):
        channels = inputs.shape[-1]
        length = inputs.shape[1]
        if channels is None or length is None:
            raise ValueError("Resize1D requires known length and channel dims.")
        x = tf.reshape(inputs, [-1, int(length), 1, int(channels)])
        x = tf.image.resize(x, size=(self.target_length, 1), method=self.method)
        return tf.reshape(x, [-1, self.target_length, int(channels)])

    def get_config(self):
        config = super().get_config()
        config.update({"target_length": self.target_length, "method": self.method})
        return config


@register_keras_serializable(package="doccorner_v2")
class Broadcast1D(layers.Layer):
    """Broadcast [B, C] → [B, target_length, C] using reshape + MUL."""

    def __init__(self, target_length: int, **kwargs):
        super().__init__(**kwargs)
        self.target_length = int(target_length)
        self._channels = None
        self._ones = None

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError("Broadcast1D requires a known channel dimension.")
        self._channels = int(channels)
        self._ones = tf.constant(
            np.ones((1, self.target_length, 1), dtype=np.float32),
            dtype=tf.float32,
            name=f"{self.name}_ones",
        )
        super().build(input_shape)

    def call(self, inputs):
        if self._channels is None or self._ones is None:
            raise RuntimeError("Broadcast1D is not built.")
        x = tf.reshape(inputs, [-1, 1, self._channels])
        return x * tf.cast(self._ones, inputs.dtype)

    def get_config(self):
        config = super().get_config()
        config["target_length"] = self.target_length
        return config


@register_keras_serializable(package="doccorner_v2")
class Conv1DAsConv2D(layers.Layer):
    """XNNPACK-friendly Conv1D equivalent implemented via Conv2D."""

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        padding: str = "same",
        use_bias: bool = True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.strides = int(strides)
        self.padding = str(padding).lower().strip()
        self.use_bias = bool(use_bias)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self._length = None
        self._in_ch = None
        self._out_len = None
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        length = input_shape[1]
        in_ch = input_shape[2]
        if length is None or in_ch is None:
            raise ValueError("Conv1DAsConv2D requires static L/C.")
        self._length = int(length)
        self._in_ch = int(in_ch)
        if self.padding == "same":
            self._out_len = int(np.ceil(self._length / self.strides))
        elif self.padding == "valid":
            self._out_len = max(0, int(np.floor((self._length - self.kernel_size) / self.strides) + 1))
        else:
            raise ValueError(f"Unsupported padding='{self.padding}'")

        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.kernel_size, self._in_ch, self.filters),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
            )
        super().build(input_shape)

    def call(self, inputs):
        if self._length is None or self._in_ch is None or self._out_len is None or self.kernel is None:
            raise RuntimeError("Conv1DAsConv2D is not built.")
        x = tf.reshape(inputs, [-1, 1, self._length, self._in_ch])
        k = tf.reshape(self.kernel, [1, self.kernel_size, self._in_ch, self.filters])
        y = tf.nn.conv2d(
            x,
            k,
            strides=[1, 1, self.strides, 1],
            padding=self.padding.upper(),
        )
        if self.use_bias and self.bias is not None:
            y = tf.nn.bias_add(y, self.bias)
        return tf.reshape(y, [-1, self._out_len, self.filters])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "use_bias": self.use_bias,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            }
        )
        return config


@register_keras_serializable(package="doccorner_v2")
class GlobalAveragePool2DAsAvgPool(layers.Layer):
    """GlobalAveragePooling2D replacement."""

    def __init__(self, impl: str = "avgpool", **kwargs):
        super().__init__(**kwargs)
        self.impl = str(impl).lower().strip()
        self._h = None
        self._w = None
        self._c = None
        self._filters_strided = None
        self._strides_strided = None

    def build(self, input_shape):
        h, w, c = input_shape[1], input_shape[2], input_shape[3]
        if h is None or w is None or c is None:
            raise ValueError("GlobalAveragePool2DAsAvgPool requires static H/W/C.")
        self._h = int(h)
        self._w = int(w)
        self._c = int(c)
        if self.impl == "dwconv_strided":
            hh = int(self._h)
            ww = int(self._w)
            filters = []
            strides = []
            factors = (8, 4, 2)
            while hh > 1 or ww > 1:
                if hh <= 1:
                    kh = 1
                else:
                    kh = next((f for f in factors if hh % f == 0), int(hh))
                if ww <= 1:
                    kw = 1
                else:
                    kw = next((f for f in factors if ww % f == 0), int(ww))
                k = np.ones((kh, kw, self._c, 1), dtype=np.float32) / float(kh * kw)
                filters.append(tf.constant(k, dtype=tf.float32, name=f"{self.name}_dwfilter_{kh}x{kw}"))
                strides.append([1, kh, kw, 1])
                hh //= kh
                ww //= kw
            self._filters_strided = filters
            self._strides_strided = strides
        super().build(input_shape)

    def call(self, inputs):
        if self._h is None or self._w is None or self._c is None:
            raise RuntimeError("GlobalAveragePool2DAsAvgPool is not built.")
        if self.impl == "dwconv_strided":
            if self._filters_strided is None or self._strides_strided is None:
                raise RuntimeError("GlobalAveragePool2DAsAvgPool dwconv_strided filters missing.")
            x = inputs
            for f, s in zip(self._filters_strided, self._strides_strided, strict=False):
                x = tf.nn.depthwise_conv2d(x, f, strides=s, padding="VALID")
            return tf.reshape(x, [-1, self._c])
        x = tf.nn.avg_pool2d(inputs, ksize=(self._h, self._w), strides=(1, 1), padding="VALID")
        return tf.reshape(x, [-1, self._c])

    def get_config(self):
        config = super().get_config()
        config["impl"] = self.impl
        return config


@register_keras_serializable(package="doccorner_v2")
class NearestUpsample2x(layers.Layer):
    """2x nearest-neighbor upsampling using reshape+broadcast (XNNPACK-friendly)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._h = None
        self._w = None
        self._c = None

    def build(self, input_shape):
        h, w, c = input_shape[1], input_shape[2], input_shape[3]
        if h is None or w is None or c is None:
            raise ValueError("NearestUpsample2x requires static H/W/C.")
        self._h = int(h)
        self._w = int(w)
        self._c = int(c)
        super().build(input_shape)

    def call(self, inputs):
        if self._h is None:
            raise RuntimeError("NearestUpsample2x is not built.")
        # Repeat width: [B,H,W,C] → [B,H,W,1,C] * ones → [B,H,2W,C]
        ones_w = tf.ones([1, 1, 1, 2, 1], dtype=inputs.dtype)
        ones_h = tf.ones([1, 1, 2, 1, 1], dtype=inputs.dtype)
        x = tf.reshape(inputs, [-1, self._h, self._w, 1, self._c])
        x = x * ones_w
        x = tf.reshape(x, [-1, self._h, self._w * 2, self._c])
        x = tf.reshape(x, [-1, self._h, 1, self._w * 2, self._c])
        x = x * ones_h
        return tf.reshape(x, [-1, self._h * 2, self._w * 2, self._c])

    def get_config(self):
        return super().get_config()


@register_keras_serializable(package="doccorner_v2")
class SimCCDecode(layers.Layer):
    """Decode SimCC logits → normalized coordinates via soft-argmax.

    Inputs: [simcc_x, simcc_y] each [B, 4, num_bins]
    Output: [B, 8] coords (x0,y0,x1,y1,x2,y2,x3,y3) in [0,1]
    """

    def __init__(self, num_bins: int = 224, tau: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_bins = int(num_bins)
        self.tau = float(tau)
        self._centers_col = None

    def build(self, input_shape):
        centers = np.linspace(0.0, 1.0, self.num_bins, dtype=np.float32).reshape(self.num_bins, 1)
        self._centers_col = tf.constant(centers, dtype=tf.float32, name=f"{self.name}_centers")
        super().build(input_shape)

    def call(self, inputs):
        if self._centers_col is None:
            raise RuntimeError("SimCCDecode is not built.")
        sx, sy = inputs  # [B, 4, num_bins]
        sx = tf.cast(sx, tf.float32)
        sy = tf.cast(sy, tf.float32)
        px = tf.nn.softmax(sx / self.tau, axis=-1)
        py = tf.nn.softmax(sy / self.tau, axis=-1)
        px2 = tf.reshape(px, [-1, self.num_bins])
        py2 = tf.reshape(py, [-1, self.num_bins])
        x = tf.reshape(tf.matmul(px2, self._centers_col), [-1, 4])
        y = tf.reshape(tf.matmul(py2, self._centers_col), [-1, 4])
        xy = tf.concat([tf.reshape(x, [-1, 4, 1]), tf.reshape(y, [-1, 4, 1])], axis=-1)
        coords = tf.reshape(xy, [-1, 8])
        return tf.clip_by_value(coords, 0.0, 1.0)

    def get_config(self):
        config = super().get_config()
        config.update({"num_bins": self.num_bins, "tau": self.tau})
        return config


@register_keras_serializable(package="doccorner_v2")
class SpatialReduceLogSumExp(layers.Layer):
    """Reduce a spatial axis of an NHWC tensor via log-sum-exp pooling."""

    def __init__(self, axis: int, **kwargs):
        super().__init__(**kwargs)
        if axis not in (1, 2):
            raise ValueError(f"SpatialReduceLogSumExp axis must be 1 or 2, got {axis}")
        self.axis = int(axis)

    def call(self, inputs):
        return tf.reduce_logsumexp(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config["axis"] = self.axis
        return config


@register_keras_serializable(package="doccorner_v2")
class HeatmapOffsetDecode(layers.Layer):
    """Decode per-corner heatmaps + local offsets into normalized coordinates."""

    def __init__(self, tau: float = 1.0, offset_scale: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.tau = float(tau)
        self.offset_scale = float(offset_scale)
        self._h = None
        self._w = None
        self._hw = None
        self._grid_x = None
        self._grid_y = None

    def build(self, input_shape):
        heatmap_shape, offset_shape = input_shape
        h, w, c = heatmap_shape[1], heatmap_shape[2], heatmap_shape[3]
        oh, ow, oc = offset_shape[1], offset_shape[2], offset_shape[3]
        if h is None or w is None or c != 4:
            raise ValueError("HeatmapOffsetDecode requires heatmap shape [B,H,W,4].")
        if oh != h or ow != w or oc != 8:
            raise ValueError("HeatmapOffsetDecode requires offset shape [B,H,W,8] aligned to heatmaps.")

        self._h = int(h)
        self._w = int(w)
        self._hw = self._h * self._w

        x_centers = (np.arange(self._w, dtype=np.float32) + 0.5) / float(self._w)
        y_centers = (np.arange(self._h, dtype=np.float32) + 0.5) / float(self._h)
        grid_x, grid_y = np.meshgrid(x_centers, y_centers)
        self._grid_x = tf.constant(grid_x.reshape(1, 1, self._hw), dtype=tf.float32)
        self._grid_y = tf.constant(grid_y.reshape(1, 1, self._hw), dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        if self._grid_x is None:
            raise RuntimeError("HeatmapOffsetDecode is not built.")

        heatmap_logits, offset_map = inputs
        heatmap_logits = tf.cast(heatmap_logits, tf.float32)
        offset_map = tf.cast(offset_map, tf.float32)
        batch_size = tf.shape(heatmap_logits)[0]

        heatmap_logits = tf.transpose(heatmap_logits, [0, 3, 1, 2])  # [B, 4, H, W]
        heatmap_logits = tf.reshape(heatmap_logits, [batch_size, 4, self._hw])
        probs = tf.nn.softmax(heatmap_logits / self.tau, axis=-1)

        offset_map = tf.reshape(offset_map, [batch_size, self._h, self._w, 4, 2])
        offset_map = tf.transpose(offset_map, [0, 3, 1, 2, 4])  # [B, 4, H, W, 2]
        offset_map = tf.reshape(offset_map, [batch_size, 4, self._hw, 2])
        offset_map = tf.clip_by_value(offset_map, -1.0, 1.0) * self.offset_scale

        dx = offset_map[..., 0] / float(self._w)
        dy = offset_map[..., 1] / float(self._h)

        x = tf.reduce_sum(probs * (self._grid_x + dx), axis=-1)
        y = tf.reduce_sum(probs * (self._grid_y + dy), axis=-1)

        xy = tf.concat([tf.expand_dims(x, -1), tf.expand_dims(y, -1)], axis=-1)
        coords = tf.reshape(xy, [batch_size, 8])
        return tf.clip_by_value(coords, 0.0, 1.0)

    def get_config(self):
        config = super().get_config()
        config.update({"tau": self.tau, "offset_scale": self.offset_scale})
        return config


@register_keras_serializable(package="doccorner_v2")
class HeatmapOffsetRefine(layers.Layer):
    """Refine coarse normalized coordinates using heatmap-weighted local offsets."""

    def __init__(self, tau: float = 1.0, offset_scale: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.tau = float(tau)
        self.offset_scale = float(offset_scale)
        self._h = None
        self._w = None
        self._hw = None

    def build(self, input_shape):
        coarse_shape, heatmap_shape, offset_shape = input_shape
        if coarse_shape[-1] != 8:
            raise ValueError("HeatmapOffsetRefine requires coarse coords shape [B,8].")
        h, w, c = heatmap_shape[1], heatmap_shape[2], heatmap_shape[3]
        oh, ow, oc = offset_shape[1], offset_shape[2], offset_shape[3]
        if h is None or w is None or c != 4:
            raise ValueError("HeatmapOffsetRefine requires heatmap shape [B,H,W,4].")
        if oh != h or ow != w or oc != 8:
            raise ValueError("HeatmapOffsetRefine requires offset shape [B,H,W,8] aligned to heatmaps.")
        self._h = int(h)
        self._w = int(w)
        self._hw = self._h * self._w
        super().build(input_shape)

    def call(self, inputs):
        coarse_coords, heatmap_logits, offset_map = inputs
        coarse_coords = tf.cast(coarse_coords, tf.float32)
        heatmap_logits = tf.cast(heatmap_logits, tf.float32)
        offset_map = tf.cast(offset_map, tf.float32)
        batch_size = tf.shape(heatmap_logits)[0]

        heatmap_logits = tf.transpose(heatmap_logits, [0, 3, 1, 2])  # [B,4,H,W]
        heatmap_logits = tf.reshape(heatmap_logits, [batch_size, 4, self._hw])
        probs = tf.nn.softmax(heatmap_logits / self.tau, axis=-1)

        offset_map = tf.reshape(offset_map, [batch_size, self._h, self._w, 4, 2])
        offset_map = tf.transpose(offset_map, [0, 3, 1, 2, 4])
        offset_map = tf.reshape(offset_map, [batch_size, 4, self._hw, 2])
        offset_map = tf.clip_by_value(offset_map, -1.0, 1.0) * self.offset_scale

        dx = tf.reduce_sum(probs * (offset_map[..., 0] / float(self._w)), axis=-1)
        dy = tf.reduce_sum(probs * (offset_map[..., 1] / float(self._h)), axis=-1)
        delta = tf.reshape(tf.stack([dx, dy], axis=-1), [batch_size, 8])
        return tf.clip_by_value(coarse_coords + delta, 0.0, 1.0)

    def get_config(self):
        config = super().get_config()
        config.update({"tau": self.tau, "offset_scale": self.offset_scale})
        return config


# ---------------------------------------------------------------------------
# Backbone feature extraction
# ---------------------------------------------------------------------------

def _get_feature_layers(backbone, img_size: int):
    """Extract C2 (img/4), C3 (img/8), C4 (img/16), C5 (img/32) from backbone."""
    c2 = c3 = c4 = c5 = None
    targets = {
        img_size // 4: "c2",
        img_size // 8: "c3",
        img_size // 16: "c4",
        img_size // 32: "c5",
    }
    results = {}
    for layer in backbone.layers:
        out = layer.output
        if not hasattr(out, "shape") or len(out.shape) != 4:
            continue
        _, h, w, c = out.shape
        if h is None or w is None or h != w or (h == 1 and w == 1):
            continue
        if h in targets:
            results[targets[h]] = out

    c2 = results.get("c2")
    c3 = results.get("c3")
    c4 = results.get("c4")
    c5 = results.get("c5")
    if c2 is None or c3 is None or c4 is None or c5 is None:
        found = {k: (v.shape if v is not None else None) for k, v in results.items()}
        raise ValueError(f"Could not find all feature scales. Found: {found}")
    return c2, c3, c4, c5


def _build_backbone(inp, alpha, backbone_weights, backbone_include_preprocessing):
    """Build MobileNetV2 backbone."""
    if backbone_include_preprocessing:
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="backbone_preprocess")(inp)
        input_tensor = x
    else:
        input_tensor = inp
    return keras.applications.MobileNetV2(
        input_tensor=input_tensor,
        include_top=False,
        weights=backbone_weights,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _separable_conv_block(x, filters, name):
    """SepConv → BN → Swish."""
    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_sepconv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("swish", name=f"{name}_swish")(x)
    return x


# ---------------------------------------------------------------------------
# Main model builder
# ---------------------------------------------------------------------------

def build_doccorner_v2(
    alpha: float = 0.35,
    fpn_ch: int = 32,
    simcc_ch: int = 96,
    img_size: int = 224,
    num_bins: int = 224,
    tau: float = 1.0,
    score_init_bias: float = 1.75,
    backbone_weights="imagenet",
    backbone_include_preprocessing: bool = False,
    simcc_kernel_size: int = 5,
    xnnpack_safe: bool = False,
):
    """
    Build DocCornerNet V2 with corner-specific spatial attention.

    Args:
        alpha: MobileNetV2 width multiplier
        fpn_ch: FPN channels
        simcc_ch: SimCC head hidden channels
        img_size: Input image size (square)
        num_bins: Number of bins for SimCC coordinate classification
        tau: Softmax temperature for decode
        score_init_bias: Initial bias for score head
        backbone_weights: 'imagenet' or None
        backbone_include_preprocessing: Include Rescaling layer in model
        simcc_kernel_size: First Conv1D kernel size in SimCC head

    Returns:
        Keras Model with dict outputs
    """
    inp = keras.Input((img_size, img_size, 3), name="image")

    # -----------------------------------------------------------------------
    # Backbone: MobileNetV2 alpha=0.35
    # -----------------------------------------------------------------------
    backbone = _build_backbone(
        inp, alpha=alpha,
        backbone_weights=backbone_weights,
        backbone_include_preprocessing=backbone_include_preprocessing,
    )
    c2, c3, c4, c5 = _get_feature_layers(backbone, img_size)

    # -----------------------------------------------------------------------
    # Mini-FPN: top-down pathway → p_fused [B, 56, 56, fpn_ch]
    # -----------------------------------------------------------------------
    p4 = layers.Conv2D(fpn_ch, 1, padding="same", use_bias=False, name="fpn_lat_c4")(c4)
    p4 = layers.BatchNormalization(name="fpn_lat_c4_bn")(p4)

    p3 = layers.Conv2D(fpn_ch, 1, padding="same", use_bias=False, name="fpn_lat_c3")(c3)
    p3 = layers.BatchNormalization(name="fpn_lat_c3_bn")(p3)

    p2 = layers.Conv2D(fpn_ch, 1, padding="same", use_bias=False, name="fpn_lat_c2")(c2)
    p2 = layers.BatchNormalization(name="fpn_lat_c2_bn")(p2)

    # Top-down: P4 → P3
    p4_up = NearestUpsample2x(name="fpn_p4_up")(p4)
    p3 = layers.Add(name="fpn_p3_add")([p3, p4_up])
    p3 = _separable_conv_block(p3, fpn_ch, "fpn_p3_refine")

    # Top-down: P3 → P2
    p3_up = NearestUpsample2x(name="fpn_p3_up")(p3)
    p2 = layers.Add(name="fpn_p2_add")([p2, p3_up])
    p2 = _separable_conv_block(p2, fpn_ch, "fpn_p2_refine")  # [B, 56, 56, fpn_ch]

    # Multi-scale fusion: P2 + upsampled P3
    p3_up_feat = layers.UpSampling2D(size=2, interpolation="bilinear", name="simcc_p3_up")(p3)
    p_cat = layers.Concatenate(name="simcc_fuse")([p2, p3_up_feat])  # [B, 56, 56, 2*fpn_ch]
    p_fused = _separable_conv_block(p_cat, fpn_ch * 2, "simcc_refine1")
    p_fused = _separable_conv_block(p_fused, fpn_ch, "simcc_refine2")  # [B, 56, 56, fpn_ch]

    # -----------------------------------------------------------------------
    # Corner-Specific Spatial Attention + Hybrid SimCC/2D Refinement Head
    # -----------------------------------------------------------------------
    head_ch = max(int(fpn_ch), int(simcc_ch) // 2)
    shared = _separable_conv_block(p_fused, head_ch, "corner_precursor")

    corner_names = ["tl", "tr", "br", "bl"]
    heatmap_logits_list = []
    offset_logits_list = []
    x_logits_list = []
    y_logits_list = []

    # Shared SimCC layers (same weights reused for each corner).
    conv1d_layer = Conv1DAsConv2D if xnnpack_safe else layers.Conv1D

    def make_gap2d(name):
        if xnnpack_safe:
            return GlobalAveragePool2DAsAvgPool(impl="dwconv_strided", name=name)
        return layers.GlobalAveragePooling2D(name=name)

    simcc_x_conv1 = conv1d_layer(simcc_ch, simcc_kernel_size, padding="same", name="simcc_x_conv1")
    simcc_x_bn1 = layers.BatchNormalization(name="simcc_x_bn1")
    simcc_x_conv2 = conv1d_layer(simcc_ch // 2, 3, padding="same", name="simcc_x_conv2")
    simcc_x_bn2 = layers.BatchNormalization(name="simcc_x_bn2")
    simcc_x_out = conv1d_layer(
        1, 1,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=keras.initializers.Zeros(),
        name="simcc_x_out",
    )

    simcc_y_conv1 = conv1d_layer(simcc_ch, simcc_kernel_size, padding="same", name="simcc_y_conv1")
    simcc_y_bn1 = layers.BatchNormalization(name="simcc_y_bn1")
    simcc_y_conv2 = conv1d_layer(simcc_ch // 2, 3, padding="same", name="simcc_y_conv2")
    simcc_y_bn2 = layers.BatchNormalization(name="simcc_y_bn2")
    simcc_y_out = conv1d_layer(
        1, 1,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=keras.initializers.Zeros(),
        name="simcc_y_out",
    )

    global_feat = make_gap2d("simcc_global_gap")(p_fused)
    global_feat = layers.Dense(simcc_ch // 2, name="simcc_global_fc")(global_feat)
    global_feat = layers.ReLU(name="simcc_global_relu")(global_feat)
    global_x_bc = Broadcast1D(target_length=num_bins, name="global_x_broadcast")(global_feat)
    global_y_bc = Broadcast1D(target_length=num_bins, name="global_y_broadcast")(global_feat)

    axis_mean_x = AxisMean(axis=1, impl="dwconv_full" if xnnpack_safe else "avgpool", name="x_marginal_pool")
    axis_mean_y = AxisMean(axis=2, impl="dwconv_full" if xnnpack_safe else "avgpool", name="y_marginal_pool")
    resize_x = Resize1D(target_length=num_bins, name="x_marginal_resize")
    resize_y = Resize1D(target_length=num_bins, name="y_marginal_resize")

    for cn in corner_names:
        att = layers.Conv2D(1, 1, padding="same", name=f"att_{cn}_conv")(shared)
        att = layers.Activation("sigmoid", name=f"att_{cn}_sigmoid")(att)
        feat = layers.Multiply(name=f"att_{cn}_mul")([p_fused, att])

        hm_feat = _separable_conv_block(feat, head_ch, f"heatmap_{cn}_refine")
        hm_logit = layers.Conv2D(
            1, 1,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=keras.initializers.Zeros(),
            name=f"heatmap_{cn}_logit",
        )(hm_feat)
        heatmap_logits_list.append(hm_logit)

        off_feat = _separable_conv_block(feat, head_ch, f"offset_{cn}_refine")
        off_raw = layers.Conv2D(
            2, 1,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=keras.initializers.Zeros(),
            name=f"offset_{cn}_raw",
        )(off_feat)
        off = layers.Activation("tanh", name=f"offset_{cn}_tanh")(off_raw)
        offset_logits_list.append(off)

        # Keep the coarse SimCC path identical to the stable v2 recipe.
        # The 2D branch only refines the final coordinates through a small delta.
        x_marg = axis_mean_x(feat)
        y_marg = axis_mean_y(feat)
        x_marg = resize_x(x_marg)
        y_marg = resize_y(y_marg)

        xf = simcc_x_conv1(x_marg)
        xf = simcc_x_bn1(xf)
        xf = layers.ReLU(name=f"simcc_x_relu1_{cn}")(xf)
        xf = simcc_x_conv2(xf)
        xf = simcc_x_bn2(xf)
        xf = layers.ReLU(name=f"simcc_x_relu2_{cn}")(xf)
        xf = layers.Concatenate(name=f"simcc_x_cat_{cn}")([xf, global_x_bc])
        x_logits_list.append(simcc_x_out(xf))

        yf = simcc_y_conv1(y_marg)
        yf = simcc_y_bn1(yf)
        yf = layers.ReLU(name=f"simcc_y_relu1_{cn}")(yf)
        yf = simcc_y_conv2(yf)
        yf = simcc_y_bn2(yf)
        yf = layers.ReLU(name=f"simcc_y_relu2_{cn}")(yf)
        yf = layers.Concatenate(name=f"simcc_y_cat_{cn}")([yf, global_y_bc])
        y_logits_list.append(simcc_y_out(yf))

    corner_heatmap = layers.Concatenate(axis=-1, name="corner_heatmap")(heatmap_logits_list)
    corner_offset = layers.Concatenate(axis=-1, name="corner_offset")(offset_logits_list)

    simcc_x = layers.Concatenate(axis=-1, name="simcc_x_stack")(x_logits_list)
    simcc_x = layers.Permute((2, 1), name="simcc_x")(simcc_x)
    simcc_y = layers.Concatenate(axis=-1, name="simcc_y_stack")(y_logits_list)
    simcc_y = layers.Permute((2, 1), name="simcc_y")(simcc_y)

    coarse_coords = SimCCDecode(num_bins=num_bins, tau=tau, name="coords_coarse")([simcc_x, simcc_y])
    coords_refined = HeatmapOffsetRefine(tau=tau, offset_scale=0.05, name="coords_refined")(
        [coarse_coords, corner_heatmap, corner_offset]
    )
    coords_2d = HeatmapOffsetDecode(tau=tau, offset_scale=0.5, name="coords_2d")(
        [corner_heatmap, corner_offset]
    )
    coords = layers.Identity(name="coords")(coords_refined)

    # -----------------------------------------------------------------------
    # Score Head: document presence from C5
    # -----------------------------------------------------------------------
    score_features = make_gap2d("score_gap")(c5)
    score_logit = layers.Dense(
        1,
        bias_initializer=keras.initializers.Constant(score_init_bias),
        name="score_logit",
    )(score_features)

    # -----------------------------------------------------------------------
    # Build Model
    # -----------------------------------------------------------------------
    model = keras.Model(
        inputs=inp,
        outputs={
            "simcc_x": simcc_x,
            "simcc_y": simcc_y,
            "corner_heatmap": corner_heatmap,
            "corner_offset": corner_offset,
            "coords_2d": coords_2d,
            "score_logit": score_logit,
            "coords": coords,
        },
        name="DocCornerNet_V2",
    )
    return model


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_model(
    alpha: float = 0.35,
    fpn_ch: int = 32,
    simcc_ch: int = 96,
    img_size: int = 224,
    num_bins: int = 224,
    tau: float = 1.0,
    score_init_bias: float = 1.75,
    backbone_weights="imagenet",
    backbone_include_preprocessing: bool = False,
    simcc_kernel_size: int = 5,
    xnnpack_safe: bool = False,
) -> keras.Model:
    """Create DocCornerNet V2 training model with dict outputs."""
    return build_doccorner_v2(
        alpha=alpha,
        fpn_ch=fpn_ch,
        simcc_ch=simcc_ch,
        img_size=img_size,
        num_bins=num_bins,
        tau=tau,
        score_init_bias=score_init_bias,
        backbone_weights=backbone_weights,
        backbone_include_preprocessing=backbone_include_preprocessing,
        simcc_kernel_size=simcc_kernel_size,
        xnnpack_safe=xnnpack_safe,
    )


def create_inference_model(train_model: keras.Model) -> keras.Model:
    """Convert training model (dict outputs) to inference model ([coords, score_logit])."""
    coords = train_model.output["coords"]
    score_logit = train_model.output["score_logit"]
    return keras.Model(
        inputs=train_model.input,
        outputs=[coords, score_logit],
        name="DocCornerNet_V2_Inference",
    )


def load_inference_model(weights_path: str, **model_kwargs) -> keras.Model:
    """Create inference model and load weights."""
    train_model = create_model(backbone_weights=None, **model_kwargs)
    train_model.load_weights(weights_path)
    return create_inference_model(train_model)
