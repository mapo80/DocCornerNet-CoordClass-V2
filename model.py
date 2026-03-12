"""
DocCornerNet V2: Document Corner Detection with Corner-Specific Spatial Attention.

Architecture:
- Backbone: MobileNetV2 alpha=0.35 (224x224 input)
- Neck: Mini-FPN top-down (C2/C3/C4 → P2 fused at 56x56)
- Corner Attention: 4 lightweight spatial attention heads on shared precursor
- Head: Per-corner SimCC 1D coordinate classification with shared Conv1D weights
- Score: Document presence from C5 global pool
- Decode: Soft-argmax on per-corner logits → coords [B, 8]

Output dict keys:
  simcc_x:     [B, 4, num_bins]  X coordinate logits per corner
  simcc_y:     [B, 4, num_bins]  Y coordinate logits per corner
  score_logit: [B, 1]            Document presence logit
  coords:      [B, 8]            Decoded normalized coordinates (x0,y0,...,x3,y3)

Target: <1M parameters, IoU >= 0.99 at 224x224
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
    """Reduce a spatial axis of an NHWC tensor via mean pooling.

    axis=1 → vertical pool:   [B, H, W, C] → [B, W, C]
    axis=2 → horizontal pool: [B, H, W, C] → [B, H, C]
    """

    def __init__(self, axis: int, **kwargs):
        super().__init__(**kwargs)
        if axis not in (1, 2):
            raise ValueError(f"AxisMean axis must be 1 or 2, got {axis}")
        self.axis = int(axis)

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config["axis"] = self.axis
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
    """Broadcast [B, C] → [B, target_length, C] by tiling."""

    def __init__(self, target_length: int, **kwargs):
        super().__init__(**kwargs)
        self.target_length = int(target_length)

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=1)  # [B, 1, C]
        return tf.tile(x, [1, self.target_length, 1])

    def get_config(self):
        config = super().get_config()
        config["target_length"] = self.target_length
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
    # Corner-Specific Spatial Attention
    # -----------------------------------------------------------------------
    # Shared precursor: lightweight SepConv on fused features
    shared = _separable_conv_block(p_fused, fpn_ch, "corner_precursor")  # [B, 56, 56, fpn_ch]

    # 4 corner attention maps: Conv2D(1,1) → sigmoid
    corner_names = ["tl", "tr", "br", "bl"]
    attended_feats = []
    for cn in corner_names:
        att = layers.Conv2D(1, 1, padding="same", name=f"att_{cn}_conv")(shared)
        att = layers.Activation("sigmoid", name=f"att_{cn}_sigmoid")(att)  # [B, 56, 56, 1]
        feat = layers.Multiply(name=f"att_{cn}_mul")([p_fused, att])  # [B, 56, 56, fpn_ch]
        attended_feats.append(feat)

    # -----------------------------------------------------------------------
    # Per-corner SimCC path (shared Conv1D weights across corners)
    # -----------------------------------------------------------------------
    # Define shared layers (called once per corner = weight sharing)
    simcc_x_conv1 = layers.Conv1D(simcc_ch, simcc_kernel_size, padding="same", name="simcc_x_conv1")
    simcc_x_bn1 = layers.BatchNormalization(name="simcc_x_bn1")
    simcc_x_conv2 = layers.Conv1D(simcc_ch // 2, 3, padding="same", name="simcc_x_conv2")
    simcc_x_bn2 = layers.BatchNormalization(name="simcc_x_bn2")
    simcc_x_out = layers.Conv1D(
        1, 1,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=keras.initializers.Zeros(),
        name="simcc_x_out",
    )

    simcc_y_conv1 = layers.Conv1D(simcc_ch, simcc_kernel_size, padding="same", name="simcc_y_conv1")
    simcc_y_bn1 = layers.BatchNormalization(name="simcc_y_bn1")
    simcc_y_conv2 = layers.Conv1D(simcc_ch // 2, 3, padding="same", name="simcc_y_conv2")
    simcc_y_bn2 = layers.BatchNormalization(name="simcc_y_bn2")
    simcc_y_out = layers.Conv1D(
        1, 1,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=keras.initializers.Zeros(),
        name="simcc_y_out",
    )

    # Global context (shared across corners)
    global_feat = layers.GlobalAveragePooling2D(name="simcc_global_gap")(p_fused)  # [B, fpn_ch]
    global_feat = layers.Dense(simcc_ch // 2, name="simcc_global_fc")(global_feat)
    global_feat = layers.ReLU(name="simcc_global_relu")(global_feat)
    global_x_bc = Broadcast1D(target_length=num_bins, name="global_x_broadcast")(global_feat)
    global_y_bc = Broadcast1D(target_length=num_bins, name="global_y_broadcast")(global_feat)

    # Shared AxisMean layers
    axis_mean_x = AxisMean(axis=1, name="x_marginal_pool")  # reduce H → [B, W, C]
    axis_mean_y = AxisMean(axis=2, name="y_marginal_pool")  # reduce W → [B, H, C]
    resize_x = Resize1D(target_length=num_bins, name="x_marginal_resize")
    resize_y = Resize1D(target_length=num_bins, name="y_marginal_resize")

    x_logits_list = []
    y_logits_list = []

    for i, feat in enumerate(attended_feats):
        cn = corner_names[i]

        # Per-corner axis marginalization
        x_marg = axis_mean_x(feat)  # [B, W=56, fpn_ch]
        y_marg = axis_mean_y(feat)  # [B, H=56, fpn_ch]

        # Resize to num_bins
        x_marg = resize_x(x_marg)  # [B, num_bins, fpn_ch]
        y_marg = resize_y(y_marg)  # [B, num_bins, fpn_ch]

        # Shared SimCC Conv1D pipeline for X
        xf = simcc_x_conv1(x_marg)
        xf = simcc_x_bn1(xf)
        xf = layers.ReLU(name=f"simcc_x_relu1_{cn}")(xf)
        xf = simcc_x_conv2(xf)
        xf = simcc_x_bn2(xf)
        xf = layers.ReLU(name=f"simcc_x_relu2_{cn}")(xf)

        # Concat global context
        xf = layers.Concatenate(name=f"simcc_x_cat_{cn}")([xf, global_x_bc])  # [B, num_bins, simcc_ch]

        # Output: 1 logit per bin for this corner
        x_logit = simcc_x_out(xf)  # [B, num_bins, 1]
        x_logits_list.append(x_logit)

        # Shared SimCC Conv1D pipeline for Y
        yf = simcc_y_conv1(y_marg)
        yf = simcc_y_bn1(yf)
        yf = layers.ReLU(name=f"simcc_y_relu1_{cn}")(yf)
        yf = simcc_y_conv2(yf)
        yf = simcc_y_bn2(yf)
        yf = layers.ReLU(name=f"simcc_y_relu2_{cn}")(yf)

        yf = layers.Concatenate(name=f"simcc_y_cat_{cn}")([yf, global_y_bc])
        y_logit = simcc_y_out(yf)  # [B, num_bins, 1]
        y_logits_list.append(y_logit)

    # Stack corners: 4 × [B, num_bins, 1] → [B, num_bins, 4] → permute → [B, 4, num_bins]
    simcc_x = layers.Concatenate(axis=-1, name="simcc_x_stack")(x_logits_list)  # [B, num_bins, 4]
    simcc_x = layers.Permute((2, 1), name="simcc_x")(simcc_x)  # [B, 4, num_bins]

    simcc_y = layers.Concatenate(axis=-1, name="simcc_y_stack")(y_logits_list)
    simcc_y = layers.Permute((2, 1), name="simcc_y")(simcc_y)

    # Decode coordinates
    coords = SimCCDecode(num_bins=num_bins, tau=tau, name="coords")([simcc_x, simcc_y])

    # -----------------------------------------------------------------------
    # Score Head: document presence from C5
    # -----------------------------------------------------------------------
    score_features = layers.GlobalAveragePooling2D(name="score_gap")(c5)
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
