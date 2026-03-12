# DocCornerNet V2 — Implementation Notes

## Architecture Overview

DocCornerNet V2 implements **corner-specific spatial attention** as the single architectural hypothesis from `V2_PROPOSAL.md` §4.2–4.3. The model separates per-corner information early in the pipeline, before SimCC coordinate classification, while keeping the backbone, FPN neck, and score head unchanged from v1's design philosophy.

```
Input [B, 224, 224, 3]
    │
    ▼
MobileNetV2 alpha=0.35  →  C2(56×56), C3(28×28), C4(14×14), C5(7×7)
    │
    ▼
Mini-FPN top-down: P4→P3→P2 + multi-scale fusion  →  p_fused [B, 56, 56, 32]
    │
    ▼
Shared SepConv2D precursor  →  [B, 56, 56, 32]
    │
    ├─── att_TL ──→ p_fused * σ(Conv2D(1,1))  →  feat_TL
    ├─── att_TR ──→ p_fused * σ(Conv2D(1,1))  →  feat_TR
    ├─── att_BR ──→ p_fused * σ(Conv2D(1,1))  →  feat_BR
    └─── att_BL ──→ p_fused * σ(Conv2D(1,1))  →  feat_BL
                        │
                        ▼
              Per-corner: AxisMean(H→X, W→Y) → Resize1D → shared Conv1D pipeline
                        │
                        ▼
              4 × (x_logits, y_logits) [B, 4, num_bins]
                        │
                        ▼
              SimCCDecode (soft-argmax)  →  coords [B, 8]

Parallel: C5 → GAP → Dense(1) → score_logit [B, 1]
```

## Parameter Count

**496,343 parameters** — matches the proposal estimate of ~497K (§7).

The corner attention adds ~1.5K parameters over v1's 495,353 (4 × Conv2D(1,1) attention maps + shared SepConv2D precursor). The SimCC Conv1D heads share weights across all 4 corners, so compute increases ~4x for spatial operations but parameter count stays nearly flat.

## Key Design Decisions

### 1. Corner Attention (§4.3)
Each corner gets its own learned spatial attention map (sigmoid gating) applied to the shared fused features. This lets each corner "focus" on its relevant spatial region before axis marginalization, addressing the hypothesis that shared features create ambiguity between corners.

### 2. Shared SimCC Weights
The Conv1D pipeline (conv1→bn→relu→conv2→bn→relu→out) is defined once and called 4 times. Keras functional API handles weight sharing automatically — same layer objects applied to different inputs. This keeps parameters low while giving each corner its own data path.

### 3. Global Context Injection
A global context vector (GAP on p_fused → Dense → Broadcast1D) is concatenated with each corner's 1D features before the final output Conv1D. This gives each corner access to whole-image context without breaking per-corner separation.

### 4. Multi-scale Fusion
Beyond the proposal's basic FPN, the implementation adds an extra fusion step: P2 is concatenated with bilinear-upsampled P3 features, then refined through two SepConv blocks. This provides richer multi-scale features at the 56×56 resolution where attention operates.

### 5. Custom Layers
All custom layers are registered with `@register_keras_serializable(package="doccorner_v2")`:
- **AxisMean**: Mean pooling along H or W axis (marginalization)
- **Resize1D**: Bilinear resize of 1D sequences via tf.image.resize
- **Broadcast1D**: Tile [B, C] → [B, L, C] for context injection
- **NearestUpsample2x**: XNNPACK-friendly nearest-neighbor 2x upsampling via reshape+broadcast
- **SimCCDecode**: Soft-argmax decoding of per-corner logits to normalized coordinates

### 6. Score Head
Unchanged from v1 design: GlobalAveragePooling2D on C5 → Dense(1) with bias initialized to 1.75 (prior toward "has document"). This is intentionally not modified per §4.4.

## Loss Design

Three loss components with proper masking:
- **SimCC Gaussian CE** (`w_simcc=1.0`): Cross-entropy between predicted logits and Gaussian target distributions. Applied only to positive samples. Uses `loss_tau` (default 0.5) as temperature for log-softmax, separate from decode `tau`.
- **CoordLoss L1** (`w_coord=0.2`): Direct L1 supervision on decoded coordinates. Applied only to positive samples.
- **Score BCE** (`w_score=1.0`): Binary cross-entropy for document presence. Applied to all samples.

## Training Configuration

- Cosine annealing with linear warmup (`WarmupCosineSchedule`)
- AdamW optimizer with weight decay
- Threaded data loading (`ThreadPoolExecutor`)
- ImageNet normalization by default
- Best model saved on validation IoU improvement

## Export Support

- **TFLite float32**: Direct conversion from Keras inference model
- **TFLite int8**: Dynamic range quantization (full int8 with representative dataset optional)
- **SavedModel**: Standard TF SavedModel format
- **Benchmarking**: Built-in TFLite inference timing

## What Is NOT in V2

Per §4.4 of the proposal:
- No PAN-Lite bottom-up path (Step C, deferred)
- No cross-corner context (Step D, backlog)
- No Adaptive Wing Loss or DFL
- No backbone change
- No input resolution change from 224×224
