# DocCornerNet V2 — Implementation Traceability

Maps each proposal requirement from `V2_PROPOSAL.md` to its implementation in the v2 codebase.

## Architecture (§4.2 MVP)

| Proposal Requirement | Implementation | File:Lines |
|---|---|---|
| MobileNetV2 alpha=0.35 backbone | `_build_backbone()` with `keras.applications.MobileNetV2` | `model.py:208-220` |
| Feature extraction C2/C3/C4/C5 | `_get_feature_layers()` extracts at img/4, img/8, img/16, img/32 | `model.py:178-205` |
| Mini-FPN top-down P4→P3→P2 | Lateral convs + NearestUpsample2x + Add + SepConv refinement | `model.py:284-301` |
| p_fused [B, 56, 56, fpn_ch] | Multi-scale fusion: P2 concat upsampled P3, refined to fpn_ch | `model.py:303-307` |
| Shared SepConv2D precursor | `_separable_conv_block(p_fused, fpn_ch, "corner_precursor")` | `model.py:313` |
| 4 corner attention maps (§4.3) | Conv2D(1,1) → sigmoid → Multiply with p_fused, per corner | `model.py:316-322` |
| Per-corner AxisMean X/Y | Shared AxisMean layers called per attended feature | `model.py:358-375` |
| Per-corner Resize1D to num_bins | Shared Resize1D layers called per corner | `model.py:360-375` |
| Shared SimCC Conv1D head | Shared Conv1D layers (conv1→bn→relu→conv2→bn→relu→out) called 4× | `model.py:328-402` |
| SimCCDecode soft-argmax | Custom `SimCCDecode` layer with matmul against linspace centers | `model.py:133-171` |
| Score head on C5, invariata | GAP(C5) → Dense(1, bias=1.75) | `model.py:417-422` |
| Output: coords [B, 8] | Stack 4 corners → Permute → SimCCDecode → clip [0,1] | `model.py:404-412` |
| ~497K parameters (§7) | 496,343 actual | Verified via `model.count_params()` |

## Custom Layers (§4.3, §8)

| Layer | Purpose | Proposal Reference | File:Lines |
|---|---|---|---|
| `AxisMean` | Spatial axis mean pooling | §4.2 per-corner X/Y marginalization | `model.py:32-52` |
| `Resize1D` | Bilinear 1D resize | §4.2 Resize1D to num_bins | `model.py:55-76` |
| `Broadcast1D` | Tile scalar to sequence | Global context injection | `model.py:79-94` |
| `NearestUpsample2x` | XNNPACK-friendly 2x upsample | §8 XNNPACK compatibility | `model.py:97-131` |
| `SimCCDecode` | Soft-argmax coordinate decode | §4.2 coords output | `model.py:133-171` |

All registered with `@register_keras_serializable(package="doccorner_v2")`.

## Loss Functions (§6)

| Proposal Requirement | Implementation | File:Lines |
|---|---|---|
| SimCC Gaussian CE as primary loss (§6.1) | `SimCCLoss` with `gaussian_1d_targets()` | `losses.py:15-76` |
| loss_tau separate from decode tau (§6.2) | `SimCCLoss.tau` parameter, default 0.5 at training | `losses.py:44-48` |
| Coordinate loss L1 or SmoothL1 (§6.1) | `CoordLoss` supporting both `l1` and `smooth_l1` | `losses.py:79-108` |
| Score BCE loss | `sigmoid_cross_entropy_with_logits` in Trainer | `losses.py:186-189` |
| Proper masking: SimCC/coord on positives only | `mask` parameter in SimCCLoss and CoordLoss | `losses.py:51,86` |
| Score loss on all samples | No mask applied to score BCE | `losses.py:186-189` |
| sigma_px configurable (§6.2) | `gaussian_1d_targets(sigma_px=...)` | `losses.py:15` |
| label_smoothing option | Blend with uniform in `gaussian_1d_targets` | `losses.py:34-36` |
| Loss weights w_simcc, w_coord, w_score | Configurable in `DocCornerNetV2Trainer.__init__` | `losses.py:131-134` |

## Training (§6.3, §10)

| Proposal Requirement | Implementation | File:Lines |
|---|---|---|
| Cosine schedule with warmup (§6.3 Phase 1) | `WarmupCosineSchedule` class | `train_ultra.py:186-215` |
| AdamW optimizer | `keras.optimizers.AdamW` with weight_decay | `train_ultra.py:349-352` |
| Full augmentation (§6.3) | Color + geometric augmentation in `dataset.py` | `dataset.py:78-144` |
| num_bins configurable (§5 Step A) | `--num_bins` CLI arg, flows to model and losses | `train_ultra.py:240` |
| sigma_px configurable (§5 Step A) | `--sigma_px` CLI arg | `train_ultra.py:253` |
| loss_tau configurable (§5 Step A) | `--loss_tau` CLI arg, default 0.5 per §5 | `train_ultra.py:254` |
| w_coord configurable (§5 Step A) | `--w_coord` CLI arg | `train_ultra.py:256` |
| backbone_weights=none option | `--backbone_weights none` strips ImageNet | `train_ultra.py:307-309` |
| init_weights warm start | `--init_weights` loads pretrained weights | `train_ultra.py:322-325` |
| Best model saving on val IoU | Save on `iou > best_iou` | `train_ultra.py:400-403` |
| Training log CSV | `training_log.csv` with epoch metrics | `train_ultra.py:417-422` |
| Config saved to output_dir | `config.json` written at training start | `train_ultra.py:279-282` |

## Dataset (§1, §2.1)

| Proposal Requirement | Implementation | File:Lines |
|---|---|---|
| YOLO OBB label format | `load_label_yolo_obb()` parsing class + 8 coords | `dataset.py:57-67` |
| ImageNet normalization | `IMAGENET_MEAN`, `IMAGENET_STD`, `normalize_image()` | `dataset.py:27-28, 147-159` |
| Negative samples support | `images-negative/` directory, `negative_` prefix | `dataset.py:213-256` |
| Split files (train.txt, val.txt) | `load_split_file()` with fallback suffixes | `dataset.py:44-54, 198-206` |
| 224×224 input (§2.2) | `img_size=224` default throughout | `dataset.py:165` |

## Evaluation (§9.2)

| Proposal Requirement | Implementation | File:Lines |
|---|---|---|
| mIoU metric | `ValidationMetrics.compute()["mean_iou"]` | `metrics.py:239` |
| Corner error in pixels | `compute_corner_error()` and accumulated in ValidationMetrics | `metrics.py:85-101, 232-237` |
| Recall@90/95/99 | Computed from per-sample IoU | `metrics.py:246-250` |
| Classification accuracy/F1 | TP/FP/TN/FN from score predictions | `metrics.py:170-186` |
| Polygon IoU with Shapely | `compute_polygon_iou()`, vectorized path with fallback | `metrics.py:42-57, 196-215` |
| Bbox IoU fallback | `compute_bbox_iou()` when Shapely unavailable | `metrics.py:60-82` |
| Config loading from model dir | `_find_config_path()` + `load_model()` | `evaluate.py:49-110` |

## Export & Deploy (§8, §9.3)

| Proposal Requirement | Implementation | File:Lines |
|---|---|---|
| TFLite export | `export_tflite()` float32 and int8 | `export.py:125-157` |
| SavedModel export | `export_savedmodel()` | `export.py:113-122` |
| INT8 quantization with representative data | Optional `representative_dataset` generator | `export.py:135-149` |
| Size < 1MB target (§2.2) | Measured at export, reported in results JSON | `export.py:155-156` |
| Inference model (coords + score only) | `create_inference_model()` strips SimCC logits | `model.py:471-479` |
| Benchmark latency | `benchmark_tflite()` with warmup + timing | `export.py:160-190` |

## Excluded Items (§4.4)

| Excluded per Proposal | Status |
|---|---|
| PAN-Lite bottom-up path (§5 Step C) | Not implemented — deferred |
| Cross-corner context (§5 Step D) | Not implemented — backlog |
| Adaptive Wing Loss (§3) | Not implemented — backlog |
| DFL (§3) | Not implemented — backlog |
| EMA (§5 Step A) | Not implemented — training-only ablation, deferred |
| Backbone change (§4.4) | Not changed — MobileNetV2 alpha=0.35 preserved |
| Score head modification (§4.4) | Not changed — preserved from v1 design |

## Test Coverage

| Module | Coverage | Key Test Files |
|---|---|---|
| `model.py` | 97% | `test_model.py`, `test_additional_coverage.py` |
| `losses.py` | 100% | `test_losses.py`, `test_additional_coverage.py` |
| `dataset.py` | 97% | `test_dataset.py`, `test_additional_coverage.py` |
| `metrics.py` | 83% | `test_metrics.py`, `test_main_functions.py` |
| `evaluate.py` | 99% | `test_evaluate_export_coverage.py`, `test_main_functions.py` |
| `export.py` | 84% | `test_cli_coverage.py`, `test_main_functions.py`, `test_evaluate_export_coverage.py` |
| `train_ultra.py` | 92% | `test_cli_coverage.py`, `test_main_functions.py`, `test_evaluate_export_coverage.py` |
| **TOTAL** | **93%** | 179 tests passing |

## No v1 Runtime Dependencies

Verified: `rg -n "from v1|import v1" v2/` returns zero matches. All v2 modules are self-contained.
