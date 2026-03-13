# DocCornerNet V2

Document corner detection with **corner-specific spatial attention**. Detects the four corners of a document in an image and classifies whether a document is present.

496,343 parameters | 224x224 input | TFLite-ready

## Architecture

```
Input [B, 224, 224, 3]
    |
    v
MobileNetV2 alpha=0.35  -->  C2(56x56), C3(28x28), C4(14x14), C5(7x7)
    |
    v
Mini-FPN top-down: P4->P3->P2 + multi-scale fusion  -->  p_fused [B, 56, 56, 32]
    |
    v
Shared SepConv2D precursor
    |
    +--- att_TL --> p_fused * sigmoid(Conv2D(1,1))  -->  feat_TL
    +--- att_TR --> p_fused * sigmoid(Conv2D(1,1))  -->  feat_TR
    +--- att_BR --> p_fused * sigmoid(Conv2D(1,1))  -->  feat_BR
    +--- att_BL --> p_fused * sigmoid(Conv2D(1,1))  -->  feat_BL
                        |
                        v
              Per-corner: AxisMean(H->X, W->Y) -> Resize1D -> shared Conv1D head
                        |
                        v
              4 x (x_logits, y_logits) [B, 4, num_bins]
                        |
                        v
              SimCCDecode (soft-argmax)  -->  coords [B, 8]

Parallel: C5 -> GAP -> Dense(1) -> score_logit [B, 1]
```

**Key idea**: each corner learns its own spatial attention map before axis marginalization, reducing ambiguity between the four corners while sharing the Conv1D classification weights.

## Requirements

- Python 3.10+
- TensorFlow 2.18+
- NumPy, Pillow
- Shapely 2.x (optional, for polygon IoU metrics)
- pytest, pytest-cov (for testing)

```bash
pip install tensorflow numpy pillow shapely pytest pytest-cov
```

## Dataset Structure

Two formats are supported (auto-detected):

**HuggingFace Parquet** (recommended):

```
dataset/
  train/
    data-00000-of-00001.parquet
  val/
    data-00000-of-00001.parquet
```

Parquet columns: `image` (struct with JPEG bytes), `filename`, `is_negative`, `corner_tl_x`, `corner_tl_y`, ..., `corner_bl_x`, `corner_bl_y`.

**File-based**:

```
dataset/
  images/          # Positive images (with documents)
  images-negative/ # Negative images (no document)
  labels/          # YOLO OBB format: class x0 y0 x1 y1 x2 y2 x3 y3
  train.txt        # Image filenames for training
  val.txt          # Image filenames for validation
```

Coordinates in label files are normalized to [0, 1]. Corner order: TL, TR, BR, BL.
Negative image filenames must be prefixed with `negative_`.

## Training

Basic training:

```bash
python -m train_ultra \
    --data_root /path/to/dataset \
    --output_dir runs/v2_experiment \
    --epochs 100 \
    --batch_size 32
```

Training with augmentation (recommended):

```bash
python -m train_ultra \
    --data_root /path/to/dataset \
    --output_dir runs/v2_aug \
    --epochs 100 \
    --batch_size 32 \
    --augment \
    --aug_factor 2 \
    --rotation_range 5.0 \
    --scale_range 0.0
```

Full training with all options:

```bash
python -m train_ultra \
    --data_root /path/to/dataset \
    --output_dir runs/v2_full \
    --epochs 100 \
    --batch_size 32 \
    --img_size 224 \
    --num_bins 224 \
    --learning_rate 2e-4 \
    --weight_decay 1e-4 \
    --warmup_epochs 5 \
    --sigma_px 2.0 \
    --loss_tau 0.5 \
    --w_simcc 1.0 \
    --w_coord 0.2 \
    --w_score 1.0 \
    --backbone_weights imagenet \
    --num_workers 32 \
    --augment \
    --aug_factor 2 \
    --rotation_range 5.0 \
    --scale_range 0.0 \
    --aug_weak_epochs 10
```

Smoke test on a small dataset (no pretrained backbone):

```bash
python -m train_ultra \
    --data_root /path/to/small_dataset \
    --output_dir runs/v2_smoke \
    --epochs 5 \
    --batch_size 16 \
    --backbone_weights none
```

Warm start from existing weights:

```bash
python -m train_ultra \
    --data_root /path/to/dataset \
    --output_dir runs/v2_finetune \
    --epochs 50 \
    --init_weights runs/v2_experiment/best_model.weights.h5
```

### Training Arguments

**Data & I/O:**

| Argument | Default | Description | Notes |
|---|---|---|---|
| `--data_root` | (required) | Dataset root directory | Auto-detects parquet vs file-based format |
| `--output_dir` | `./runs/v2` | Output directory for weights and logs | Created automatically if missing |
| `--input_norm` | `imagenet` | Input normalization (`imagenet`, `zero_one`, `raw255`) | Use `imagenet` with `--backbone_weights imagenet`. Use `raw255` for custom pipelines with no normalization |
| `--num_workers` | `32` | Threads for parallel parquet/data loading | Set to CPU core count. Diminishing returns above 32. Only affects initial data loading, not training |

**Model architecture:**

| Argument | Default | Description | Notes |
|---|---|---|---|
| `--alpha` | `0.35` | MobileNetV2 width multiplier | 0.35 = ~496K params. Higher (0.5, 1.0) increases capacity and latency. Keep 0.35 for TFLite mobile deployment |
| `--fpn_ch` | `32` | FPN feature channels | Tied to backbone output width at alpha=0.35. Increasing adds params quadratically in conv layers |
| `--simcc_ch` | `96` | SimCC Conv1D hidden channels | Controls capacity of the coordinate classification head. 64-128 reasonable range |
| `--img_size` | `224` | Input image size (square) | Must match `--num_bins` unless you want sub-pixel binning. 224 is standard MobileNet input |
| `--num_bins` | `224` | Number of SimCC coordinate bins | Resolution of coordinate output. Higher = finer precision but larger logit tensors. Typically equal to `--img_size` |
| `--tau` | `1.0` | SimCC decode temperature (soft-argmax) | <1.0 = sharper peaks (more confident), >1.0 = smoother. Affects inference only, not loss |
| `--simcc_kernel_size` | `5` | SimCC Conv1D kernel size | Receptive field of 1D classifier. 3-7 reasonable. Odd values only |
| `--backbone_weights` | `imagenet` | Backbone init (`imagenet` or `none`) | Always use `imagenet` for real training. Use `none` only for smoke tests or architecture debugging |

**Training schedule:**

| Argument | Default | Description | Notes |
|---|---|---|---|
| `--batch_size` | `32` | Batch size | 32-64 for small datasets (<5K). Scale LR linearly if increasing batch size (2e-4 calibrated for bs=32) |
| `--epochs` | `100` | Number of training epochs | Monitor val IoU plateau. Small datasets may overfit before 100 epochs |
| `--learning_rate` | `2e-4` | Base learning rate (cosine schedule with warmup) | Scale linearly with batch size. 2e-4 for bs=32, ~4e-4 for bs=64, ~8e-4 for bs=128 |
| `--weight_decay` | `1e-4` | AdamW weight decay | Increase to 5e-4 if overfitting (train loss << val loss). Decrease if underfitting |
| `--warmup_epochs` | `5` | Linear warmup epochs | 5-10% of total epochs. Prevents early divergence with pretrained backbone |
| `--init_weights` | `None` | Path to `.weights.h5` for warm start | Use for fine-tuning or resuming interrupted training. Loads all layers by name |

**Loss configuration:**

| Argument | Default | Description | Notes |
|---|---|---|---|
| `--sigma_px` | `2.0` | Gaussian target sigma in pixels | Controls target distribution width. 1.5-3.0 typical. Lower = sharper targets, harder to learn. Higher = easier but less precise |
| `--loss_tau` | `0.5` | Temperature for SimCC log-softmax loss | Separate from decode `--tau`. Lower = sharper gradients, can cause instability. 0.3-1.0 reasonable range |
| `--w_simcc` | `1.0` | SimCC cross-entropy loss weight | Primary corner loss. Keep at 1.0 as reference, adjust others relative to this |
| `--w_coord` | `0.2` | Coordinate L1 loss weight | Auxiliary direct supervision. 0.1-0.5 typical. Higher helps early convergence but can conflict with SimCC at fine scale |
| `--w_score` | `1.0` | Score BCE loss weight | Document presence classification. Reduce to 0.5 if dataset is heavily imbalanced (few negatives) |
| `--label_smoothing` | `0.0` | Label smoothing for SimCC targets (0=disabled) | 0.01-0.05 for large datasets to prevent overconfidence. Avoid on small datasets (<5K), slows convergence |

**Augmentation:**

| Argument | Default | Description | Notes |
|---|---|---|---|
| `--augment` | `false` | Enable data augmentation during training | Pass flag to activate. Augmentation is applied batch-wise in the training loop using TensorFlow ops (GPU-accelerated). Validation and test are never augmented |
| `--aug_factor` | `1` | Virtual train multiplier via repeated stochastic views per epoch | `1` = default. `>1` makes each epoch consume more augmented batches from the same base dataset without writing anything to disk. In the current implementation it requires `--augment`, `--aug_start_epoch 1`, and `--aug_min_iou 0.0` |
| `--rotation_range` | `5.0` | Random rotation range in degrees | Applied per sample, uniform in [-range, +range]. Requires `--augment`. Uses `ImageProjectiveTransformV3`; falls back to 0 if unavailable |
| `--scale_range` | `0.0` | Random scale range | 0.15 means uniform scale in [0.85, 1.0] in the current implementation (zoom-out only). Uses `crop_and_resize`. Start with 0.0, enable after smoke test on small dataset. Samples where coords go OOB are left unchanged |
| `--aug_weak_epochs` | `0` | Final N epochs use color-only augmentation | 0 = disabled. When active, the last N epochs switch from full augmentation (geometric + photometric) to photometric only. Helps stabilize training precision in final phase |
| `--aug_start_epoch` | `1` | Epoch from which augmentation can activate | 1 = immediate. Useful with `--init_weights` warm start to let the model stabilize before applying augmentation. Both `--aug_start_epoch` AND `--aug_min_iou` must be satisfied (AND logic) |
| `--aug_min_iou` | `0.0` | Min best IoU before augmentation activates | 0.0 = immediate. Once activated, augmentation stays active for the rest of training (latch). Combined with `--aug_start_epoch` via AND logic |

### Augmentation Pipeline

When `--augment` is active, the following augmentations are applied batch-wise on GPU at every training step (fresh random values each time):

**Photometric** (applied to all samples):

| Transform | Range | Notes |
|---|---|---|
| Brightness | delta in [-0.2, +0.2] * brightness_scale | brightness_scale adapts to `--input_norm` (1.0 for imagenet/zero_one, 255.0 for raw255) |
| Contrast | factor in [0.8, 1.2] | Applied around per-image mean |
| Saturation | factor in [0.85, 1.15] | Interpolation between grayscale and original |

**Geometric** (applied only to positive samples with `has_doc=1`):

| Transform | Range | Default | Notes |
|---|---|---|---|
| Horizontal flip | 50% probability | Always on | Correctly remaps corner order: TL<->TR, BL<->BR, x -> 1-x |
| Rotation | uniform [-range, +range] degrees | 5.0° | Image rotated via projective transform, coordinates via forward rotation matrix. Coords clipped to [0, 1] |
| Scale | uniform [1-range, 1.0] | 0.0 (disabled) | Zoom via `crop_and_resize` in the current implementation (zoom-out only). Samples where transformed coords exit [0, 1] are left unchanged |

**Clipping**: images are clipped to the normalization range (imagenet: [-3, 3], zero_one: [0, 1], raw255: [0, 255]). Coordinates are always clipped to [0, 1].

**Delayed activation** (`--aug_start_epoch` / `--aug_min_iou`): augmentation can be delayed until the model reaches a stable baseline. Both conditions must be met (AND logic). Once activated, augmentation stays active for the rest of training. Example: `--aug_start_epoch 3 --aug_min_iou 0.5` waits until epoch >= 3 AND best IoU >= 0.5.

**Dataset multiplication without filesystem** (`--aug_factor > 1`): the train dataset is repeated in-memory via `tf.data.repeat()` and each epoch consumes more batches from the same base samples. The extra diversity comes from the existing on-the-fly batch augmentation, so no augmented images are ever saved to disk. In the current implementation this mode requires augmentation to be active from epoch 1 (`--augment --aug_start_epoch 1 --aug_min_iou 0.0`).

**Weak augmentation** (`--aug_weak_epochs > 0`): in the final N epochs, geometric augmentations are disabled and only mild photometric augmentations are applied (brightness +-0.15, contrast [0.85, 1.15], saturation [0.85, 1.15]). This allows the model to refine precision on unperturbed geometry.

**Augmentation phases**: With all options enabled, training goes through: no aug (delayed start) -> full aug (geometric + photometric) -> weak aug (photometric only, final epochs).

**Note**: `augment_sample()` in `dataset.py` is a PIL-based utility for offline/debug use only. It applies photometric augmentations but no geometric transforms. The training loop uses `tf_augment_batch()` exclusively.

### Training Outputs

```
runs/v2_experiment/
  config.json                # Full training configuration
  best_model.weights.h5      # Best model (highest val IoU)
  final_model.weights.h5     # Final model (last epoch)
  training_log.csv           # Per-epoch metrics
```

## Evaluation

Loads a trained model, runs inference on a dataset split (val or test), and reports geometry metrics (IoU, corner error, recall) and classification metrics (accuracy, F1). Automatically reads `config.json` from the model directory to reconstruct the architecture.

Evaluate on validation set:

```bash
python -m evaluate \
    --model_path runs/v2_experiment \
    --data_root /path/to/dataset \
    --split val \
    --batch_size 32
```

Evaluate on test set:

```bash
python -m evaluate \
    --model_path runs/v2_experiment \
    --data_root /path/to/dataset \
    --split test
```

Evaluate a specific weights file:

```bash
python -m evaluate \
    --model_path runs/v2_experiment/best_model.weights.h5 \
    --data_root /path/to/dataset \
    --split val
```

### Evaluation Arguments

| Argument | Default | Description | Notes |
|---|---|---|---|
| `--model_path` | (required) | Path to model directory or `.weights.h5` file | If directory, auto-loads `config.json` + `best_model.weights.h5` |
| `--data_root` | (required) | Dataset root directory | Same format as training (parquet or file-based) |
| `--split` | `val` | Dataset split to evaluate | `train`, `val`, or `test` |
| `--batch_size` | `32` | Batch size for inference | Increase for faster evaluation on GPU |
| `--input_norm` | `imagenet` | Input normalization | Must match the normalization used during training |

**Model config overrides** (only needed if `config.json` is not available):

| Argument | Default | Description |
|---|---|---|
| `--alpha` | `0.35` | MobileNetV2 width multiplier |
| `--fpn_ch` | `32` | FPN feature channels |
| `--simcc_ch` | `96` | SimCC hidden channels |
| `--img_size` | `224` | Input image size |
| `--num_bins` | `224` | SimCC coordinate bins |
| `--tau` | `1.0` | Soft-argmax temperature |

### Evaluation Output

```
Geometry Metrics (on N images with documents):
  Mean IoU:             0.8748
  Median IoU:           0.8974
  Corner Error (mean):  7.12 px
  Corner Error (p95):   25.08 px
  Recall@90:            48.4%
  Recall@95:            11.5%
  Recall@99:            0.2%

Classification Metrics (on M total images):
  Accuracy:             99.8%
  F1 Score:             99.9%

Targets: IoU >= 99%: 87.48% [FAIL] | Error <= 1px: 7.12px [FAIL]
```

| Metric | Description |
|---|---|
| Mean/Median IoU | Polygon IoU between predicted and ground truth document corners |
| Corner Error | Mean Euclidean distance between predicted and GT corners in pixels |
| Corner Error p95 | 95th percentile of corner error (worst-case indicator) |
| Recall@90/95/99 | Fraction of samples with IoU above 90%/95%/99% |
| Accuracy/F1 | Document presence classification metrics |
| Targets | Pass/fail against production-quality thresholds (IoU >= 99%, error <= 1px) |

## Export

Converts a trained model to deployment formats (SavedModel, TFLite float32, TFLite int8, ONNX). Reads `config.json` from the weights directory to reconstruct the architecture automatically.

TFLite float32:

```bash
python -m export \
    --weights runs/v2_experiment \
    --output_dir exported \
    --format tflite
```

TFLite int8 with representative data:

```bash
python -m export \
    --weights runs/v2_experiment \
    --output_dir exported \
    --format tflite_int8 \
    --representative_data /path/to/dataset/images
```

SavedModel + TFLite:

```bash
python -m export \
    --weights runs/v2_experiment \
    --output_dir exported \
    --format savedmodel tflite
```

### Export Arguments

| Argument | Default | Description | Notes |
|---|---|---|---|
| `--weights` | (required) | Path to model directory or `.weights.h5` file | If directory, auto-loads `config.json` + `best_model.weights.h5` |
| `--output_dir` | `./exported_v2` | Output directory for exported models | Created automatically if missing |
| `--format` | `savedmodel tflite` | Export format(s) | One or more of: `savedmodel`, `tflite`, `tflite_int8`, `onnx`. Multiple formats can be specified |
| `--representative_data` | `None` | Path to images directory for int8 calibration | Required only for `tflite_int8`. Uses ~100 images to determine quantization ranges |

**Model config overrides** (only needed if `config.json` is not available):

| Argument | Default | Description |
|---|---|---|
| `--alpha` | `0.35` | MobileNetV2 width multiplier |
| `--fpn_ch` | `32` | FPN feature channels |
| `--simcc_ch` | `96` | SimCC hidden channels |
| `--img_size` | `224` | Input image size |
| `--num_bins` | `224` | SimCC coordinate bins |
| `--tau` | `1.0` | Soft-argmax temperature |

### Export Formats

| Format | File | Size (~) | Use case |
|---|---|---|---|
| `savedmodel` | `saved_model/` | ~2 MB | TF Serving, Python inference |
| `tflite` | `model.tflite` | ~600 KB | Mobile/edge (float32) |
| `tflite_int8` | `model_int8.tflite` | ~200 KB | Mobile/edge (quantized, fastest) |
| `onnx` | `model.onnx` | ~2 MB | Cross-framework inference |

## Tests

```bash
python -m pytest tests/ -v
```

With coverage:

```bash
python -m pytest tests/ --cov=. --cov-report=term-missing
```

200 tests, 93% coverage.

## Project Structure

```
v2/
  model.py          # Model architecture (487 lines)
  losses.py         # Loss functions and training wrapper
  dataset.py        # Data loading, augmentation, tf.data pipeline
  metrics.py        # Validation metrics (IoU, corner error, classification)
  train_ultra.py    # Training script (cross-platform)
  evaluate.py       # Evaluation script
  export.py         # TFLite/SavedModel export + benchmarking
  tests/            # 200 tests, 93% coverage
  V2_PROPOSAL.md    # Original design proposal
  IMPLEMENTATION_NOTES.md
  IMPLEMENTATION_TRACEABILITY.md
```

## Model Outputs

| Key | Shape | Description |
|---|---|---|
| `simcc_x` | `[B, 4, num_bins]` | X coordinate logits per corner |
| `simcc_y` | `[B, 4, num_bins]` | Y coordinate logits per corner |
| `score_logit` | `[B, 1]` | Document presence logit |
| `coords` | `[B, 8]` | Decoded normalized coordinates (x0,y0,...,x3,y3) |
