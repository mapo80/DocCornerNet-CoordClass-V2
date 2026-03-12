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
    --num_workers 32
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

### Training Outputs

```
runs/v2_experiment/
  config.json                # Full training configuration
  best_model.weights.h5      # Best model (highest val IoU)
  final_model.weights.h5     # Final model (last epoch)
  training_log.csv           # Per-epoch metrics
```

## Evaluation

```bash
python -m evaluate \
    --model_path runs/v2_experiment \
    --data_root /path/to/dataset \
    --split val \
    --batch_size 32
```

Reports: mean/median IoU, corner error (min, mean, p95, max), recall@50/75/90/95, classification accuracy/F1.

## Export

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

## Tests

```bash
python -m pytest tests/ -v
```

With coverage:

```bash
python -m pytest tests/ --cov=. --cov-report=term-missing
```

179 tests, 93% coverage.

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
  tests/            # 179 tests, 93% coverage
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
