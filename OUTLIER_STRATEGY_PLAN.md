# Outlier Strategy Plan

Version: 0.1  
Status: Analysis complete  
Owner: Codex review  
Date: 2026-03-15

## Objective

Explain the large `validation -> test` gap for the current best checkpoint and define the highest-impact next strategy to improve real-world performance.

Current checkpoint:
- `runs/v3_full_losssup_continue/best_model.weights.h5`

Observed metrics:
- `validation mean IoU = 0.9821`
- `test mean IoU = 0.8579`

This is a real gap, not an evaluation bug.

---

## Key Findings

## 1. The gap is real

Using the same checkpoint and the same evaluation script:
- `val`: `mean_iou = 0.9821`, `median_iou = 0.9876`, `corner_error = 1.15 px`
- `test`: `mean_iou = 0.8579`, `median_iou = 0.9536`, `corner_error = 8.93 px`

Conclusion:
- evaluation code is not the problem
- the issue is data composition and domain difficulty

---

## 2. Test is significantly harder than validation

Positive-sample geometry is harder in `test`:

- bbox area p10
  - `val = 0.229`
  - `test = 0.145`
- polygon area p10
  - `val = 0.199`
  - `test = 0.111`

Interpretation:
- `test` contains many more small documents
- under `224x224`, small documents are much harder for corner precision

---

## 3. Test is reweighted toward hard sources

Several difficult domains are heavily underrepresented in `train/val` and strongly overrepresented in `test`.

Examples:

| Source | Train share | Val share | Test share |
|---|---:|---:|---:|
| `idcard_final_polygon` | `0.31%` | `0.20%` | `5.27%` |
| `idcard_jj` | `0.54%` | `0.43%` | `5.25%` |
| `idcard_g2wbl` | `0.47%` | `0.63%` | `4.42%` |
| `segrec` | `0.26%` | `0.30%` | `2.64%` |
| `book_page_segmentation_tecgp` | `1.74%` | `1.72%` | `2.54%` |
| `receipt_occam` | `0.23%` | `0.28%` | `0.92%` |
| `receipt_segmentation_z4vqo` | `0.27%` | `0.08%` | `1.28%` |
| `receipts_segmentation` | `0.22%` | `0.24%` | `1.16%` |
| `receipts_coolstuff` | `0.24%` | `0.21%` | `1.24%` |

Meanwhile easier or well-covered sources lose relative weight in `test`:
- `midv500`: `19.20%` in `val`, only `7.91%` in `test`
- `id_detections`: `14.02%` in `val`, only `8.73%` in `test`

Conclusion:
- `validation` is much more favorable than `test`
- `test` is not unseen-OOD, but it is strongly reweighted toward harder domains

---

## 4. The test gap is concentrated, not diffuse

Worst `test` sources:

| Source | Positives | Mean IoU | Notes |
|---|---:|---:|---|
| `documento` | `10` | `0.4895` | tiny, very rare |
| `idcard_final_polygon` | `309` | `0.5737` | catastrophic, many FN |
| `documento_nlsft` | `12` | `0.6192` | tiny, very rare |
| `seg_paper` | `97` | `0.6672` | paper-like hard cases |
| `receipts_segmentation` | `68` | `0.7062` | receipt family |
| `receipt_occam` | `54` | `0.7071` | receipt family |
| `receipt_segmentation_z4vqo` | `75` | `0.7272` | receipt family |
| `idcard_jj` | `308` | `0.7328` | strong test failure |
| `segrec` | `155` | `0.7343` | strong test failure |
| `documents_segmentation` | `80` | `0.7362` | harder paper family |
| `receipts_coolstuff` | `73` | `0.7413` | receipt family |
| `idcard_g2wbl` | `259` | `0.7479` | strong test failure |

Impact estimate:
- these worst 12 sources are about `25.6%` of test positives
- removing them raises estimated test mean IoU from `0.8579` to about `0.9141`
- the 3 `idcard_*` sources alone are about `14.9%` of test positives
- removing just those 3 raises estimated mean IoU to about `0.8889`

Conclusion:
- the gap is driven by a concentrated tail of hard domains

---

## 5. Validation outliers exist, but they are not the right training target

Worst `val` sources inside the worst-100 validation examples:
- `more2`
- `filtered_document_segmentation`
- `midv500`
- `ktp_sim_passport_segmentation`
- `segrec`
- `id_detections`
- `dccc_vn_instseg`
- `receipt_detection`
- `document_segmentation_tzzh7`
- `midv2019`

Worst individual `val` samples:
- `more2_001757.jpg`
- `filtered_document_segmentation_003806.jpg`
- `receipt_occam_000052.jpg`
- `segrec_003013.jpg`
- `filtered_document_segmentation_001777.jpg`
- `receipts_segmentation_000586.jpg`
- `more2_002894.jpg`
- `id_detections_007492.jpg`
- `segrec_001482.jpg`
- `idcard_jj_004845.jpg`

Important observation:
- some overlap exists (`segrec`, `receipt*`, `idcard_jj`)
- but the main test killers, especially `idcard_final_polygon` and `idcard_g2wbl`, are not strongly exposed by `val` outliers
- in validation, `idcard_final_polygon` actually looks fine overall:
  - `val mean_iou = 0.9665`
  - but in test it collapses to `0.5737`

Conclusion:
- **using validation outliers directly as a training target would be methodologically wrong and strategically weak**
- it would optimize validation even further
- it would still miss major test failure domains

---

## Should We Do Validation-Outlier-Specific Augmentation?

Short answer:
- **No, not as the main strategy**

Reasons:
- it leaks validation into training decisions
- it optimizes the already-favorable validation split
- it misses the main test gap drivers
- it is a weak proxy for the real failure distribution

Validation outliers are still useful:
- for diagnosis
- for building a harder evaluation subset
- for sanity checks on regression

But they should not drive the main training recipe.

---

## Best Strategy

The best next strategy is:

**domain-aware hard-case training based on train-side source families that correlate with the real test failures**

Not:
- validation-outlier-specific augmentation
- dynamic validation-guided augmentation
- global stronger augmentation

---

## Recommended Plan

## Phase 1: Freeze the current best baseline

Keep the current best checkpoint as baseline.

Do not overwrite its role as reference.

---

## Phase 2: Create a hard-domain taxonomy

Primary hard groups from test analysis:
- `idcard_final_polygon`
- `idcard_jj`
- `idcard_g2wbl`
- `segrec`
- `receipt_occam`
- `receipt_segmentation_z4vqo`
- `receipts_segmentation`
- `receipts_coolstuff`
- `seg_paper`
- `documents_segmentation`
- `book_page_segmentation_tecgp`

Secondary diagnostic overlap from validation:
- `more2`
- `filtered_document_segmentation`
- `ktp_sim_passport_segmentation`
- `midv2019`

---

## Phase 3: Do not augment validation outliers directly

Instead:
- use validation outliers only to define:
  - `val_hard`
  - source-level dashboards
  - regression checks

Recommended `val_hard`:
- worst `200` validation positives by IoU
- plus source-balanced inclusion of overlapping hard families like:
  - `segrec`
  - `receipt*`
  - `idcard_jj`

Purpose:
- model selection
- not training

---

## Phase 4: Train on hard domains from the training split

Main lever:
- source-aware oversampling or weighting on `train`

Why this is better than val-outlier augmentation:
- no validation leakage
- targets the real weak domains
- works even when `val` underexposes a source

Priority order:
1. oversample `idcard_*` hard groups
2. oversample `receipt*` groups
3. oversample `book/page/paper` hard groups
4. keep strong representation of broad generalist groups like `midv500` and `id_detections`

This should be a balanced curriculum, not a full collapse into hard-only training.

---

## Phase 5: Be careful with augmentation choice

Important constraint from current pipeline:
- current `scale_range` only performs `1.0 -> smaller`
- it does **not** zoom in

That means:
- current scale augmentation is poorly aligned with the test failure mode
- `test` is already harder because documents are smaller
- making documents smaller in training is unlikely to be the main fix

Implication:
- do **not** center the next strategy on current scale augmentation
- if augmentation is used, keep it conservative

What likely helps more:
- source-aware sampling
- not stronger generic augmentation

If augmentation is tested at all:
- keep `rotation` conservative
- keep `scale` at `0.0` initially
- only consider changing this if the augmentation implementation supports useful zoom-in behavior

---

## Phase 6: Evaluate per source, not just globally

For every next experiment, track:
- global validation
- `val_hard`
- per-source validation for the known hard families

If test has already been consulted:
- do not keep tuning against test repeatedly
- use the current test reading as diagnostic guidance only

---

## Why This Is the Best Next Strategy

This strategy is best because it attacks the actual structure of the failure:
- concentrated source-level weakness
- small-document weakness
- domain mixture shift between `val` and `test`

It avoids the wrong optimization target:
- further fitting an easy validation split

It also avoids a weak lever:
- generic stronger augmentation

---

## Final Recommendation

Best next move:

**Do not build a training recipe around validation outliers.**

Instead:

1. keep validation outliers only for diagnostics and `val_hard`
2. build the next training iteration around train-side hard-domain oversampling
3. focus first on `idcard_*`, `receipt*`, `paper/book` families
4. avoid relying on current scale augmentation as a primary fix
5. evaluate per source, not just on aggregate IoU

In one line:

**The right strategy is hard-domain rebalancing, not validation-outlier-specific augmentation.**
