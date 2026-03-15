# V2 Best Full Source-Balance Current

Source run:
- `runs/v4_full_sourcebalance_lossonly_conservative`

Frozen files:
- `best_model.weights.h5`
- `config.json`
- `training_log.csv`
- `test_eval.log`
- `selector_weights.conservative.txt`

Checkpoint selection:
- best validation checkpoint from the source run

Training recipe highlights:
- backbone init: `imagenet`
- dataset: `dataset/DocCornerDataset`
- `alpha=0.35`
- `fpn_ch=48`
- `simcc_ch=128`
- `img_size=224`
- `num_bins=224`
- `w_simcc=1.0`
- `w_coord=0.2`
- `w_heatmap=0.0`
- `w_coord2d=0.25`
- `w_score=1.0`
- `selector_weight_file=docs/selector_weights.conservative.txt`
- `source_balance_power=0.20`
- `source_balance_cap=2.0`
- `source_weight_sampling=false`

Best validation metrics:
- epoch: `19`
- mean IoU: `0.9838700460`
- median IoU: `0.9884478798`
- corner error mean: `1.0233831406 px`
- corner error p95: `2.6291509271 px`
- recall@95: `96.6%`
- classification accuracy: `100.0%`
- classification F1: `100.0%`

Known test evaluation:
- split: `DocCornerDataset/test`
- mean IoU: `0.8679`
- median IoU: `0.9569`
- corner error mean: `8.29 px`
- corner error p95: `44.80 px`
- recall@90: `66.4%`
- recall@95: `52.5%`
- recall@99: `12.5%`
- classification accuracy: `97.1%`
- classification F1: `98.3%`

Comparison vs previous frozen baseline:
- previous baseline path: `models/v2_best_full_current`
- delta mean IoU on test: `+0.0036`
- delta median IoU on test: `+0.0033`
- delta corner error mean on test: `-0.64 px`
- delta recall@95 on test: `+0.9 pt`

SHA256:
- `best_model.weights.h5`: `4272d8cba0bba2d36eb612ef2723105494b9750b89a4d5f29663f1f4f6100689`
