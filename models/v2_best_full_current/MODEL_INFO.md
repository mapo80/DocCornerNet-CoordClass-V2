# V2 Best Full Current

Source run:
- `runs/v3_full_losssup_continue_20ep`

Frozen files:
- `best_model.weights.h5`
- `config.json`
- `training_log.csv`

Checkpoint selection:
- best validation checkpoint from the source run

Best validation metrics:
- epoch: `17`
- mean IoU: `0.9837061453`
- median IoU: `0.9885415498`
- corner error mean: `1.0420547724 px`
- corner error p95: `2.6295085192 px`
- recall@95: `0.9657704918`

Known test evaluation:
- split: `DocCornerDataset/test`
- mean IoU: `0.8643`
- median IoU: `0.9569`
- corner error mean: `8.53 px`
- corner error p95: `46.98 px`
- recall@95: `52.8%`
- classification accuracy: `97.5%`
- classification F1: `98.6%`

SHA256:
- `best_model.weights.h5`: `85b94bbb9c8c2371737263f3b3687a84907be52e1914880840814c4d37c794a3`
