# Selector Weight Format

`train_ultra.py` supports external loss/sampling weights through:

```bash
--selector_weight_file /path/to/selector_weights.txt
```

The file is plain text. Blank lines and lines starting with `#` are ignored.

Supported line formats:

```text
# Bare value with assignment = source weight
idcard_jj=2.0

# Explicit source selector
source:receipt_occam=2.5

# Exact sample filename selector
sample:receipt_occam_000123.jpg=4.0

# Optional negative-class multiplier
negative=0.5
```

Rules:

- Bare values are treated as `source:<value>=<weight>`.
- `source:<value>=<weight>` matches the extracted source family derived from filenames like
  `source_name_000123.jpg -> source_name`.
- `sample:<value>=<weight>` matches the exact filename basename.
- `negative=<weight>` applies to samples with `has_doc=0`.
- Multiple matching rules multiply together.
- Weights are normalized to mean `1.0` on the training split before entering the trainer.

Automatic source balancing:

```bash
python -m train_ultra \
  ... \
  --source_balance_power 0.5 \
  --source_balance_cap 4.0
```

This computes a positive-source multiplier from train frequency:

```text
weight(source) = min((max_count / source_count) ^ power, cap)
```

Typical workflows:

Loss weighting only:

```bash
python -m train_ultra \
  ... \
  --selector_weight_file configs/selector_weights.txt
```

Source-balanced sampling from epoch 1:

```bash
python -m train_ultra \
  ... \
  --selector_weight_file configs/selector_weights.txt \
  --source_balance_power 0.5 \
  --source_weight_sampling
```

Recommended workflow:

1. Start with `--source_balance_power 0.3` to `0.5`.
2. Add only a few verified `source:` weights in the file.
3. Enable `--source_weight_sampling` for full retrains, not late-stage continuations.
4. Keep validation metrics unweighted; use weighted training only.
