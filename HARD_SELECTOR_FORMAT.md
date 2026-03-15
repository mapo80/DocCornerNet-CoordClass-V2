# Hard Selector Format

`train_ultra.py` supports an external selector file through:

```bash
--hard_selector_file /path/to/hard_selectors.txt
```

The file is plain text. Blank lines and lines starting with `#` are ignored.

Supported line formats:

```text
# Bare value = source selector
idcard_jj

# Explicit source selector
source:receipt_occam

# Exact sample filename selector
sample:receipt_occam_000123.jpg
```

Rules:

- Bare values are treated as `source:<value>`.
- `source:<value>` matches the extracted source family derived from filenames like
  `source_name_000123.jpg -> source_name`.
- `sample:<value>` matches the exact filename basename.
- Selectors are OR-ed together.
- Only positive samples (`has_doc=1`) are eligible for hard mixing / hard reporting.

Typical usage:

```bash
python -m train_ultra \
  ... \
  --hard_selector_file configs/hard_selectors.txt \
  --hard_selector_mix_weight 0.10 \
  --report_val_hard
```

Selector-only geometry augmentation:

```bash
python -m train_ultra \
  ... \
  --augment \
  --rotation_range 2.0 \
  --scale_range 0.05 \
  --hard_selector_file configs/hard_selectors.txt \
  --augment_selector_only
```

With `--augment_selector_only`, brightness/contrast/saturation still apply to the whole batch,
while geometric transforms are restricted to samples matched by the selector file.

Recommended workflow:

1. Start with source-level selectors only.
2. Use a conservative `--hard_selector_mix_weight` like `0.10` or `0.15`.
3. Track both global validation metrics and `ValHard`.
4. Only add sample-level selectors for persistent corner cases you have verified manually.
