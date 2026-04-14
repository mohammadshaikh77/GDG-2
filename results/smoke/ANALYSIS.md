# Smoke Run Analysis

This is a lightweight validation run intended to prove end-to-end execution of:

`train -> checkpoint -> measure -> plot`

## Artifacts produced

- `exp1_smoke_seed0_log.csv`
- `exp1_smoke_seed0_step500.pt`
- `exp1_seed0_step500_metrics.npz`
- `exp1_smoke_vocab.json`
- Plot PDFs in `plots/smoke/`

## What this confirms

1. Data loading and vocabulary construction are functioning.
2. Model training loop executes and writes logs/checkpoints.
3. Measurement pipeline can load checkpoint and export NPZ metrics.
4. Plotting pipeline can read metrics and generate visual outputs.

## Scope note

This smoke run is intentionally small and does **not** represent final research-quality experiment outcomes.
It is a technical validation checkpoint.
