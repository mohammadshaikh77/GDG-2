# Work Log

This file tracks hands-on implementation and debugging steps performed in this repository.

## 2026-04-14

### 1) Baseline transfer and setup
- Brought the project codebase into this repository.
- Added base dependency file (`requirements.txt`) and project usage documentation.
- Added a unified runner (`run.py`) so experiment scripts can be launched with consistent paths.

### 2) Pipeline mismatch fixed (Exp6)
- Found that plotting for Experiment 6 expected `activation_norm` in metric files.
- Updated `measure.py` to compute and save `activation_norm` from residual states.
- Re-ran syntax checks to ensure no parse issues.

### 3) Reproducibility improvements
- Updated all `scripts/run_exp*.sh` scripts to support configurable paths:
  - `DATA_DIR`, `OUT_DIR`, `PLOTS_DIR`
  - `SOURCE_DIR` for experiment 5
- This removed hardcoded local path assumptions.

### 4) Data and smoke verification
- Downloaded WikiText-103 raw dataset for local smoke testing.
- Ran a short smoke training pass and produced:
  - checkpoint
  - log CSV
  - metrics NPZ
  - smoke plots

### 5) Repository artifact strategy
- Initially push failed due to GitHub file size limits with raw dataset files.
- Added ignore strategy for heavy/generated assets.
- Committed lightweight smoke artifacts under `results/smoke` and `plots/smoke` as visible proof of execution.

### 6) Experiment 5 implementation completed
- Implemented position-stratified cosine metrics in `measure.py`.
- Extended model forward path to optionally return attention weights for analysis mode.
- Added attention-entropy extraction per layer and token position in `measure.py`.
- Implemented `plot_exp5(...)` visualizations for:
  - group-wise cosine vs layer,
  - per-position cosine profile,
  - attention entropy vs token position.
- Re-ran smoke measurement and plotting; exp5 now saves 3 plots successfully.

## Next planned work
- Add minimal test coverage for metric keys and expected output shape contracts.
