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

## Next planned work
- Implement full Experiment 5 measurement path (position-group cosine + attention entropy extraction support).
- Add minimal test coverage for metric keys and expected output shape contracts.
