# Changes Log

Date: 14 April 2026
Repository: `Vidit-01/GDG`
Branch: `main`

## 1) Repository Setup and Initial Work

- Cloned the full repository into workspace path:
  - `/Applications/gdg 2/GDG`
- Scanned project structure and identified core modules:
  - `train.py`, `measure.py`, `src/model.py`, `src/data.py`, `src/metrics.py`, `src/plot.py`, `scripts/run_exp*.sh`, `docs/exp*.md`

## 2) Functional Code Improvements

### 2.1 `measure.py` updates (Experiment 6 support)

Implemented activation norm extraction and saving in measurement outputs:

- Added helper function:
  - `flatten_without_center(hidden_state)`
- Added helper function:
  - `compute_activation_norms(residual_states)`
- Added residual-state raw extraction in `measure(...)`:
  - `residual_states_raw = [hidden_states[index] for index in residual_indices]`
- Added result key in `.npz` output:
  - `results["activation_norm"] = compute_activation_norms(residual_states_raw)`

Why:
- `src/plot.py` for `exp6` expects `activation_norm`.
- This fixes a pipeline mismatch and unblocks Experiment 6 plotting requirements.

## 3) New Project Files Added

### 3.1 `README.md` (new)

Added a repository-level README with:
- Project layout
- Prerequisites
- Dependency installation
- Quick start flow
- Manual train/measure/plot commands
- Unified runner usage (`run.py`)
- Notes on context/spec files

### 3.2 `requirements.txt` (new)

Added Python dependency manifest:
- `numpy>=1.24`
- `matplotlib>=3.7`
- `torch>=2.1`

### 3.3 `run.py` (new)

Added master runner for experiment dispatch:
- Supports `--exp 1..6`
- Supports custom directories:
  - `--data_dir`
  - `--results_root`
  - `--plots_root`
  - `--source_root` (for Exp5)
- Injects environment variables and runs matching `scripts/run_exp*.sh`

## 4) Script Improvements (Configurability)

Updated all experiment scripts to remove hardcoded paths and allow overrides through environment variables.

Updated files:
- `scripts/run_exp1.sh`
- `scripts/run_exp2.sh`
- `scripts/run_exp3.sh`
- `scripts/run_exp4.sh`
- `scripts/run_exp5.sh`
- `scripts/run_exp6.sh`

Applied pattern:
- `DATA_DIR="${DATA_DIR:-...}"`
- `OUT_DIR="${OUT_DIR:-...}"`
- `PLOTS_DIR="${PLOTS_DIR:-...}"`
- `SOURCE_DIR="${SOURCE_DIR:-...}"` (exp5 only)

## 5) Environment and Dependency Setup

- Configured Python environment for workspace.
- Installed dependencies for active interpreter (`/usr/bin/python3`) using `requirements.txt`.
- Installed `datasets` package for dataset fallback/download support.

## 6) Data and Pipeline Execution (Smoke Validation)

### 6.1 Dataset

- Downloaded WikiText raw data into:
  - `./data/wikitext-103-raw`

### 6.2 Smoke training and outputs

Ran short and extended smoke training runs.

Observed/generated:
- Vocabulary build output (e.g., `Vocab size: 4980`)
- Log file generated:
  - `./results/smoke/exp1_smoke_seed0_log.csv`
- Vocab file generated:
  - `./results/smoke/exp1_smoke_vocab.json`
- Checkpoint generated at step 500:
  - `./results/smoke/exp1_smoke_seed0_step500.pt`

### 6.3 Smoke measurement

Ran `measure.py` on smoke checkpoint.

Generated:
- `./results/smoke/exp1_seed0_step500_metrics.npz`

### 6.4 Smoke plotting

Ran plotting for exp1 smoke results.

Output:
- `Saved 5 plot(s) to ./plots/smoke`

## 7) Quality Checks Performed

- Python syntax compile check (`py_compile`) on core files: **PASS**
- Shell script syntax check (`bash -n`) on all run scripts: **PASS**
- Workspace error scan: **No errors found**

## 8) Known Remaining Work

- `exp5` full implementation still pending in codebase:
  - `src/plot.py` currently raises a `ValueError` intentionally for `plot_exp5(...)`
  - Attention entropy extraction path is not fully implemented in model/measurement flow
- Additional tests can be added for metric keys and experiment-specific outputs

## 9) Summary

Completed work so far includes:
- setup hardening,
- runner and docs creation,
- script configurability improvements,
- measurement metric fix (`activation_norm`),
- and successful end-to-end smoke validation (train → measure → plot).
