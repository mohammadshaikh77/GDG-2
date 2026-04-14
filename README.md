# GDG — Transformer Redundancy Experiments

This repository contains training, measurement, and plotting code for six transformer redundancy experiments described in `docs/`.

## Project layout

- `train.py` — unified trainer for experiment variants
- `measure.py` — checkpoint metric extraction (`.npz` outputs)
- `src/plot.py` — plot generation from measurement files
- `scripts/run_exp*.sh` — end-to-end runners per experiment
- `docs/exp*.md` — experiment specifications

## Prerequisites

- Python 3.9+
- PyTorch-compatible machine (GPU recommended)
- WikiText-103 raw files in:
  - `<data_dir>/wiki.train.raw`
  - `<data_dir>/wiki.valid.raw`

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Quick start (Experiment 1)

Run full train → measure → plot pipeline:

```bash
bash scripts/run_exp1.sh
```

Or use the unified runner:

```bash
python3 run.py --exp 1
```

By default, scripts expect dataset at:

```text
./data/wikitext103
```

and write outputs to:

- results: `./results/exp1`
- plots: `./plots/exp1`

## Manual workflow

Train a model:

```bash
python3 train.py \
  --n_layer 6 --n_embd 256 --n_head 4 \
  --block_size 128 --dropout 0.0 \
  --seed 0 \
  --data_dir ./data/wikitext103 \
  --out_dir ./results/exp1 \
  --exp_name exp1
```

Measure one checkpoint:

```bash
python3 measure.py \
  --checkpoint ./results/exp1/exp1_seed0_step200000.pt \
  --data_dir ./data/wikitext103 \
  --out_dir ./results/exp1 \
  --exp_name exp1 \
  --seed 0
```

Generate plots:

```bash
python3 src/plot.py \
  --exp_name exp1 \
  --results_dir ./results/exp1 \
  --out_dir ./plots/exp1 \
  --format pdf
```

## Unified runner with custom paths

`run.py` dispatches to `scripts/run_exp*.sh` and injects configurable paths.

```bash
python3 run.py \
  --exp 3 \
  --data_dir ./data/wikitext103 \
  --results_root ./results \
  --plots_root ./plots
```

For experiment 5, you can also override the location of exp1 checkpoints:

```bash
python3 run.py --exp 5 --source_root ./results
```

## Notes

- Random-seed conventions and extraction rules are centralized in `CONTEXT.md`.
- Shared metric implementations are in `src/metrics.py`.
- For experiment-specific requirements and expected outputs, follow the matching `docs/expN_*.md` file.

## Smoke validation evidence

This repository includes a lightweight smoke run under `results/smoke/` and `plots/smoke/` to provide visible proof that the pipeline executes end-to-end.

- Training produced a checkpoint and CSV log.
- Measurement produced NPZ metrics.
- Plotting generated PDF outputs.

See:
- `results/smoke/README.md`
- `results/smoke/ANALYSIS.md`
- `WORKLOG.md`
