#!/bin/bash
set -e

DATA_DIR="${DATA_DIR:-./data/wikitext-103-raw}"
OUT_DIR="${OUT_DIR:-./results/smoke}"
PLOTS_DIR="${PLOTS_DIR:-./plots/smoke}"

mkdir -p "$OUT_DIR"
mkdir -p "$PLOTS_DIR"

python3 train.py \
  --n_layer 2 --n_embd 64 --n_head 2 \
  --block_size 64 --dropout 0.0 \
  --batch_size 8 --total_steps 500 --warmup_steps 20 \
  --lr 3e-4 --min_lr 1e-5 \
  --seed 0 \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --exp_name exp1_smoke

python3 measure.py \
  --checkpoint "$OUT_DIR/exp1_smoke_seed0_step500.pt" \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --exp_name exp1 \
  --seed 0 \
  --n_sequences 32 \
  --block_size 64

python3 src/plot.py \
  --exp_name exp1 \
  --results_dir "$OUT_DIR" \
  --out_dir "$PLOTS_DIR" \
  --format pdf

echo "Smoke run complete: $OUT_DIR and $PLOTS_DIR"
