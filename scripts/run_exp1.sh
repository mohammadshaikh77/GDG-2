#!/bin/bash
set -e
DATA_DIR="${DATA_DIR:-./data/wikitext103}"
OUT_DIR="${OUT_DIR:-./results/exp1}"
PLOTS_DIR="${PLOTS_DIR:-./plots/exp1}"
mkdir -p $OUT_DIR

# Train all 5 seeds
for SEED in 0 1 2 3 4; do
  python train.py \
    --n_layer 6 --n_embd 256 --n_head 4 \
    --block_size 128 --dropout 0.0 \
    --seed $SEED \
    --data_dir $DATA_DIR \
    --out_dir $OUT_DIR \
    --exp_name exp1
done

# Measure all checkpoints for all seeds
for SEED in 0 1 2 3 4; do
  for STEP in 500 1000 2000 10000 50000 200000; do
    CKPT="$OUT_DIR/exp1_seed${SEED}_step${STEP}.pt"
    if [ -f "$CKPT" ]; then
      python measure.py \
        --checkpoint $CKPT \
        --data_dir $DATA_DIR \
        --out_dir $OUT_DIR \
        --exp_name exp1 \
        --seed $SEED
    fi
  done
done

# Plot results
mkdir -p $PLOTS_DIR
python src/plot.py \
  --exp_name exp1 \
  --results_dir $OUT_DIR \
  --out_dir $PLOTS_DIR \
  --format pdf

echo "Experiment 1 complete."
