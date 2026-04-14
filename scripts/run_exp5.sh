#!/bin/bash
set -e
DATA_DIR="${DATA_DIR:-./data/wikitext103}"
SOURCE_DIR="${SOURCE_DIR:-./results/exp1}"
OUT_DIR="${OUT_DIR:-./results/exp5}"
PLOTS_DIR="${PLOTS_DIR:-./plots/exp5}"
mkdir -p $OUT_DIR

for SEED in 0 1 2; do
  CKPT="$SOURCE_DIR/exp1_seed${SEED}_step200000.pt"
  if [ -f "$CKPT" ]; then
    python measure.py \
      --checkpoint $CKPT \
      --data_dir $DATA_DIR \
      --out_dir $OUT_DIR \
      --exp_name exp5 \
      --seed $SEED
  fi
done

# Plot results
mkdir -p $PLOTS_DIR
python src/plot.py \
  --exp_name exp5 \
  --results_dir $OUT_DIR \
  --out_dir $PLOTS_DIR \
  --format pdf

echo "Experiment 5 complete."
