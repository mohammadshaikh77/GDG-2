#!/bin/bash
set -e
DATA_DIR="${DATA_DIR:-./data/wikitext103}"
OUT_DIR="${OUT_DIR:-./results/exp3}"
PLOTS_DIR="${PLOTS_DIR:-./plots/exp3}"
mkdir -p $OUT_DIR

for N_EMBD in 128 256 512; do
  if [ "$N_EMBD" -eq 128 ]; then
    N_HEAD=2
  elif [ "$N_EMBD" -eq 256 ]; then
    N_HEAD=4
  else
    N_HEAD=8
  fi

  EXP_NAME="exp3_W${N_EMBD}"

  for SEED in 0 1 2; do
    python train.py \
      --n_layer 6 --n_embd $N_EMBD --n_head $N_HEAD \
      --block_size 128 --dropout 0.0 \
      --seed $SEED \
      --data_dir $DATA_DIR \
      --out_dir $OUT_DIR \
      --exp_name $EXP_NAME
  done

  for SEED in 0 1 2; do
    for STEP in 500 1000 2000 10000 50000 200000; do
      CKPT="$OUT_DIR/${EXP_NAME}_seed${SEED}_step${STEP}.pt"
      if [ -f "$CKPT" ]; then
        python measure.py \
          --checkpoint $CKPT \
          --data_dir $DATA_DIR \
          --out_dir $OUT_DIR \
          --exp_name $EXP_NAME \
          --seed $SEED
      fi
    done
  done
done

# Plot results
mkdir -p $PLOTS_DIR
python src/plot.py \
  --exp_name exp3 \
  --results_dir $OUT_DIR \
  --out_dir $PLOTS_DIR \
  --format pdf

echo "Experiment 3 complete."
