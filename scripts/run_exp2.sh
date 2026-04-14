#!/bin/bash
set -e
DATA_DIR="${DATA_DIR:-./data/wikitext103}"
OUT_DIR="${OUT_DIR:-./results/exp2}"
PLOTS_DIR="${PLOTS_DIR:-./plots/exp2}"
mkdir -p $OUT_DIR

for N_LAYER in 3 6 12 24; do
  EXP_NAME="exp2_L${N_LAYER}"

  for SEED in 0 1 2; do
    python train.py \
      --n_layer $N_LAYER --n_embd 256 --n_head 4 \
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
  --exp_name exp2 \
  --results_dir $OUT_DIR \
  --out_dir $PLOTS_DIR \
  --format pdf

echo "Experiment 2 complete."
