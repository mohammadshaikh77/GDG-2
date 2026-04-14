#!/bin/bash
set -e
DATA_DIR="${DATA_DIR:-./data/wikitext103}"
OUT_DIR="${OUT_DIR:-./results/exp6}"
PLOTS_DIR="${PLOTS_DIR:-./plots/exp6}"
mkdir -p $OUT_DIR

for RESIDUAL_ALPHA in 0.5 1.0 1.5 2.0; do
  EXP_TAG=$(printf "%s" "$RESIDUAL_ALPHA" | tr '.' '_')
  EXP_NAME="exp6_alpha${EXP_TAG}"

  for SEED in 0 1 2; do
    python train.py \
      --n_layer 6 --n_embd 256 --n_head 4 \
      --block_size 128 --dropout 0.0 \
      --residual_alpha $RESIDUAL_ALPHA \
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
  --exp_name exp6 \
  --results_dir $OUT_DIR \
  --out_dir $PLOTS_DIR \
  --format pdf

echo "Experiment 6 complete."
