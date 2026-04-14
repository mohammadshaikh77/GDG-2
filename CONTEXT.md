# CONTEXT.md

## Dataset
WikiText-103, character-level tokenization.
Vocab built from train split only.
Sequences: non-overlapping chunks of 128 tokens.

## Base model
NanoGPT-style Pre-LN transformer.
6 layers, d=256, 4 heads, MLP inner=1024.
Dropout=0.0 always during measurement.

## Extraction rule
Always extract residual stream AFTER MLP addition, 
BEFORE next LayerNorm. Never extract LayerNorm outputs.

## Shared metric functions
All metric functions (erank, srank, delta, cosine_sim, CKA)
should live in src/metrics.py and be imported by all experiments.
Do not reimplement them per experiment.

## Random seeds
Cosine similarity pairs: always seed=42.
CKA subsampling: always seed=42.
Model training seeds: passed as CLI argument --seed.

## Hardware
Single GPU. Batch operations where possible.
No multi-GPU code.