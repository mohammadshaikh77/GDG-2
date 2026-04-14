# Experiment 1: Baseline Redundancy Profile Across Depth

## Objective

Establish the baseline shape of representational redundancy across all 6 layers of a NanoGPT model.
Determine whether redundancy is monotonic, front-loaded, or irregular across depth.
This experiment makes no causal claims — it only characterizes the phenomenon.

---

## Model

| Parameter | Value |
|---|---|
| Architecture | GPT-style decoder-only transformer, Pre-LN |
| n_layer | 6 |
| n_embd (d_model) | 256 |
| n_head | 4 |
| head_dim | 64 |
| block_size (sequence length) | 128 |
| MLP inner dim | 1024 (4 × 256) |
| Dropout | 0.0 — must be disabled for clean geometry |
| Tokenization | Character-level |
| Vocab size | ~100 (built from WikiText-103 training set only) |

---

## Dataset

- **Dataset:** WikiText-103 (raw text, not the pre-tokenized version)
- **Tokenization:** Character-level. Build vocabulary from unique characters in the training set only. Map each character to an integer id.
- **Sequence construction:** Concatenate all training text into one long character stream. Chunk into non-overlapping sequences of 128 tokens. Drop the final incomplete chunk.
- **Training data:** WikiText-103 train split
- **Measurement data:** WikiText-103 validation split only. Never used during training.

---

## Training

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| LR schedule | Cosine decay to 1e-5 |
| Warmup steps | 2000 |
| Batch size | 64 sequences |
| Total steps | 200,000 |
| Weight decay | 0.1 |
| Beta1, Beta2 | 0.9, 0.95 |
| Gradient clip | 1.0 |
| Seeds | 5 independent runs (seeds 0, 1, 2, 3, 4) |

**Checkpoints to save:** steps 1000, 2000, 10000, 50000, 200000.

The 1000 and 2000 checkpoints are critical. Early transient geometry often determines the final structure. Do not skip them.

---

## Measurement Protocol

At every checkpoint, for every seed:

1. Load checkpoint. Set model to `eval()`. Confirm all dropout is disabled.
2. Sample **512 sequences** sequentially from the validation split (no randomness — use the first 512 chunks of the validation set for reproducibility across seeds and checkpoints).
3. Run a single forward pass. Extract the residual stream hidden state at the **output of each transformer block**.

The Pre-LN transformer computes each block as:
```
H_attn  = LayerNorm_1(H^l)
A^l     = Attention(H_attn)
R^l     = H^l + A^l                    ← post-attention residual
H_mlp   = LayerNorm_2(R^l)
M^l     = MLP(H_mlp)
H^{l+1} = R^l + M^l                   ← post-block residual stream
```

**Extract $H^{l+1}$ after the MLP addition.** Do NOT extract LayerNorm outputs. LayerNorm standardizes variance per token and will artificially inflate rank estimates.

Also extract $H^0$ (token embedding + positional embedding, before block 1) as the baseline.

This gives 7 matrices total: $H^0, H^1, \ldots, H^6$, each of shape $(512, 128, 256)$.

4. Reshape each to $\mathbb{R}^{65536 \times 256}$ by flattening the batch and sequence dimensions.
5. Center each matrix: subtract the column mean (mean computed over the 65536 token dimension).
6. Compute all metrics on this centered matrix.

---

## Metrics

All metrics are computed on $H^l \in \mathbb{R}^{N \times d}$ where $N = 65536$, $d = 256$.

Let $\Sigma = \frac{1}{N} H^{l\top} H^l$ be the $d \times d$ empirical covariance matrix.
Let $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_d \geq 0$ be its eigenvalues.
Let $p_i = \lambda_i / \sum_j \lambda_j$ be the normalized eigenvalue distribution.

### Metric 1: Roy's Effective Rank

$$\text{erank}(H^l) = \exp\!\left(-\sum_{i=1}^{d} p_i \log p_i\right)$$

Exponential of the Shannon entropy of the normalized spectrum. Ranges from 1 (rank-1) to $d$ (uniform spectrum). High value = spread representation. Low value = collapsed representation.

### Metric 2: Stable Rank

$$\text{srank}(H^l) = \frac{\|H^l\|_F^2}{\|H^l\|_2^2} = \frac{\sum_i \lambda_i}{\lambda_1}$$

More sensitive to the dominant eigenvalue than effective rank. Ranges from 1 to $d$.

### Metric 3: Fractional Effective Rank

$$\text{ferank}(H^l) = \frac{\text{erank}(H^l)}{d}$$

Normalizes effective rank by dimension. Ranges from $1/d$ to 1. Enables comparison across different widths if needed later.

### Metric 4: Eigenvalue Dominance

$$\delta(H^l) = p_1 = \frac{\lambda_1}{\sum_i \lambda_i}$$

Fraction of total variance captured by the single largest direction. High $\delta$ means collapse to a single dominant direction. This distinguishes single-direction collapse from collapse to a lower-dimensional subspace. Effective rank cannot distinguish these two; $\delta$ can.

### Metric 5: Mean Pairwise Cosine Similarity

Sample $M = 2000$ pairs $(i, j)$ with $i \neq j$ uniformly at random from the 65536 token vectors. Use a fixed random seed (seed = 42) for this sampling, identical across all layers, checkpoints, and model seeds.

$$\bar{c}(H^l) = \frac{1}{M} \sum_{(i,j)} \frac{h_i \cdot h_j}{\|h_i\| \|h_j\|}$$

### Metric 6: Variance of Pairwise Cosine Similarity

Using the same $M = 2000$ pairs:

$$\text{Var}_c(H^l) = \frac{1}{M} \sum_{(i,j)} \left(\frac{h_i \cdot h_j}{\|h_i\| \|h_j\|} - \bar{c}(H^l)\right)^2$$

Mean cosine similarity alone hides bimodality. If some token pairs are very similar and others are orthogonal, the mean can be moderate while the variance is high. Always report both.

### Metric 7: CKA Between All Layer Pairs

Linear Centered Kernel Alignment between $H^l$ and $H^{l'}$:

$$\text{CKA}(H^l, H^{l'}) = \frac{\|H^{l'\top} H^l\|_F^2}{\|H^{l\top} H^l\|_F \cdot \|H^{l'\top} H^{l'}\|_F}$$

Due to memory: subsample to **4096 tokens** before computing CKA. Use a fixed random seed (seed = 42) for this subsampling, identical across all layer pairs, checkpoints, and model seeds.

Compute for all $\binom{7}{2} + 7 = 28$ pairs (including self-similarity on diagonal, which should equal 1 and serves as a sanity check). This produces a $7 \times 7$ symmetric matrix per checkpoint per seed.

### Metric 8: Redundancy Index

$$R(H^l) = \delta(H^l) + \bar{c}(H^l)$$

Scalar combining eigenvalue dominance with angular collapse. Ranges from 0 to 2. Use this as the headline single-number summary across all experiments.

---

## What to Save

For each combination of (seed ∈ {0,1,2,3,4}) × (checkpoint ∈ {1k, 2k, 10k, 50k, 200k}) × (layer ∈ {0,1,2,3,4,5,6}), record:

| Field | Shape |
|---|---|
| `erank` | scalar |
| `srank` | scalar |
| `ferank` | scalar |
| `delta` | scalar |
| `mean_cosine` | scalar |
| `var_cosine` | scalar |
| `redundancy_index` | scalar |
| `eigenvalues` | vector, length 256 |

For each (seed) × (checkpoint), record:

| Field | Shape |
|---|---|
| `cka_matrix` | 7 × 7 matrix |
| `val_loss` | scalar |

Save everything to a single `.npz` file per seed. Do not discard raw eigenvalues — you will need them for spectral plots.

---

## Plots

All plots: mean across 5 seeds with ±1 std shaded band, unless stated otherwise.

### Plot 1: Effective Rank vs Layer — All Checkpoints
- X: layer index (0–6)
- Y: effective rank
- One colored line per checkpoint (5 lines)
- Shaded band = ±1 std across seeds
- Marks on x-axis at each integer layer

### Plot 2: Stable Rank vs Layer — Final Checkpoint
- X: layer index (0–6)
- Y: left axis = effective rank, right axis = stable rank (dual y-axis)
- Step 200k only
- Shows whether the two metrics agree or diverge

### Plot 3: Eigenvalue Dominance vs Layer — All Checkpoints
- X: layer index (0–6)
- Y: $\delta = \lambda_1 / \sum \lambda_i$
- One line per checkpoint
- Shaded band = ±1 std across seeds

### Plot 4: Eigenvalue Spectra — Selected Layers
- 4 subplots: layers {0, 1, 3, 6}
- X: eigenvalue index (1 to 256)
- Y: eigenvalue magnitude (log scale)
- Plot all 5 seeds as individual thin lines (no averaging — show raw spread)
- Checkpoint: 200k only
- This reveals whether the spectrum has a sharp cliff or gradual decay

### Plot 5: Mean Cosine Similarity vs Layer — All Checkpoints
- X: layer index (0–6)
- Y: $\bar{c}$
- One line per checkpoint, shaded band = ±1 std across seeds

### Plot 6: Variance of Cosine Similarity vs Layer — All Checkpoints
- Same structure as Plot 5
- Y: $\text{Var}_c$

### Plot 7: CKA Heatmap — Final Checkpoint
- 7×7 matrix, layers 0–6 on both axes
- Values: mean across 5 seeds
- Color: viridis, range fixed [0, 1]
- Annotate each cell with its numeric value (2 decimal places)

### Plot 8: CKA Heatmap — Temporal Evolution
- 5 subplots in a row: checkpoints 1k, 2k, 10k, 50k, 200k
- Same format as Plot 7
- Fixed color scale across all 5 subplots so comparison is valid

### Plot 9: Redundancy Index vs Layer — All Checkpoints
- X: layer index (0–6)
- Y: $R = \delta + \bar{c}$
- One line per checkpoint, shaded band = ±1 std across seeds
- **This is the headline figure.**

---

## Questions to Answer

After producing all plots, explicitly answer the following in writing:

1. **Shape of redundancy profile:** Is it monotonic, front-loaded (early dip then recovery), oscillatory, or irregular?
2. **Location of minimum effective rank:** At which layer index? Is this consistent across seeds (std < 0.5 layers)?
3. **Type of collapse:** Is $\delta$ high at the same layers where erank is low? If yes, collapse is to a single dominant direction. If $\delta$ is low but erank is also low, collapse is to a subspace.
4. **Angular collapse agreement:** Does $\bar{c}$ peak at the same layer as the erank minimum? If they disagree, report both locations.
5. **Bimodality:** Is $\text{Var}_c$ elevated at any layer? If yes, representations are splitting into clusters.
6. **CKA stabilization step:** Compare the 2k and 10k heatmaps. If they look nearly identical, the representational structure is set by step 2k — well before convergence.
7. **Redundancy index location:** At which layer is $R$ maximized at the final checkpoint?

---

## Failure Conditions

| Condition | What it means |
|---|---|
| Std across seeds > 20% of mean at the minimum layer | Not robust; do not make a claim about that layer |
| Profile shape changes qualitatively between 50k and 200k | Phenomenon is phase-dependent; describe phases instead |
| Effective rank and cosine similarity disagree on which layer is most redundant | Report both without collapsing to a single story |
| CKA heatmap nearly uniform (all off-diagonal values > 0.9) at step 200k | Representations are homogeneous across depth; architecture may be too shallow to show the effect |
| erank minimum is at the last layer | Redundancy is late, not early; your central claim changes |

---

## Compute Estimate (Single GPU)

| Task | Time |
|---|---|
| Training one seed to 200k steps | ~5–7 hours (Colab A100) |
| 5 seeds total | ~25–35 hours (parallelize across sessions) |
| Measurement at all checkpoints, one seed | ~20 minutes |
| Total measurement, 5 seeds | ~2 hours |
| **Total** | **~30–37 hours** |
