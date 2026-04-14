# Experiment 3: Width Scaling of Redundancy Concentration

## Objective

Test whether redundancy concentration is driven by **dimensional capacity** (the hidden dimension $d$), or whether it is a structural property of the transformer computation independent of width.

If fractional effective rank ($\text{erank}/d$) is approximately invariant across widths, the collapse is structural — the same fraction of the available dimensions collapses regardless of how many dimensions exist. If fractional rank increases with width, wider models preserve more of their capacity, suggesting the collapse is a capacity bottleneck.

Additionally, test whether collapse is to a **single dominant direction** (high eigenvalue dominance $\delta$) or to a **lower-dimensional subspace** (low erank but moderate $\delta$). This distinction matters for any intervention design.

---

## Models

Train three models varying only hidden dimension. **Hold depth fixed at 6 layers.**

| Variant | n_layer | n_embd | n_head | head_dim | MLP inner | Approx params |
|---|---|---|---|---|---|---|
| W128 | 6 | 128 | 2 | 64 | 512 | ~3M |
| W256 | 6 | 256 | 4 | 64 | 1024 | ~10M |
| W512 | 6 | 512 | 8 | 64 | 2048 | ~38M |

All other settings identical:
- Pre-LN
- Dropout: 0.0
- block_size: 128
- Tokenization: character-level (same vocabulary across all variants)
- Positional embedding: learned

Note: head_dim is held at 64 across all variants. This changes the number of heads with width, which is the standard scaling practice.

---

## Dataset

- **Dataset:** WikiText-103 (raw text)
- **Tokenization:** Character-level. Use the same character vocabulary built from the WikiText-103 training set for all variants.
- **Sequence construction:** Concatenate all training text into one long stream. Chunk into non-overlapping sequences of 128 tokens. Drop the final incomplete chunk.
- **Training data:** WikiText-103 train split
- **Measurement data:** WikiText-103 validation split only

---

## Training

All three variants use identical training settings:

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
| Seeds | 3 independent runs per variant (seeds 0, 1, 2) |

**Checkpoints to save:** steps 2000, 50000, 200000.

**Total training runs:** 3 variants × 3 seeds = 9 runs.

---

## Measurement Protocol

At each checkpoint, for each seed, for each model variant:

1. Load checkpoint. Set model to `eval()`. Confirm dropout is disabled.
2. Sample the **first 512 sequences** from the validation split sequentially (same 512 sequences for all variants for comparability).
3. Run forward pass. Extract residual stream hidden state at the output of each transformer block (after MLP addition).

The Pre-LN block computes:
```
H_attn  = LayerNorm_1(H^l)
A^l     = Attention(H_attn)
R^l     = H^l + A^l
H_mlp   = LayerNorm_2(R^l)
M^l     = MLP(H_mlp)
H^{l+1} = R^l + M^l          ← extract this
```

**Do not extract LayerNorm outputs.** Extract only the raw residual stream after the MLP addition.

Also extract $H^0$ (embedding output before block 1).

This gives 7 matrices per model variant, each of shape:
- W128: $(512, 128, 128)$
- W256: $(512, 128, 256)$
- W512: $(512, 128, 512)$

4. Reshape to $\mathbb{R}^{65536 \times d}$ for the respective $d$.
5. Center by subtracting column mean.
6. Compute all metrics.

---

## Metrics

Let $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_d \geq 0$ be eigenvalues of $\frac{1}{N} H^{l\top} H^l$.
Let $p_i = \lambda_i / \sum_j \lambda_j$.

### Metric 1: Effective Rank

$$\text{erank}(H^l) = \exp\!\left(-\sum_{i} p_i \log p_i\right)$$

### Metric 2: Fractional Effective Rank

$$\text{ferank}(H^l) = \frac{\text{erank}(H^l)}{d}$$

**Primary metric for this experiment.** If this is approximately equal across W128, W256, W512 at each layer, the collapse is structural.

### Metric 3: Stable Rank

$$\text{srank}(H^l) = \frac{\sum_i \lambda_i}{\lambda_1}$$

### Metric 4: Fractional Stable Rank

$$\text{fsrank}(H^l) = \frac{\text{srank}(H^l)}{d} = \frac{\sum_i \lambda_i}{d \cdot \lambda_1}$$

If both fractional effective rank and fractional stable rank are invariant to width, the structural interpretation is very strong.

### Metric 5: Eigenvalue Dominance

$$\delta(H^l) = \frac{\lambda_1}{\sum_i \lambda_i}$$

Tests whether collapse is to a single dominant direction. $\delta$ is independent of $d$ (it is already a fraction), so it can be compared across widths directly without normalization.

### Metric 6: Top-k Eigenvalue Dominance

$$\delta_k(H^l) = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_i \lambda_i}$$

Compute for $k \in \{1, 5, 10\}$. This gives a fuller picture of how concentrated the spectrum is. A matrix with $\delta_1 = 0.5$ and $\delta_5 = 0.9$ has a 5-dimensional subspace capturing 90% of variance — fundamentally different from one where $\delta_1 = 0.5$ and $\delta_5 = 0.6$.

### Metric 7: Mean Pairwise Cosine Similarity

Sample $M = 2000$ random pairs from the 65536 token vectors (fixed seed = 42 for all layers, variants, and checkpoints):

$$\bar{c}(H^l) = \frac{1}{M}\sum_{(i,j)} \frac{h_i \cdot h_j}{\|h_i\|\|h_j\|}$$

### Metric 8: Redundancy Index

$$R(H^l) = \delta(H^l) + \bar{c}(H^l)$$

This is width-independent (both components are already normalized), so it can be compared across variants directly.

---

## What to Save

For each (variant ∈ {W128, W256, W512}) × (seed ∈ {0,1,2}) × (checkpoint ∈ {2k, 50k, 200k}) × (layer ∈ {0,…,6}):

| Field | Shape |
|---|---|
| `erank` | scalar |
| `ferank` | scalar |
| `srank` | scalar |
| `fsrank` | scalar |
| `delta` | scalar |
| `delta_5` | scalar |
| `delta_10` | scalar |
| `mean_cosine` | scalar |
| `redundancy_index` | scalar |
| `eigenvalues` | vector of length $d$ (128, 256, or 512) |
| `val_loss` | scalar (one per variant/seed/ckpt) |

---

## Plots

### Plot 1: Fractional Effective Rank vs Layer — All Widths (Final Checkpoint)
- X: layer index (0–6)
- Y: $\text{ferank} = \text{erank}/d$
- One colored line per variant (W128, W256, W512)
- Shaded band = ±1 std across 3 seeds
- **Primary figure.** If lines overlap, collapse is structural. If W512 is consistently higher, wider models preserve more capacity.

### Plot 2: Effective Rank vs Layer — All Widths (Final Checkpoint)
- Same structure as Plot 1
- Y: raw effective rank (not normalized by $d$)
- Compare with Plot 1 to show what normalization does

### Plot 3: Fractional Stable Rank vs Layer — All Widths
- Same structure as Plot 1
- Y: $\text{fsrank} = \text{srank}/d$
- If both ferank (Plot 1) and fsrank (Plot 3) are invariant, the structural claim is robust

### Plot 4: Eigenvalue Dominance vs Layer — All Widths
- Same structure as Plot 1
- Y: $\delta = \lambda_1 / \sum \lambda_i$
- $\delta$ is already normalized, so lines are directly comparable

### Plot 5: Top-k Dominance at Layer 1 — All Widths
- X: $k \in \{1, 5, 10, 20, 50\}$
- Y: $\delta_k$ (cumulative eigenvalue fraction)
- One line per variant
- Final checkpoint only, layer 1 (or whichever layer shows maximum redundancy)
- This is the spectral concentration curve — shows the subspace dimensionality of the collapse

### Plot 6: Eigenvalue Spectra — Layer 1 vs Layer 6
- Two subplots: left = layer 1, right = layer 6
- X: eigenvalue index (normalized to [0, 1] to allow comparison across widths)
- Y: $p_i = \lambda_i / \sum \lambda_i$ (log scale)
- One line per variant, all 3 seeds as thin individual lines

### Plot 7: Redundancy Index vs Layer — All Widths
- X: layer index (0–6)
- Y: $R = \delta + \bar{c}$
- One line per variant, shaded band = ±1 std across seeds
- Final checkpoint

### Plot 8: Mean Cosine Similarity vs Layer — All Widths
- Same structure as Plot 7
- Y: $\bar{c}$

---

## Hypotheses

State these before plotting. Record which the data supports.

- **H1 (structural):** Fractional effective rank is approximately invariant across W128, W256, W512 at each layer. Width does not change the fraction of capacity that collapses.
- **H2 (capacity-driven):** Fractional effective rank increases with width. Wider models preserve proportionally more of their representational capacity.
- **H3 (single direction):** $\delta$ is high (> 0.3) at the layer of maximum redundancy across all widths. Collapse is to a single dominant direction.
- **H4 (subspace collapse):** $\delta$ is moderate (< 0.2) but erank is low. Collapse is to a multi-dimensional subspace, not a single direction.

---

## Questions to Answer

1. Is fractional effective rank approximately invariant across widths at each layer?
2. Is eigenvalue dominance ($\delta$) consistent across widths? Does it implicate a single-direction collapse?
3. What does $\delta_5$ and $\delta_{10}$ reveal about the dimensionality of the collapsed subspace?
4. Do fractional effective rank and fractional stable rank agree on which layers are most collapsed?
5. Does the redundancy index $R$ show the same layer-wise pattern across all three widths?

---

## Failure Conditions

| Condition | Interpretation |
|---|---|
| Fractional effective rank varies by > 0.1 across widths at the collapsed layer | Collapse is capacity-dependent, not purely structural |
| $\delta > 0.5$ at collapsed layer | Collapse is effectively rank-1 — nearly all variance in one direction |
| $\delta < 0.05$ but erank is still low | Collapse is to a diffuse low-dimensional subspace |
| W128 shows no collapse (ferank flat) | Width 128 is too narrow for the phenomenon to appear |
| Std across 3 seeds > 20% of mean | Not robust enough to claim |

---

## Compute Estimate (Single GPU)

| Task | Time |
|---|---|
| Training W128, 1 seed | ~3 hours |
| Training W256, 1 seed | ~5 hours |
| Training W512, 1 seed | ~10 hours |
| 3 seeds each (9 runs total) | ~54 hours |
| Measurement (all variants, seeds, checkpoints) | ~2 hours |
| **Total** | **~56 hours** |
