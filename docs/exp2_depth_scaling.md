# Experiment 2: Depth Scaling of Redundancy Concentration

## Objective

Determine whether early-depth redundancy concentration is a structural phenomenon that persists across model depths, or an artifact of a specific 6-layer setting.

The central question: when depth increases, does the point of maximum redundancy stay at the same **absolute** layer (e.g., always layer 1), or does it stay at the same **relative** depth (e.g., always the first ~15% of layers)?

A finding that redundancy concentrates at fixed relative depth across all model sizes is a robust structural claim. A finding that it always occurs at absolute layer 1 is fragile and architecture-specific.

---

## Models

Train four models varying only depth. **Hold d_model fixed at 256 across all variants.** This is a deliberate choice: you are testing depth-induced dynamics, not iso-capacity scaling. Explicitly acknowledge in any writeup that total parameter count scales with depth under this design.

| Variant | n_layer | n_embd | n_head | head_dim | MLP inner | Approx params |
|---|---|---|---|---|---|---|
| D3 | 3 | 256 | 4 | 64 | 1024 | ~5M |
| D6 | 6 | 256 | 4 | 64 | 1024 | ~10M |
| D12 | 12 | 256 | 4 | 64 | 1024 | ~19M |
| D24 | 24 | 256 | 4 | 64 | 1024 | ~38M |

All other architecture settings are identical:
- Pre-LN (LayerNorm before each sublayer)
- Dropout: 0.0
- block_size: 128
- Tokenization: character-level (same vocabulary as D6 baseline)
- Positional embedding: learned

---

## Dataset

- **Dataset:** WikiText-103 (raw text)
- **Tokenization:** Character-level. Use the same character vocabulary built from the training set for all four model variants.
- **Sequence construction:** Concatenate all training text. Chunk into non-overlapping sequences of 128 tokens. Drop the final incomplete chunk.
- **Training data:** WikiText-103 train split
- **Measurement data:** WikiText-103 validation split only

---

## Training

All four variants use identical training settings:

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

Three seeds per variant gives enough to detect whether the argmin location is stable without requiring 5× the compute. Prioritize D6 and D12 — these are the most informative comparison.

**Total training runs:** 4 variants × 3 seeds = 12 runs.

---

## Measurement Protocol

At each checkpoint, for each seed, for each model variant:

1. Load checkpoint. Set model to `eval()`. Confirm dropout is disabled.
2. Sample the **first 512 sequences** from the validation split sequentially (same sequences for all variants and all checkpoints for comparability).
3. Run forward pass. Extract residual stream hidden state at the output of each transformer block (after MLP addition, before next LayerNorm).

The Pre-LN block computes:
```
H_attn  = LayerNorm_1(H^l)
A^l     = Attention(H_attn)
R^l     = H^l + A^l
H_mlp   = LayerNorm_2(R^l)
M^l     = MLP(H_mlp)
H^{l+1} = R^l + M^l          ← extract this
```

Also extract $H^0$ (embedding output, before block 1).

This gives:
- D3: 4 matrices ($H^0$ through $H^3$)
- D6: 7 matrices ($H^0$ through $H^6$)
- D12: 13 matrices ($H^0$ through $H^{12}$)
- D24: 25 matrices ($H^0$ through $H^{24}$)

Each matrix shape: $(512, 128, 256)$.

4. Reshape to $\mathbb{R}^{65536 \times 256}$.
5. Center by subtracting column mean.
6. Compute all metrics.

---

## Metrics

Compute on each $H^l \in \mathbb{R}^{65536 \times 256}$.

Let $\lambda_1 \geq \ldots \geq \lambda_{256}$ be eigenvalues of the empirical covariance $\frac{1}{N} H^{l\top} H^l$.
Let $p_i = \lambda_i / \sum_j \lambda_j$.

### Metric 1: Effective Rank

$$\text{erank}(H^l) = \exp\!\left(-\sum_{i} p_i \log p_i\right)$$

### Metric 2: Stable Rank

$$\text{srank}(H^l) = \frac{\sum_i \lambda_i}{\lambda_1}$$

### Metric 3: Eigenvalue Dominance

$$\delta(H^l) = \frac{\lambda_1}{\sum_i \lambda_i}$$

### Metric 4: Redundancy Index

$$R(H^l) = \delta(H^l) + \bar{c}(H^l)$$

where $\bar{c}$ is the mean pairwise cosine similarity computed from $M = 2000$ random pairs (fixed seed = 42 for all layers and variants).

### Metric 5: Normalized Layer Index

For each layer $l$ in a model of depth $L$ (not counting the embedding):

$$l_{\text{norm}} = \frac{l}{L}$$

$l_{\text{norm}} = 0$ is the first transformer block output; $l_{\text{norm}} = 1$ is the last.
$H^0$ (embedding) is excluded from normalized depth plots.

### Metric 6: Argmin Rank Layer

For each model variant, seed, and checkpoint:

$$l^* = \arg\min_{l \in \{1,\ldots,L\}} \text{erank}(H^l)$$

Record both the absolute layer index and the normalized layer index $l^*_{\text{norm}} = l^*/L$.

### Metric 7: Depth-Normalized Slope of Rank Drop

Measure the steepness of the rank drop in the first third of the network:

$$\text{slope} = \frac{\text{erank}(H^{\lceil L/3 \rceil}) - \text{erank}(H^1)}{L/3}$$

A negative slope means rank is dropping in early layers. Compare magnitude across depths.

---

## What to Save

For each (variant ∈ {D3, D6, D12, D24}) × (seed ∈ {0,1,2}) × (checkpoint ∈ {2k, 50k, 200k}) × (layer), record:

| Field | Shape |
|---|---|
| `erank` | scalar |
| `srank` | scalar |
| `delta` | scalar |
| `mean_cosine` | scalar |
| `redundancy_index` | scalar |
| `l_norm` | scalar |
| `argmin_layer` | scalar (one per variant/seed/ckpt, not per layer) |
| `argmin_l_norm` | scalar (one per variant/seed/ckpt, not per layer) |
| `early_slope` | scalar (one per variant/seed/ckpt, not per layer) |
| `val_loss` | scalar (one per variant/seed/ckpt) |

---

## Plots

### Plot 1: Effective Rank vs Normalized Depth — All Variants (Final Checkpoint)
- X: normalized layer index $l_{\text{norm}} \in [0, 1]$
- Y: effective rank
- One colored line per variant (D3, D6, D12, D24)
- Shaded band = ±1 std across 3 seeds
- **This is the central figure for this experiment.** If the four lines show a dip at the same normalized depth, the phenomenon is structural.

### Plot 2: Effective Rank vs Absolute Layer — All Variants (Final Checkpoint)
- X: absolute layer index $l \in \{1, \ldots, L\}$
- Y: effective rank
- Same four lines as Plot 1
- Compare directly with Plot 1 to distinguish absolute vs relative localization

### Plot 3: Redundancy Index vs Normalized Depth — All Variants
- Same structure as Plot 1
- Y: Redundancy index $R$

### Plot 4: Argmin Layer — Absolute and Normalized
- Two side-by-side bar charts
- Left: absolute argmin layer ($l^*$) per variant, mean ± std across 3 seeds
- Right: normalized argmin layer ($l^*_{\text{norm}}$) per variant, mean ± std across 3 seeds
- If left bars grow with depth but right bars are approximately constant, the phenomenon is at fixed relative depth

### Plot 5: Depth-Normalized Early Slope vs Depth
- X: model depth (3, 6, 12, 24)
- Y: early slope value
- Points = mean across 3 seeds, error bars = ±1 std
- Shows whether steeper drop correlates with depth

### Plot 6: Eigenvalue Dominance vs Normalized Depth — All Variants
- Same structure as Plot 1
- Y: $\delta$

### Plot 7: Effective Rank Over Training Time (D6 and D12 only)
- X: normalized depth
- One subplot per checkpoint (2k, 50k, 200k)
- Overlay D6 and D12
- Shows whether temporal evolution of the profile is consistent across depths

---

## Hypotheses

State these before plotting. Record which hypothesis the data supports.

- **H1 (relative depth):** $l^*_{\text{norm}}$ is approximately constant across D3, D6, D12, D24. The collapse occurs at fixed fraction of total depth regardless of total layers.
- **H0 (absolute depth):** $l^*$ is approximately equal across variants (e.g., always layer 1 or 2). The collapse is fixed to the first few absolute layers.
- **H2 (no stable pattern):** $l^*_{\text{norm}}$ varies substantially across variants with no consistent trend. The phenomenon is not structural.

Only H1 supports the claim that early-depth redundancy concentration is a robust structural phenomenon.

---

## Questions to Answer

1. Is the normalized depth of minimum effective rank approximately stable across D3, D6, D12, D24?
2. Does the magnitude of rank collapse (how low erank drops) increase or decrease with depth?
3. Is the early slope steeper for deeper models?
4. Are the results consistent across the 3 seeds for each variant?
5. Does the argmin location shift between the 2k and 200k checkpoints?

---

## Failure Conditions

| Condition | Interpretation |
|---|---|
| $l^*_{\text{norm}}$ std > 0.15 across variants | No consistent relative localization |
| $l^*$ is identical (e.g., always 1) for all depths | Collapse is at absolute layer 1 — fragile, depth-specific claim |
| D3 shows no collapse (erank monotonically increasing) | Phenomenon requires minimum depth to appear |
| Std across 3 seeds > 20% of mean at $l^*$ | Not robust enough to claim |

---

## Compute Estimate (Single GPU)

| Task | Time |
|---|---|
| Training D3, 1 seed | ~3 hours |
| Training D6, 1 seed | ~5 hours |
| Training D12, 1 seed | ~9 hours |
| Training D24, 1 seed | ~17 hours |
| 3 seeds each (total 12 runs) | ~100 hours total |
| Measurement (all variants, all seeds, all checkpoints) | ~3 hours |
| **Total** | **~103 hours** |

**Recommendation:** Run D6 and D12 first (6 runs, ~42 hours). If Plot 1 is already informative, add D3 and D24 for completeness. D24 is the most expensive and least critical if D12 already shows the pattern.
