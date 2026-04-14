# Experiment 6: Residual Scaling Ablation

## Objective

Test whether redundancy concentration at early depth is **caused by additive residual accumulation**, or whether it originates inside the sublayer computations (attention, MLP) independently of how their outputs are added to the stream.

The mechanism under test: residual connections add each sublayer's output directly to the stream. If redundancy accumulates because successive additions reinforce the same directions, then scaling the residual contribution should shift or eliminate the collapse. If scaling has no effect, the collapse is internal to the sublayer, not a consequence of the addition itself.

This experiment also tracks optimization stability, because large residual scaling ($\alpha > 1$) can cause training instability. Any geometry change at high $\alpha$ must be distinguished from instability-induced artifacts.

---

## Modification

Modify the residual connection in each transformer block from:

```
H^{l+1} = H^l + A^l + M^l
```

to:

```
H^{l+1} = H^l + alpha * A^l + alpha * M^l
```

where $\alpha$ is a scalar hyperparameter fixed at initialization and constant throughout training. Both the attention and MLP contributions are scaled by the same $\alpha$.

$\alpha = 1.0$ is the standard model (control condition).

---

## Model

| Parameter | Value |
|---|---|
| Architecture | GPT-style decoder-only transformer, Pre-LN |
| n_layer | 6 |
| n_embd | 256 |
| n_head | 4 |
| head_dim | 64 |
| block_size | 128 |
| MLP inner dim | 1024 |
| Dropout | 0.0 |
| Tokenization | Character-level (WikiText-103 vocabulary) |

Train four variants of this model, differing only in $\alpha$:

| Variant | $\alpha$ |
|---|---|
| RS05 | 0.5 |
| RS10 | 1.0 (control — standard transformer) |
| RS15 | 1.5 |
| RS20 | 2.0 |

---

## Dataset

- **Dataset:** WikiText-103 (raw text)
- **Tokenization:** Character-level. Same vocabulary across all variants.
- **Sequence construction:** Concatenate training text, chunk into non-overlapping sequences of 128 tokens.
- **Training data:** WikiText-103 train split
- **Measurement data:** WikiText-103 validation split only

---

## Training

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| LR schedule | Cosine decay to 1e-5 |
| Warmup steps | 2000 |
| Batch size | 64 |
| Total steps | 200,000 |
| Weight decay | 0.1 |
| Beta1, Beta2 | 0.9, 0.95 |
| Gradient clip | 1.0 |
| Seeds | 3 per variant (seeds 0, 1, 2) |

**Checkpoints to save:** steps 500, 1000, 2000, 10000, 50000, 200000.

The 500 and 1000 checkpoints are added here (not present in other experiments) because residual scaling is expected to affect early-training dynamics most strongly. The collapse may appear earlier or later depending on $\alpha$, and you need fine-grained early checkpoints to detect this.

**Total training runs:** 4 variants × 3 seeds = 12 runs.

### Stability Monitoring During Training

Log the following at **every 100 steps** for all variants:

- Training loss
- Validation loss (small subsample, e.g., 64 sequences)
- Gradient norm (before clipping)
- Activation norm: mean $\|H^l\|_2$ across batch at each layer

If RS20 shows diverging loss or gradient norm > 10× the RS10 norm at any checkpoint, flag that checkpoint as potentially unstable. Do not discard the data — report it alongside the stability flags.

---

## Measurement Protocol

At each checkpoint, for each seed, for each $\alpha$ variant:

1. Load checkpoint. Set model to `eval()`. Confirm dropout disabled.
2. Check stability flag. If flagged as unstable at this checkpoint, still measure but annotate results.
3. Sample the first 512 validation sequences of length 128 sequentially.
4. Run forward pass. Extract residual stream $H^l$ at the output of each block (after scaled MLP addition):

```
H_attn  = LayerNorm_1(H^l)
A^l     = Attention(H_attn)
R^l     = H^l + alpha * A^l
H_mlp   = LayerNorm_2(R^l)
M^l     = MLP(H_mlp)
H^{l+1} = R^l + alpha * M^l     ← extract this
```

Also extract $H^0$ (embedding before block 1).

Reshape each to $\mathbb{R}^{65536 \times 256}$. Center (subtract column mean). Compute all metrics.

---

## Metrics

Let $\lambda_1 \geq \ldots \geq \lambda_{256}$ be eigenvalues of $\frac{1}{N} H^{l\top} H^l$.
Let $p_i = \lambda_i / \sum_j \lambda_j$.

### Metric 1: Effective Rank

$$\text{erank}(H^l) = \exp\!\left(-\sum_i p_i \log p_i\right)$$

### Metric 2: Stable Rank

$$\text{srank}(H^l) = \frac{\sum_i \lambda_i}{\lambda_1}$$

### Metric 3: Eigenvalue Dominance

$$\delta(H^l) = \frac{\lambda_1}{\sum_i \lambda_i}$$

### Metric 4: Mean Pairwise Cosine Similarity

$$\bar{c}(H^l) = \frac{1}{M}\sum_{(i,j)} \frac{h_i \cdot h_j}{\|h_i\|\|h_j\|}$$

Use $M = 2000$ pairs, fixed seed = 42.

### Metric 5: Redundancy Index

$$R(H^l) = \delta(H^l) + \bar{c}(H^l)$$

### Metric 6: Argmin Rank Layer

$$l^* = \arg\min_{l \in \{1,\ldots,6\}} \text{erank}(H^l)$$

Record for each variant × seed × checkpoint.

### Metric 7: Collapse Depth (Temporal)

At which checkpoint does the argmin layer first stabilize (i.e., $l^*$ stops changing between consecutive checkpoints)? Compare across variants.

### Metric 8: Activation Norm vs Layer

$$\text{anorm}(H^l) = \frac{1}{N} \sum_{i=1}^{N} \|h_i\|_2$$

Mean L2 norm of token vectors at each layer. This is a stability diagnostic — if $\alpha = 2.0$ causes norms to explode across layers, that explains any geometry changes independently of the redundancy mechanism.

### Metric 9: Gradient Norm (from training logs)

Retrieve from logged training metrics: gradient norm at steps 500, 1000, 2000, 10000, 50000, 200000 for all variants and seeds. This is read from logs, not computed during measurement.

---

## What to Save

For each (variant ∈ {RS05, RS10, RS15, RS20}) × (seed ∈ {0,1,2}) × (checkpoint ∈ {500, 1k, 2k, 10k, 50k, 200k}) × (layer ∈ {0,…,6}):

| Field | Shape |
|---|---|
| `erank` | scalar |
| `srank` | scalar |
| `delta` | scalar |
| `mean_cosine` | scalar |
| `redundancy_index` | scalar |
| `activation_norm` | scalar |
| `eigenvalues` | vector, length 256 |

For each (variant) × (seed) × (checkpoint):

| Field | Shape |
|---|---|
| `argmin_layer` | scalar |
| `val_loss` | scalar |
| `grad_norm` | scalar (from training logs) |
| `stability_flag` | boolean |

---

## Plots

### Plot 1: Effective Rank vs Layer — All Alpha Variants (Final Checkpoint)
- X: layer index (0–6)
- Y: effective rank
- One colored line per variant (RS05, RS10, RS15, RS20)
- Shaded band = ±1 std across 3 seeds
- **Primary figure.** If lines are identical, residual scaling does not affect rank geometry.

### Plot 2: Argmin Layer vs Alpha (Scalar Summary)
- X: $\alpha \in \{0.5, 1.0, 1.5, 2.0\}$
- Y: argmin layer $l^*$
- Points = mean across 3 seeds, error bars = ±1 std
- Final checkpoint only
- If $l^*$ shifts with $\alpha$, the collapse location is residual-dependent

### Plot 3: Redundancy Index vs Layer — All Alpha Variants
- Same structure as Plot 1
- Y: $R = \delta + \bar{c}$

### Plot 4: Eigenvalue Dominance vs Layer — All Alpha Variants
- Same structure as Plot 1
- Y: $\delta$

### Plot 5: Effective Rank Profile Over Training — All Alpha Variants
- 6 subplots (one per checkpoint: 500, 1k, 2k, 10k, 50k, 200k)
- Each subplot: effective rank vs layer, 4 lines (one per $\alpha$)
- Shows when the rank profiles diverge (if ever) across training

### Plot 6: Activation Norm vs Layer — All Alpha Variants
- X: layer index (0–6)
- Y: mean activation norm $\text{anorm}(H^l)$
- One line per variant, final checkpoint
- **Stability diagnostic.** Rapidly growing norms in RS20 indicate instability.

### Plot 7: Gradient Norm vs Training Step — All Alpha Variants
- X: training step
- Y: gradient norm (from training logs)
- One line per variant, mean across 3 seeds
- Log scale on Y
- Any divergence in RS20 or RS15 must be noted before interpreting their rank profiles

### Plot 8: Validation Loss vs Training Step — All Alpha Variants
- X: training step (log scale)
- Y: validation loss
- One line per variant, mean across 3 seeds
- Shows whether scaling degrades or improves learning, independent of geometry

---

## Causal Interpretation Framework

This experiment is designed to test a specific causal question. Before plotting, commit to the following interpretation rules:

**If $l^*$ shifts to later layers as $\alpha$ decreases:**
- Residual accumulation is causal. Smaller $\alpha$ reduces the signal each sublayer writes to the stream, delaying the point at which the stream becomes dominated by a low-rank subspace.

**If $l^*$ is invariant across all $\alpha$:**
- Residual addition is not causal. The collapse originates inside the sublayer computation (attention or MLP), not in how their outputs are summed into the stream. This points toward Experiment 4's subspace novelty results as the primary mechanism.

**If rank profiles change but $l^*$ does not:**
- Scaling changes the magnitude of collapse but not its location. This is a weaker result — the residual mechanism affects severity but not onset.

**If RS20 shows different geometry AND has gradient instability:**
- Cannot attribute the geometry change to residual scaling. Report the instability and exclude RS20 from causal interpretation. Use RS15 as the upper bound instead.

---

## Questions to Answer

1. Does $l^*$ (argmin rank layer) shift with $\alpha$? By how many layers?
2. Is the rank profile of RS05 and RS10 similar, or does even a small scaling change matter?
3. Is RS20 stable throughout training (gradient norm, activation norm)?
4. Does the redundancy index profile shift with $\alpha$, or does the shape stay the same with only magnitude changes?
5. Does $\alpha$ affect validation loss? If so, is there a tradeoff between geometric health and task performance?

---

## Failure Conditions

| Condition | Interpretation |
|---|---|
| All four $\alpha$ variants produce identical rank profiles | Residual scaling is not the cause; look to Experiment 4 sublayer novelty |
| RS20 or RS15 shows diverging loss or gradient norm > 10× RS10 | Flag as unstable; do not interpret geometry from flagged checkpoints |
| $l^*$ shifts but so does val loss by > 0.5 nats | Cannot separate geometry effect from optimization effect |
| Std across seeds > 25% of mean at $l^*$ for any variant | Not robust; increase seeds to 5 for that variant only |

---

## Compute Estimate (Single GPU)

| Task | Time |
|---|---|
| Training one variant, one seed | ~5 hours |
| 4 variants × 3 seeds = 12 runs | ~60 hours |
| Measurement (all variants, seeds, checkpoints) | ~3 hours |
| **Total** | **~63 hours** |

**Recommendation:** Run RS10 (control) and RS05 first. If their profiles differ meaningfully, proceed with RS15 and RS20. If RS10 and RS05 are identical, reconsider whether this experiment is necessary.
