# Experiment 4: Component-wise Source of Redundancy

## Objective

Identify whether redundancy is introduced by the **attention sublayer**, the **MLP sublayer**, or the **residual addition** itself, by measuring effective rank at every intermediate point within each transformer block.

This experiment goes one level deeper than profiling the final residual stream: it attributes the rank change at each layer to a specific computational component.

Critically, this experiment also measures **subspace novelty** — whether each sublayer is introducing new representational directions or reinforcing existing ones. This distinction separates a descriptive finding ("rank drops") from a mechanistic one ("MLP fails to introduce new directions at early layers").

---

## Model

Use a single model trained to convergence. Do not train a new model for this experiment.

| Parameter | Value |
|---|---|
| Architecture | GPT-style decoder-only transformer, Pre-LN |
| n_layer | 6 |
| n_embd (d_model) | 256 |
| n_head | 4 |
| head_dim | 64 |
| block_size | 128 |
| MLP inner dim | 1024 |
| Dropout | 0.0 |
| Tokenization | Character-level (WikiText-103 vocabulary) |

Train this model once with seed 0, 1, and 2 (3 seeds). Use checkpoints at steps **2000, 50000, 200000**.

---

## Dataset

- **Dataset:** WikiText-103 (raw text)
- **Tokenization:** Character-level. Same vocabulary as used in training.
- **Measurement data:** WikiText-103 validation split only.
- **Sequences:** First 512 validation sequences of length 128, sampled sequentially.

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
| Seeds | 3 (seeds 0, 1, 2) |

Save checkpoints at steps: 2000, 50000, 200000.

---

## Measurement Protocol

At each checkpoint, for each seed:

1. Load checkpoint. Set model to `eval()`. Confirm dropout is disabled.
2. Sample the first 512 validation sequences of length 128.
3. Run a forward pass with hooks to extract **5 intermediate tensors per layer**.

### Exact Extraction Points

The Pre-LN transformer block computes:

```
Input:     H^l                                     ← (1) extract here
           ↓
           LayerNorm_1(H^l)
           ↓
           Attention(...)  → A^l
           ↓
Add:       R^l = H^l + A^l                         ← (2) extract here
           ↓
           LayerNorm_2(R^l)
           ↓
           MLP(...)  → M^l
           ↓
Add:       H^{l+1} = R^l + M^l                    ← (3) extract here

Also extract:
           A^l   (the attention output before addition) ← (4) extract here
           M^l   (the MLP output before addition)       ← (5) extract here
```

**Do NOT extract LayerNorm outputs.** Do NOT extract post-LayerNorm inputs to attention or MLP. Extract only the five tensors listed: $H^l$, $A^l$, $R^l$, $M^l$, $H^{l+1}$.

Also extract $H^0$ (embedding before block 1).

Each tensor has shape $(512, 128, 256)$. Reshape to $\mathbb{R}^{65536 \times 256}$ and center (subtract column mean) before all metric computations.

---

## Metrics

All metrics are computed on a matrix $X \in \mathbb{R}^{N \times d}$ with $N = 65536$, $d = 256$.

Let $\lambda_1 \geq \ldots \geq \lambda_d$ be eigenvalues of $\frac{1}{N} X^\top X$.
Let $p_i = \lambda_i / \sum_j \lambda_j$.

### Metric 1: Effective Rank

$$\text{erank}(X) = \exp\!\left(-\sum_i p_i \log p_i\right)$$

### Metric 2: Stable Rank

$$\text{srank}(X) = \frac{\sum_i \lambda_i}{\lambda_1}$$

### Metric 3: Eigenvalue Dominance

$$\delta(X) = \frac{\lambda_1}{\sum_i \lambda_i}$$

### Metric 4: Rank Change Due to Attention

$$\Delta_{\text{attn}}^l = \text{erank}(R^l) - \text{erank}(H^l)$$

Positive = attention increases rank (adds information). Negative = attention reduces rank (concentrates information).

### Metric 5: Rank Change Due to MLP

$$\Delta_{\text{mlp}}^l = \text{erank}(H^{l+1}) - \text{erank}(R^l)$$

Positive = MLP increases rank. Negative = MLP reduces rank.

### Metric 6: Subspace Novelty of Attention Output

This is the most important metric in this experiment. It measures whether $A^l$ introduces directions that are genuinely new relative to $H^l$, or merely reinforces existing directions.

**Procedure:**

1. Compute the SVD of $H^l$ (centered): $H^l = U \Sigma V^\top$, where $V \in \mathbb{R}^{d \times d}$.
2. Take the top-$k$ right singular vectors $V_k \in \mathbb{R}^{d \times k}$ that capture 90% of variance in $H^l$. Choose $k$ as the smallest integer such that $\sum_{i=1}^k \lambda_i / \sum_j \lambda_j \geq 0.90$.
3. Project $A^l$ onto the orthogonal complement of the $H^l$ subspace:

$$A^l_\perp = A^l (I - V_k V_k^\top)$$

4. Compute the effective rank of $A^l_\perp$:

$$\text{novelty}_{\text{attn}}^l = \text{erank}(A^l_\perp)$$

High novelty = attention is writing into directions that $H^l$ does not already contain.
Low novelty = attention is only reinforcing existing directions in $H^l$.

**Also compute** the fraction of variance of $A^l$ that lies in the complement:

$$\text{novelty\_frac}_{\text{attn}}^l = \frac{\|A^l_\perp\|_F^2}{\|A^l\|_F^2}$$

This tells you what fraction of the attention output is genuinely new, vs. what fraction is a projection into existing directions.

### Metric 7: Subspace Novelty of MLP Output

Same procedure as Metric 6, but for $M^l$ relative to $R^l$.

1. Take top-$k$ right singular vectors $V_k$ of $R^l$ capturing 90% of its variance.
2. Project: $M^l_\perp = M^l (I - V_k V_k^\top)$
3. Compute:

$$\text{novelty}_{\text{mlp}}^l = \text{erank}(M^l_\perp)$$

$$\text{novelty\_frac}_{\text{mlp}}^l = \frac{\|M^l_\perp\|_F^2}{\|M^l\|_F^2}$$

### Metric 8: Mean Pairwise Cosine Similarity

For $H^l$, $R^l$, $H^{l+1}$ (not for $A^l$, $M^l$ which are updates, not representations):

$$\bar{c}(X) = \frac{1}{M} \sum_{(i,j)} \frac{x_i \cdot x_j}{\|x_i\|\|x_j\|}$$

Use $M = 2000$ random pairs, fixed seed = 42.

### Metric 9: Redundancy Index

$$R(X) = \delta(X) + \bar{c}(X)$$

Compute for $H^l$, $R^l$, $H^{l+1}$ at each layer.

---

## What to Save

For each (seed ∈ {0,1,2}) × (checkpoint ∈ {2k, 50k, 200k}) × (layer ∈ {1,…,6}):

| Field | Shape |
|---|---|
| `erank_input` (for $H^l$) | scalar |
| `erank_post_attn` (for $R^l$) | scalar |
| `erank_output` (for $H^{l+1}$) | scalar |
| `srank_input` | scalar |
| `srank_post_attn` | scalar |
| `srank_output` | scalar |
| `delta_input` | scalar |
| `delta_post_attn` | scalar |
| `delta_output` | scalar |
| `delta_rank_attn` | scalar |
| `delta_rank_mlp` | scalar |
| `novelty_attn` | scalar |
| `novelty_mlp` | scalar |
| `novelty_frac_attn` | scalar |
| `novelty_frac_mlp` | scalar |
| `mean_cosine_input` | scalar |
| `mean_cosine_post_attn` | scalar |
| `mean_cosine_output` | scalar |
| `redundancy_index_input` | scalar |
| `redundancy_index_output` | scalar |
| `val_loss` | scalar (per seed/ckpt) |

---

## Plots

### Plot 1: Rank Progression Through Sublayers — All Layers (Final Checkpoint)
- X: computation stage (5 points per layer: $H^l$, $A^l$, $R^l$, $M^l$, $H^{l+1}$)
- Y: effective rank
- One panel per layer (6 panels), arranged in a row
- Mean ± std across 3 seeds
- This shows where within each block the rank changes

### Plot 2: Rank Change by Component vs Layer
- X: layer index (1–6)
- Two lines: $\Delta_{\text{attn}}^l$ and $\Delta_{\text{mlp}}^l$
- Horizontal dashed line at 0
- Mean ± std across seeds
- Shows which sublayer is the primary contributor to rank change at each layer

### Plot 3: Subspace Novelty vs Layer
- X: layer index (1–6)
- Two lines: $\text{novelty}_{\text{attn}}^l$ and $\text{novelty}_{\text{mlp}}^l$
- Shaded band = ±1 std across seeds
- Final checkpoint only
- **Key mechanistic figure.** Low novelty at early layers = sublayer is not introducing new directions.

### Plot 4: Novelty Fraction vs Layer
- Same structure as Plot 3
- Y: $\text{novelty\_frac}_{\text{attn}}^l$ and $\text{novelty\_frac}_{\text{mlp}}^l$
- Shows what fraction of each sublayer's output is genuinely new

### Plot 5: Novelty Over Training Time (Attention Only)
- X: layer index
- 3 subplots: checkpoints 2k, 50k, 200k
- Y: $\text{novelty\_frac}_{\text{attn}}$
- Shows whether attention learns to introduce new directions as training progresses

### Plot 6: Novelty Over Training Time (MLP Only)
- Same as Plot 5 for MLP

### Plot 7: Redundancy Index Through the Block — All Layers
- X: computation stage (input → post-attn → output)
- One panel per layer
- Y: redundancy index $R$
- Shows whether redundancy is being added or removed at each sublayer

### Plot 8: Cosine Similarity — Input vs Post-Attention vs Output
- X: layer index
- Three lines: $\bar{c}(H^l)$, $\bar{c}(R^l)$, $\bar{c}(H^{l+1})$
- Mean ± std across seeds, final checkpoint

---

## Questions to Answer

1. Which sublayer (attention or MLP) is the primary driver of rank change at early layers?
2. Is the rank change by MLP consistently negative at early layers? Consistently positive at later layers?
3. What is the novelty fraction of attention and MLP at layer 1? Is it below 0.1 (nearly no new directions) or above 0.5 (majority new directions)?
4. Does novelty fraction increase over training, or does it remain low from step 2k onward?
5. Is the dominant component of redundancy directional (single direction) or subspace (multi-dimension)?
6. Does the subspace novelty analysis at layer 1 suggest that both sublayers are writing into the same subspace already present in $H^1$?

---

## Failure Conditions

| Condition | Interpretation |
|---|---|
| $\Delta_{\text{attn}}$ and $\Delta_{\text{mlp}}$ are both near zero at all layers | Rank changes are negligible — the phenomenon may be elsewhere |
| Novelty fraction > 0.8 at early layers | Updates are mostly new — redundancy is not coming from sublayer collapse |
| Novelty fraction is the same at all layers | No depth-dependence in the mechanism — consider other explanations |
| Std across seeds > 25% of mean for novelty metrics | Subspace geometry is not stable across initializations |

---

## Compute Estimate (Single GPU)

| Task | Time |
|---|---|
| Training, 1 seed to 200k steps | ~5 hours |
| 3 seeds total | ~15 hours |
| Measurement (all checkpoints, all seeds) | ~1 hour |
| SVD computation (6 layers × 5 tensors × 3 ckpts × 3 seeds) | ~30 minutes |
| **Total** | **~17 hours** |

This is the least expensive experiment because only one model configuration is trained.
