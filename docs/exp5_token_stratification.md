# Experiment 5: Token Position Stratification (Diagnostic)

## Objective

Determine whether early-depth redundancy concentration is **uniform across all token positions**, or whether it is driven by representations at specific positions in the sequence.

This is a diagnostic sanity check, not a hypothesis-testing experiment. It ensures that the redundancy signal seen in Experiments 1–4 is not an artifact of a small number of positions (e.g., position 0 due to positional embedding geometry, or late positions due to causal masking). If redundancy is uniform across positions, the phenomenon generalizes to the full residual stream. If it is position-specific, the earlier experiments need to be reinterpreted.

This experiment is intentionally narrow in scope. It does not compute full rank metrics across all layers — only the most informative summary statistics at each position group.

---

## Model

Use the same trained model as Experiment 1 (6-layer, d=256, character-level WikiText-103). Do not train a new model.

| Parameter | Value |
|---|---|
| Architecture | GPT-style decoder-only transformer, Pre-LN |
| n_layer | 6 |
| n_embd | 256 |
| n_head | 4 |
| block_size | 128 |
| Dropout | 0.0 |
| Tokenization | Character-level |

Use checkpoints from seeds 0, 1, 2 at step **200,000 only**. This experiment uses the final trained model.

---

## Dataset

- **Dataset:** WikiText-103 validation split
- **Sequences:** First 512 validation sequences of length 128, sampled sequentially (same as all other experiments)
- **No training is performed in this experiment**

---

## Measurement Protocol

1. Load the step-200k checkpoint for seeds 0, 1, 2. Set to `eval()`. Confirm dropout disabled.
2. Sample the first 512 validation sequences of length 128.
3. Run forward pass. Extract residual stream $H^l \in \mathbb{R}^{512 \times 128 \times 256}$ at **every layer** $l \in \{0, 1, \ldots, 6\}$.

### Position Grouping

Split the 128 token positions into 4 groups:

| Group | Positions | Label |
|---|---|---|
| Early | 0–31 | First 25% of sequence |
| Mid-early | 32–63 | Second 25% |
| Mid-late | 64–95 | Third 25% |
| Late | 96–127 | Final 25% |

For each group, slice $H^l$ along the sequence dimension to get:

$$H^l_{\text{group}} \in \mathbb{R}^{512 \times 32 \times 256}$$

Reshape to $\mathbb{R}^{16384 \times 256}$. Center by subtracting column mean (mean over the 16384 dimension).

---

## Metrics

Compute only the following two metrics per group per layer. Do not compute full rank decompositions.

### Metric 1: Mean Pairwise Cosine Similarity (Within Group)

Sample $M = 1000$ random pairs from the 16384 token vectors of each group (fixed seed = 42):

$$\bar{c}_{\text{group}}(H^l) = \frac{1}{M} \sum_{(i,j)} \frac{h_i \cdot h_j}{\|h_i\|\|h_j\|}$$

This measures how similar token representations are within each position group at each layer.

### Metric 2: Attention Entropy per Position

For each head $h$ at each layer $l$, the attention weight matrix for a given sequence has shape $(128, 128)$ (query positions × key positions, causally masked).

For each query position $t$, compute the entropy of the attention distribution:

$$\text{entropy}(t, h, l) = -\sum_{s \leq t} \alpha_{ts}^{(h,l)} \log \alpha_{ts}^{(h,l)}$$

where $\alpha_{ts}^{(h,l)}$ is the attention weight from query $t$ to key $s$ at head $h$, layer $l$.

Average across heads and sequences:

$$\bar{\text{entropy}}(t, l) = \frac{1}{H \cdot B} \sum_{h=1}^{H} \sum_{b=1}^{B} \text{entropy}(t, h, l)$$

where $H = 4$ heads, $B = 512$ sequences.

This produces a matrix of shape $(128, 6)$: entropy at each position and each layer.

---

## What to Save

For each (seed ∈ {0,1,2}) × (layer ∈ {0,…,6}) × (group ∈ {early, mid-early, mid-late, late}):

| Field | Shape |
|---|---|
| `mean_cosine_by_group` | scalar |

For each (seed ∈ {0,1,2}) × (layer ∈ {1,…,6}) × (position ∈ {0,…,127}):

| Field | Shape |
|---|---|
| `attention_entropy` | scalar |

---

## Plots

### Plot 1: Mean Cosine Similarity by Position Group vs Layer
- X: layer index (0–6)
- Y: mean cosine similarity
- One line per position group (4 lines: Early, Mid-early, Mid-late, Late)
- Mean ± std across 3 seeds
- If all four lines track together, redundancy is position-uniform

### Plot 2: Attention Entropy vs Position — All Layers
- X: token position (0–127)
- Y: mean attention entropy
- One colored line per layer (6 lines)
- Mean across 3 seeds
- Reveals whether attention is focused or diffuse at each position, and how this changes with depth

### Plot 3: Cosine Similarity at Layer 1 vs Token Position
- X: token position (0–127)
- Y: mean cosine similarity computed between each position's vector and all other positions' vectors in the same sequence (mean over sequences and seeds)
- This is a position-resolved version of the group metric
- Highlights whether position 0 is anomalous

---

## Questions to Answer

1. Is mean cosine similarity approximately equal across the four position groups at the layer with maximum global redundancy?
2. Is position 0 anomalous compared to other positions? (Expected: yes, due to positional embedding geometry)
3. Does attention entropy drop at early layers for all positions, or only specific positions?
4. Is there a correlation between low attention entropy and high cosine similarity at any layer-position combination?

---

## Interpretation Rules

- **If all position groups show the same cosine similarity pattern vs layer:** Position stratification does not affect the main finding. Report this as a positive control.
- **If early positions show much higher cosine similarity than late positions at all layers:** The redundancy signal is front-loaded in the sequence, not the depth. This would require reinterpreting Experiments 1–4.
- **If only position 0 is anomalous:** This is a known artifact of positional embeddings. Note it, exclude position 0 from other experiments if needed, and move on.

---

## Failure Conditions

| Condition | What it means |
|---|---|
| Early position group cosine similarity > 2× late group at collapsed layer | Redundancy is position-specific; Experiments 1–4 need position-excluded reruns |
| Attention entropy at layer 1 is near-zero for all positions | Attention has collapsed to near-argmax — verify this isn't a training issue |
| Std across 3 seeds > 30% of mean for any metric | Not enough seeds for this diagnostic; add 2 more |

---

## Compute Estimate

No training required. Uses checkpoints from Experiment 1 (seeds 0, 1, 2, step 200k).

| Task | Time |
|---|---|
| Measurement (3 seeds, all layers, attention extraction) | ~30 minutes |
| Plotting | ~30 minutes |
| **Total** | **~1 hour** |
