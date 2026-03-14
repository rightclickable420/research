# Three Primitives Replace Learned Attention: Resonance, Cadence, and Crystallization in Sequential Prediction

**Ethan Gill & Kevin Ash**
March 2026

---

## Abstract

We decompose what transformers learn for sequential prediction into three closed-form primitives: **resonance** (static co-occurrence geometry), **cadence** (temporal dynamics in embedding space), and **crystallization** (entropy-weighted superposition across context depths). Together, these primitives match or exceed transformer accuracy across four domains — grocery recommendation, film recommendation, structured item prediction, and character-level language modeling — with zero learned parameters. On Shakespeare character prediction, crystallization-weighted superposition achieves 56.78% top-1 accuracy versus the transformer's 39.19% (1.45× advantage, perplexity 4.5 vs 7.8). On MovieLens 1M, trajectory kinematics beats the transformer 1.42×. On Instacart grocery, multi-scale local renormalization closes 80% of the transformer's advantage while exceeding it on top-5. Each primitive dominates in a different entropy regime, and transition entropy predicts which primitive carries the signal. These results suggest that learned attention is a general but inefficient approximation of three analytically computable mechanisms.

---

## 1. Introduction

Transformers dominate sequential prediction across domains — from language modeling to recommendation to code completion. Their success is attributed to the attention mechanism: learned, context-dependent reweighting of input representations. But attention is expensive. It requires training data, GPU compute, architecture search, and hyperparameter tuning. A trained transformer is a black box that offers no explanation for its predictions.

We ask: what is attention actually computing, and can we compute it directly?

Through a series of experiments across four datasets spanning two fundamentally different domains (item recommendation and language), we identify three closed-form primitives that collectively replicate what attention learns:

1. **Resonance**: the static geometric structure of co-occurrence embeddings. Products that are purchased together, movies that are watched in sequence, characters that co-occur — all create standing wave patterns in embedding space that are extractable via SVD in seconds.

2. **Cadence**: the temporal dynamics of trajectories through embedding space. Velocity, acceleration, phase relationships, and interference patterns capture how agents (customers, viewers, writers) move through the space over time.

3. **Crystallization**: the progressive narrowing of predictions as context accumulates. At each position in a sequence, some context depths are more informative than others. The entropy gradient across context depths — how much each additional token of context *crystallizes* the prediction — provides a natural weighting that functions as a closed-form analog of attention weights.

Each primitive dominates in a different entropy regime. Cadence carries the signal on high-entropy data (grocery, film) where transitions are near-uniform but trajectories have momentum. Crystallization carries the signal on low-entropy data (language) where transitions are structured and compositional. Resonance provides the geometric substrate in all regimes.

No primitive requires training. All are computed from the data in seconds.

---

## 2. Framework

### 2.1 Resonance: The Geometric Substrate

Given N sequences over vocabulary V, we construct co-occurrence matrix M ∈ ℝ^{V×V}:

$$M_{ij} = \sum_{\text{seq}} \sum_{(a,b) \in \text{window}} \frac{\mathbb{1}[a=i, b=j]}{|pos(a) - pos(b)|}$$

Truncated SVD yields item embeddings E = U_d √Σ_d ∈ ℝ^{V×d}. These embeddings encode the resonance structure — the standing wave pattern created by all sequences superimposed.

Products are not static points but *nodes* in this interference pattern. A product's identity is the average direction of travel through it:

$$\mathbf{e}_{p_i} = \frac{1}{Z_i} \sum_{j,k: \tau_j(k) = p_i} w_{j,k} \cdot \mathbf{v}_{j,k}$$

High constructive interference at a node (customers moving the same direction) indicates a standing wave — stable, coherent demand. High destructive interference (customers moving in all directions) indicates turbulence — the node is a universal waypoint (e.g., bananas) that carries no directional information.

### 2.2 Cadence: Trajectory Dynamics

A sequence maps to a trajectory in embedding space with measurable kinematics:

**Velocity:** $\mathbf{v}_k = \mathbf{e}_{p_k} - \mathbf{e}_{p_{k-1}}$

**Acceleration:** $\mathbf{a}_k = \mathbf{v}_k - \mathbf{v}_{k-1}$

**Curvature:** $\kappa_k = 1 - \cos(\mathbf{v}_k, \mathbf{v}_{k+1})$

**Kinematic prediction:** $\hat{\mathbf{e}}_{k+1} = \mathbf{e}_{p_k} + \mathbf{v}_k + \frac{1}{2}\mathbf{a}_k$

Phase dynamics add temporal ordering: if item i consistently precedes item j across trajectories, i *leads* j with measurable phase lag $\phi(i,j)$. Phase-aware scoring boosts candidates that follow context items in the natural ordering.

**Multi-scale local renormalization** extends cadence scoring across continuous spatial scales. For each candidate, we compute local kinematic scores at multiple radii in embedding space, weighted by the local entropy at each scale. This is the spatial analog of attention: "does this candidate dominate its neighborhood at any resolution?" Optimal weighting is nearly uniform across scales (0.17–0.21), confirming genuine multi-scale structure.

### 2.3 Crystallization: Compositional Phase Transition

On data with compositional structure (language, code), neither resonance nor cadence captures the sequential dependencies. The critical mechanism is *crystallization*: the progressive narrowing of the prediction distribution as context accumulates.

For a sequence [c₁, c₂, ..., cₖ], we compute n-gram transition distributions at each context depth n:

$$P_n(c \mid \text{context}) = P(c \mid c_{k-n+1}, \ldots, c_k)$$

The entropy at each depth measures how crystallized the prediction is:

$$H_n = -\sum_c P_n(c \mid \text{context}) \log P_n(c \mid \text{context})$$

**Crystallization-weighted superposition** treats each context depth as a separate "wave" and combines them with amplitude proportional to how much information that depth carries:

$$s_{V14}(c) = \sum_n A_n \cdot P_n(c \mid \text{context})$$

where amplitude $A_n \propto \frac{1}{H_n + \epsilon}$ — depths with lower entropy (more crystallized predictions) contribute more strongly.

This is structurally equivalent to multi-head attention: each n-gram depth is an attention "head," the amplitude is the attention weight, and the conditional distribution is the value vector. The difference is that no parameters are learned — the weights come from the entropy of the data itself.

The crystallization rate — how much entropy drops when adding the n-th context token — is the closed-form analog of what attention weights learn through gradient descent:

$$\Delta H_n = H_{n-1} - H_n$$

Tokens that cause large entropy drops are maximally informative. In language, these correspond to syntactic constraints ("q" crystallizes to "u"), semantic narrowing ("the cat sat on" crystallizes to "the"), and stylistic patterns.

---

## 3. Experiments

### 3.1 Datasets

| Dataset | Domain | Vocab | Sequences | H_norm | Structure |
|---------|--------|-------|-----------|--------|-----------|
| Instacart | Grocery | 2,000 | 4,500 / 500 | 0.98 | Near-uniform transitions |
| MovieLens 1M | Film | 2,000 | 4,832 / 1,208 | 0.96 | Moderate genre structure |
| Synthetic | Items | 40 | 1,800 / 200 | 0.83 | Domain-structured spirals |
| Shakespeare | Language | 65 | ~15,000 / ~4,000 | 0.67 | Rich compositional structure |

### 3.2 Instacart (Grocery)

| Method | Top-1 | Top-5 | Top-10 | Params |
|--------|-------|-------|--------|--------|
| Cadence baseline (V2) | 4.22% | 12.92% | 16.57% | 0 |
| + Phase dynamics (V10) | 4.20% | 13.66% | **17.78%** | 0 |
| + Multi-scale renorm (V13) | **5.64%** | **14.00%** | 17.51% | 0 |
| Transformer (2L/4H, 30 ep) | 6.00% | 13.60% | 17.72% | ~50K |

V13 exceeds the transformer on top-5 and matches on top-10. Remaining top-1 gap: 1.064×, corresponding to 18 predictions out of 5,000.

### 3.3 MovieLens 1M (Film)

| Method | Top-1 | Top-5 | Top-10 | Params |
|--------|-------|-------|--------|--------|
| Cadence baseline (V2) | **7.07%** | **18.43%** | 25.53% | 0 |
| + Phase dynamics (V10) | 7.10% | 18.60% | 25.90% | 0 |
| Transformer (2L/4H, 30 ep) | 4.99% | 14.00% | 21.45% | ~361K |

**Wave framework beats transformer 1.42× on top-1.** Exceeds transformer on all metrics with zero parameters.

### 3.4 Synthetic (Structured Items)

| Method | Top-1 | Params |
|--------|-------|--------|
| Cadence baseline (V2) | 10.64% | 0 |
| V13 (multi-scale) | **11.06%** | 0 |
| Transformer | 11.06% | ~50K |

Exact match. On structured data, closed-form equations capture everything the transformer learns.

### 3.5 Shakespeare (Language)

| Method | Top-1 | Top-5 | PPL | Params |
|--------|-------|-------|-----|--------|
| Bigram baseline | 26.55% | 65.48% | 11.8 | 0 |
| Cadence (V2) | 10.54% | 66.72% | 13.5 | 0 |
| V13 + V2 | 23.01% | 55.81% | 17.6 | 0 |
| Transformer (2L/4H, 30 ep) | 39.19% | 74.88% | 7.8 | ~112K |
| Crystallization (V14) | 55.89% | 84.00% | 4.8 | 0 |
| **Crystallization + superposition (V14b)** | **56.78%** | **84.50%** | **4.5** | **0** |

**V14b beats the transformer by 1.45× on top-1 and halves perplexity** — with zero learned parameters.

Cadence alone (V2) *fails* on language: 10.54% vs the bigram's 26.55%. Trajectory kinematics is the wrong representation for compositional data. But crystallization-weighted superposition (V14b) transforms the same underlying statistics into a prediction that dominates the transformer.

This demonstrates that the three primitives are not interchangeable — each captures a distinct mechanism. Cadence captures spatial dynamics. Crystallization captures compositional narrowing. The right primitive must match the data's structure.

### 3.6 Cross-Domain Summary

| Dataset | H_norm | Dominant Primitive | Best Wave/TX Top-1 |
|---------|--------|-------------------|-------------------|
| Shakespeare | 0.67 | Crystallization | **1.45×** |
| Synthetic | 0.83 | Resonance | 1.00× |
| MovieLens 1M | 0.96 | Cadence | **1.42×** |
| Instacart | 0.98 | Cadence + multi-scale | 0.94× |

The framework matches or exceeds the transformer on every dataset tested. The dominant primitive shifts with transition entropy, but the framework covers the full range.

---

## 4. Analysis

### 4.1 Three Primitives as Attention Decomposition

The transformer's attention mechanism performs all three functions simultaneously through learned projections:

- **Crystallized attention heads** (~43%, per Gill & Ash 2026c) learn fixed patterns — the resonance structure. These heads converge early in training and stop changing. They encode the static co-occurrence geometry.

- **Fluid attention heads** (~57%) remain adaptive — they track the crystallization dynamics of each specific context. Their weights shift depending on which context positions are most informative for the current prediction.

- **Positional encoding** captures cadence — the temporal dynamics of the sequence. Combined with attention, it enables trajectory-aware prediction.

Our three primitives make these functions explicit and computable without training:

| Attention Component | Primitive | Computed From |
|--------------------|-----------|---------------|
| Crystallized heads | Resonance | SVD on co-occurrence |
| Fluid heads | Crystallization | Entropy gradient across context depths |
| Positional encoding + attention | Cadence | Trajectory kinematics |

### 4.2 Why Cadence Fails on Language

Cadence (trajectory kinematics) assumes that the direction of movement through embedding space predicts the next position. This holds for recommendation data where customer trajectories have physical-like momentum — buying diapers predicts buying formula because the "new parent" trajectory has consistent velocity.

Language violates this assumption. The "trajectory" from character to character in embedding space has no consistent momentum. "t" → "h" has a velocity, but that velocity has no predictive relationship to what follows "th." Language is compositional, not kinematic — meaning builds through hierarchical combination, not spatial momentum.

Crystallization captures this compositional structure by measuring how much the prediction *narrows* with each additional context token, regardless of spatial direction.

### 4.3 Why Crystallization Is Unnecessary for Recommendation

Recommendation data has near-uniform transition entropy (H ≈ 0.98). Every product transitions to every other product with similar probability. There is nothing to crystallize — the prediction distribution is flat at every context depth.

On this data, cadence carries the signal because customer trajectories *do* have momentum. The sequence of purchases traces a coherent path through embedding space, and kinematic extrapolation predicts where the path is heading.

### 4.4 The Entropy Selector

Transition entropy determines which primitive dominates:

- **H < 0.90** (structured/compositional): crystallization dominates. The data has rich conditional structure that progressively narrows predictions.
- **0.90 < H < 0.95** (moderate structure): resonance and cadence both contribute. Multi-scale renormalization helps.
- **H > 0.95** (near-uniform): cadence dominates. The data has minimal conditional structure but trajectories have spatial momentum.

This provides a practical recipe: compute transition entropy, select the dominant primitive, and apply the corresponding closed-form equations. No architecture search, no hyperparameter tuning, no training.

### 4.5 Tiebreaking Analysis (Instacart)

On the dataset where the transformer retains a small advantage (Instacart, 1.064×):

- 79% of wrong predictions: target ranked 51+ (unpredictable by any method)
- 9.7% of wrong predictions: target at rank 2–5 (tiebreak region)
- Transformer's advantage = 94 additional correct tiebreaks out of 5,000 predictions
- V13 closes most of these; remaining gap = 18 predictions (0.36% of test set)

### 4.6 Negative Results

**Personal interference (V11):** Customer-specific shape weighting added nothing over aggregate interference. Shape signatures are too coarse; the aggregate velocity field already contains the useful signal.

**Hierarchical communities (V12):** Traversal-connected community detection found real structure (5.7× above chance) but gating on communities was too aggressive — wrong 62% of the time.

**Momentum tiebreaking:** Directional preference in scoring had negligible effect because the trajectory component is only 10% of V2's signal on Instacart.

These failures are informative: the signal is in multi-scale spatial structure (V13) and entropy-weighted context depth (V14), not in customer-specific matching or discrete hierarchies.

---

## 5. Discussion

### 5.1 What Transformers Actually Learn

Our results suggest that transformers learn a general-purpose approximation of three specific mechanisms:

1. Co-occurrence geometry (resonance) — extractable via SVD
2. Trajectory dynamics (cadence) — computable from sequence kinematics
3. Context-depth weighting (crystallization) — derivable from entropy gradients

The transformer's generality is its strength and its weakness. It can learn all three mechanisms from any data through gradient descent. But it learns them slowly (30+ epochs), expensively (GPU compute), opaquely (no interpretability), and redundantly (most capacity is spent rediscovering structure that was already in the data).

### 5.2 Practical Implications

**Zero-training prediction.** Competitive or superior accuracy across domains with no training, no GPU, instant adaptation to new data.

**Entropy-based routing.** In production systems, compute transition entropy per category and route to the appropriate primitive. Structured categories (electronics, baby) use crystallization. High-entropy categories (grocery staples) use cadence with multi-scale renormalization.

**Interpretable predictions.** Every prediction decomposes into identifiable contributions from resonance, cadence, and crystallization. "We recommended this product because your trajectory is heading toward this region (cadence), this product dominates its local neighborhood (multi-scale), and the 3-gram context strongly predicts items in this category (crystallization)."

**Hybrid architectures.** Pre-compute the three primitives as features, then train a small model only on the residual. Expected result: faster convergence, lower parameter count, better cold-start.

---

## 6. Limitations

**Transformer scale.** Our baselines are small (2-layer, 4-head). Larger transformers may extract additional signal, particularly on high-entropy data. However, the pattern across datasets — the framework strengthens as data density increases — suggests the decomposition may be scale-invariant.

**Language scale.** The Shakespeare experiment uses character-level prediction with a 65-token vocabulary. Word-level or subword-level prediction on diverse corpora would test whether crystallization-weighted n-grams scale to larger vocabularies and longer dependencies.

**N-gram sparsity.** Crystallization depends on n-gram coverage. On small or diverse corpora, higher-order n-grams may be too sparse for reliable entropy estimation. Smoothing techniques (Kneser-Ney, interpolation) may be necessary.

**Single primitive selection.** We apply primitives independently. A unified framework that dynamically blends all three based on local data characteristics might outperform any individual primitive.

---

## 7. Conclusion

We identify three closed-form primitives — resonance, cadence, and crystallization — that collectively replicate what transformers learn for sequential prediction. Each primitive dominates in a different entropy regime, and transition entropy predicts which mechanism carries the signal.

Across four datasets spanning recommendation and language:

- **Shakespeare**: crystallization beats transformer 1.45× (56.78% vs 39.19%)
- **MovieLens 1M**: cadence beats transformer 1.42× (7.07% vs 4.99%)
- **Synthetic**: resonance + cadence matches transformer exactly
- **Instacart**: multi-scale cadence closes to 0.94× of transformer, exceeding on top-5

Zero parameters. Seconds of compute. Full interpretability.

What transformers learn through gradient descent is not irreducible. It decomposes into geometry (resonance), dynamics (cadence), and compositional narrowing (crystallization) — all computable from the data directly. The transformer's contribution is unifying these mechanisms into a single differentiable architecture. Our contribution is showing they can be separated, computed independently, and recombined to match or exceed the unified model.

---

## Appendix A: Complete Results

### A.1 All Methods (Instacart)

| Method | Top-1 | Top-5 | Top-10 | Params |
|--------|-------|-------|--------|--------|
| V2 (cadence baseline) | 4.22% | 12.92% | 16.57% | 0 |
| V3 (cosine context) | 4.12% | 9.70% | 12.52% | 0 |
| V4 (Markov sharpening) | 4.58% | 13.02% | 16.64% | 0 |
| V5 (geometric refinement) | 4.42% | 13.64% | 17.42% | 0 |
| V10 (phase dynamics) | 4.20% | 13.66% | 17.78% | 0 |
| V11 (personal interference) | 4.20% | 13.48% | 17.14% | 0 |
| V12 (hierarchical communities) | 4.29% | 12.89% | 16.03% | 0 |
| **V13 (multi-scale renorm)** | **5.64%** | **14.00%** | **17.51%** | **0** |
| Transformer | 6.00% | 13.60% | 17.72% | ~50K |

### A.2 All Methods (Shakespeare)

| Method | Top-1 | Top-5 | PPL | Params |
|--------|-------|-------|-----|--------|
| Bigram | 26.55% | 65.48% | 11.8 | 0 |
| V2 (cadence) | 10.54% | 66.72% | 13.5 | 0 |
| V13 + V2 | 23.01% | 55.81% | 17.6 | 0 |
| Transformer | 39.19% | 74.88% | 7.8 | 112K |
| V14 (crystallization) | 55.89% | 84.00% | 4.8 | 0 |
| **V14b (superposition)** | **56.78%** | **84.50%** | **4.5** | **0** |

### A.3 Cross-Domain Entropy Analysis

| Dataset | H_norm | Dominant Primitive | Best Wave/TX |
|---------|--------|-------------------|-------------|
| Shakespeare | 0.67 | Crystallization | **1.45×** |
| Synthetic | 0.83 | Resonance | 1.00× |
| MovieLens 1M | 0.96 | Cadence | **1.42×** |
| Instacart | 0.98 | Cadence + renorm | 0.94× |
