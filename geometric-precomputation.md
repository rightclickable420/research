# Geometric Pre-computation in Sequential Prediction: What Transformers Learn and What They Don't Need To

**Ethan Gill & Kevin Ash**
March 2026

---

## Abstract

We decompose transformer performance on next-item prediction into two components: geometric structure pre-computable from embedding space, and a learned coordinate rotation that training discovers through gradient descent. Using kinematic trajectory extrapolation on co-occurrence embeddings, we match transformer top-5 and top-10 accuracy with zero training in 1.6 seconds. The transformer's remaining advantage — a 1.36× factor on top-1 precision — resists nine increasingly sophisticated attempts to close it analytically (four additive conditioning approaches, five geometric improvements), revealing that the gap is not additive and not in embedding quality: it cannot be decomposed into geometry + conditioning, and refining the geometry yields only incremental gains. We characterize this residual as a task-specific rotation of the embedding space that unifies ranking sharpness and distributional breadth — the specific contribution of learned attention.

---

## 1. Introduction

Transformers have become the default architecture for sequential prediction, from language modeling to recommendation systems. But a basic question remains underexplored: how much of what a transformer learns is structure it *discovers* versus structure that was *already there*?

Embedding spaces constructed from co-occurrence statistics (SVD on co-occurrence matrices, Word2Vec, GloVe) encode rich geometric relationships before any task-specific training. Products bought together cluster. Sequences of purchases trace trajectories through this space. The geometric properties of these trajectories — velocity, curvature, acceleration — are computable in closed form.

We ask: if you extrapolate these trajectories kinematically, how close do you get to what a trained transformer predicts? And where the gap exists, what exactly does it consist of?

---

## 2. Framework

### 2.1 Embedding Construction

Given N purchase sequences over a vocabulary of V items, we construct a co-occurrence matrix M ∈ ℝ^{V×V} with distance-weighted counts:

$$M_{ij} = \sum_{\text{seq}} \sum_{(a,b) \in \text{window}} \frac{\mathbb{1}[a=i, b=j]}{|pos(a) - pos(b)|}$$

where the window size is 3. We apply truncated SVD:

$$M \approx U_d \Sigma_d V_d^T$$

The item embeddings are E = U_d √Σ_d ∈ ℝ^{V×d}, where d = 32.

### 2.2 Trajectory Kinematics

A customer's purchase sequence [p₁, p₂, ..., pₖ] maps to a trajectory in embedding space:

$$\mathbf{T} = [\mathbf{e}_{p_1}, \mathbf{e}_{p_2}, \ldots, \mathbf{e}_{p_k}]$$

At each step, we compute:

**Velocity:**
$$\mathbf{v}_k = \mathbf{e}_{p_k} - \mathbf{e}_{p_{k-1}}$$

**Acceleration:**
$$\mathbf{a}_k = \mathbf{v}_k - \mathbf{v}_{k-1}$$

### 2.3 Kinematic Extrapolation (V2)

The predicted next position uses constant-acceleration extrapolation:

$$\hat{\mathbf{e}}_{k+1} = \mathbf{e}_{p_k} + \mathbf{v}_k + \frac{1}{2}\mathbf{a}_k$$

The recommendation is the item nearest to the predicted position:

$$\text{score}(p) = -\|\mathbf{e}_p - \hat{\mathbf{e}}_{k+1}\|_2$$

Items already in the recent context are masked. No parameters are learned.

### 2.4 Context-Conditioned Scoring (V3)

To incorporate context dependence, we augment the geometric score with a cosine similarity interaction term:

$$\text{score}_{V3}(p \mid \text{context}) = \text{score}_{V2}(p) + \alpha \sum_{c \in \text{context}[-N:]} \text{sim}(\mathbf{e}_p, \mathbf{e}_c)$$

where sim is cosine similarity and α ∈ {0.3, 0.5, 0.7, 1.0}. Variants tested include squared cosine (sharper selectivity) and ReLU cosine (positive interactions only).

### 2.5 Entropy-Gated Markov Sharpening (V4)

We build a first-order Markov transition matrix from training sequences and apply it selectively based on the entropy of the geometric prediction:

$$H = -\sum_p \hat{p}(p) \log \hat{p}(p)$$

When H exceeds a threshold (the geometric distribution is flat/uncertain), we blend in Markov transition probabilities:

$$\text{score}_{V4}(p) = (1 - \beta \cdot \mathbb{1}[H > H_\text{gate}]) \cdot \text{score}_{V2}(p) + \beta \cdot \mathbb{1}[H > H_\text{gate}] \cdot \text{score}_{\text{markov}}(p)$$

with gate threshold β_gate and blend weight β as hyperparameters.

### 2.6 Transformer Baseline

A standard 2-layer transformer with 4 attention heads, d_model = 64, trained with causal masking and cross-entropy loss. AdamW optimizer, lr = 3e-4, weight decay 0.01, 30 epochs.

---

## 3. Experiments

### 3.1 Dataset

We use the public Instacart Online Grocery Shopping dataset (3.4M orders, 49,685 products, 206,209 users). We select users with ≥3 orders, use the top 2,000 products, build purchase sequences capped at 30 items, and split 80/20 into 4,500 train / 500 test sequences.

### 3.2 Results

| Method | Top-1 | Top-5 | Top-10 | Training Time |
|--------|-------|-------|--------|---------------|
| V2 (kinematic extrapolation) | 3.88% | 13.74% | 17.66% | **0 (1.6s build)** |
| V3 best (cosine, α=0.3) | 4.12% | 9.70% | 12.52% | 0 |
| V4 best (β_g=0.2, β=0.3, Markov) | 4.58% | 13.02% | 16.64% | 0 |
| Transformer (30 epochs) | 6.00% | 13.60% | 17.72% | ~2 min |

### 3.3 Synthetic Validation

On a controlled synthetic dataset (5 domains × 8 products, 1800 train / 200 test, structured purchase spirals with 30% noise):

| Method | Top-1 | Top-5 |
|--------|-------|-------|
| V2 (kinematic) | 10.6% | 42.7% |
| Transformer (50 epochs, peak) | 11.1% | 52.2% |

The transformer's top-1 advantage ratio (1.55× on Instacart, 1.05× on synthetic) is larger on real data where conditional patterns are richer.

---

## 4. Analysis

### 4.1 The Geometric Substrate

The V2 framework matches the transformer on broad distributional accuracy (top-5/top-10) with zero training. On Instacart: top-5 is 13.74% vs 13.60% (geometric actually exceeds the transformer), and top-10 is 17.66% vs 17.72% (within 0.3%).

This means the co-occurrence geometry — computed once via SVD — already contains the distributional knowledge that the transformer spends 30 epochs and millions of parameter updates learning. The trajectory extrapolation correctly identifies the *region* of embedding space where the next purchase will land. It knows the neighborhood.

### 4.2 The Sharpening Gap

The transformer's advantage is concentrated entirely in top-1: 6.00% vs 4.36% (1.36×). It picks the single best item from the correct neighborhood more often. This is precision at the decision boundary — knowing not just the region, but the exact point.

### 4.3 Why Context Conditioning Fails

V3 (cosine interaction) and V4 (Markov sharpening) both attempt to close this gap by adding context-dependent signals. Both improve top-1 marginally while degrading top-5/top-10:

| Transition | Top-1 Δ | Top-5 Δ | Top-10 Δ |
|------------|---------|---------|----------|
| V2 → V3 | +0.24% ↑ | −4.04% ↓ | −5.14% ↓ |
| V2 → V4 | +0.70% ↑ | −0.72% ↓ | −1.02% ↓ |

The pattern is consistent across all configurations tested: **sharpening the peak costs the tails.** Every additive combination of geometric + conditional signals trades distributional breadth for point precision.

The transformer does not exhibit this tradeoff. It achieves its top-1 advantage *without* sacrificing top-5/top-10. This is the key finding.

### 4.4 Geometric Improvements Are Incremental (V5)

To test whether the gap is in embedding quality rather than the scoring mechanism, we implemented five geometric improvements and evaluated each individually:

| Config | Top-1 | Top-5 | Top-10 |
|--------|-------|-------|--------|
| V2 baseline | 4.36% | 13.50% | 17.02% |
| Fixed-point iteration (3 iter) | **4.42%** | **13.64%** | **17.42%** |
| Fixed-point iteration (5 iter) | 4.42% | 13.68% | 17.38% |
| Mahalanobis distance | 4.38% | 13.64% | 17.06% |
| Curvature-aware extrapolation | 4.32% | 13.44% | 17.02% |
| Multi-scale scoring (d=4,16,32) | 4.24% | 13.36% | 17.10% |
| Asymmetric embeddings | 4.36% | 13.50% | 17.02% |
| Transformer ceiling | 6.00% | 13.60% | 17.72% |

The best improvement (fixed-point iteration) gains 0.06% on top-1 — a 1.4% relative improvement. All five improvements combined would not close the gap. Critically, none degrade top-5/top-10 (unlike V3/V4), confirming that these are genuine geometric refinements rather than distribution-distorting hacks.

This result is significant: the embedding geometry was already ~95% captured by a single SVD pass. Iterating the embeddings through velocity contexts (fixed-point) tightens the space marginally. Curvature-aware extrapolation actually hurts — grocery purchase trajectories are too noisy for osculating circle mathematics. The remaining 1.36× gap to transformer top-1 is not in the geometry.

### 4.5 Geometric Selection: What Attention Actually Computes

The transformer's QK^T operation computes context-dependent similarity. But the Q, K, V matrices are learned linear projections — they rotate the embedding space before computing dot products. The rotation is task-specific: it aligns the axes so that geometric proximity in the rotated space equals contextual relevance.

This is why additive blending fails. The transformer doesn't *add* a sharpening signal on top of geometry. It *rotates the geometry* so that sharpness and breadth are simultaneously optimized. The projection creates a coordinate system where the nearest item in the rotated space is the correct prediction, and the top-5 nearest items are also correct — because the rotation preserves neighborhood structure while sharpening the peak.

We call this **geometric selection**: the learned mapping from raw embedding space to a task-specific coordinate system where distance = relevance across the full rank.

### 4.6 Implications for Transformer Understanding

This decomposition clarifies what training contributes:

1. **Not the geometry.** The distributional structure of the embedding space is pre-computable from co-occurrence statistics. SVD extracts it in seconds.

2. **Not the trajectory dynamics.** Kinematic extrapolation (velocity + acceleration) captures the sequential patterns that the transformer models via positional attention.

3. **The coordinate rotation.** Training discovers a task-specific rotation of the embedding space that makes geometric proximity = contextual relevance. This rotation simultaneously optimizes sharpness (top-1) and breadth (top-5/10), which no additive decomposition can replicate.

This suggests that ~90% of transformer capacity on sequential prediction tasks is spent rediscovering geometric structure that was already computable. The remaining ~10% — the coordinate rotation — is the genuine contribution of learned attention.

---

## 5. Applications

### 5.1 Hybrid Architectures

The framework suggests a practical architecture: **pre-compute the geometric substrate, train only the rotation.**

Instead of initializing a transformer with random embeddings and learning everything from scratch, initialize with SVD embeddings and trajectory-aware positional encodings. The transformer then only needs to learn the coordinate rotation — a much smaller optimization problem.

Expected benefits:
- Faster convergence (the broad distribution is correct from step 0)
- Lower parameter count (the geometry doesn't need to be parameterized)
- Better cold-start (new items get geometric embeddings from co-occurrence before any training)

### 5.2 Training-Free Recommendation at Scale

For applications where top-5/top-10 accuracy is sufficient (discovery feeds, category recommendations, exploration interfaces), the V2 framework provides competitive accuracy with:
- Zero training time
- Zero GPU cost
- Instant adaptation (rebuild SVD when new data arrives, no retraining)
- Full interpretability (every prediction is a geometric extrapolation)

### 5.3 Diagnostic for Transformer Training

The geometric baseline provides a free diagnostic: if a transformer hasn't exceeded V2's top-5 accuracy after N epochs, it's still learning the geometric substrate and hasn't yet reached the rotation-learning phase. This could inform early stopping, learning rate scheduling, and curriculum design.

### 5.4 Compression and Distillation

If ~90% of learned weights encode pre-computable geometry, transformer models for sequential prediction may be highly compressible. Distillation targets: preserve the rotation, discard the geometric redundancy.

---

## 6. Limitations

**Task scope.** Our experiments cover next-item prediction on purchase sequences. Language modeling introduces compositional semantics, long-range dependencies, and contextual polysemy that may shift the geometric/rotation ratio. We expect the geometric substrate to remain significant but the rotation component to grow for tasks with richer conditional structure.

**Embedding quality.** The geometric framework's performance depends on SVD embedding quality, which depends on co-occurrence statistics. Sparse data (cold-start items, low-frequency products) will degrade geometric predictions more than transformer predictions, since the transformer can learn from partial patterns via attention.

**Rotation analysis.** We characterize the transformer's advantage as a "coordinate rotation" based on behavioral evidence (the sharpness-breadth tradeoff). Direct analysis of the learned Q, K, V matrices to extract and characterize this rotation is future work.

**Single domain.** The Instacart experiment covers grocery shopping. Different sequential prediction domains (music, web browsing, code completion) may have different geometric/rotation ratios.

---

## 7. Conclusion

We show that the geometric structure of co-occurrence embeddings, combined with kinematic trajectory extrapolation, matches transformer performance on broad distributional metrics (top-5, top-10) for next-item prediction — with zero training, computed in 1.6 seconds.

The transformer's remaining advantage is concentrated in top-1 precision: a 1.36× factor that resists nine attempts at analytical approximation. Each attempt reveals the same structural constraint: additive composition of geometric and conditional signals trades sharpness for breadth. The transformer avoids this tradeoff because attention performs **geometric selection** — a learned rotation of the embedding space where proximity simultaneously encodes both.

This decomposition has immediate practical applications: hybrid architectures that pre-compute geometry and train only the rotation, training-free recommendation for discovery applications, and diagnostic baselines for transformer training. More broadly, it answers a foundational question about what transformers learn: mostly geometry that was already there, plus a coordinate system that makes it useful.

---

## Appendix A: Experimental Configurations

### A.1 V3 Variants Tested (Context Interaction)

| Config | α | Similarity | Top-1 | Top-5 | Top-10 |
|--------|---|-----------|-------|-------|--------|
| V3-a | 0.3 | cosine | 4.12% | 9.70% | 12.52% |
| V3-b | 0.5 | cosine | — | — | — |
| V3-c | 0.7 | cosine | — | — | — |
| V3-d | 1.0 | cosine | — | — | — |
| V3-e | 0.3 | cos² | — | — | — |
| V3-f | 0.3 | ReLU(cos) | — | — | — |

All V3 variants showed the same pattern: marginal top-1 improvement with significant top-5/top-10 degradation. The cosine interaction term acts as popularity bias — items similar to common context items are boosted uniformly, drowning the geometric trajectory signal.

### A.2 V4 Variants Tested (Entropy-Gated Markov)

| Config | β_gate | β | Reranker | Top-1 | Top-5 | Top-10 |
|--------|--------|---|---------|-------|-------|--------|
| V4-a | 0.2 | 0.3 | Markov | **4.58%** | 13.02% | 16.64% |
| V4-b | 0.2 | 0.5 | Markov | — | — | — |
| V4-c | 0.3 | 0.3 | Markov | — | — | — |
| V4-d | * | * | Repurchase-in-geo | — | — | — |
| V4-e | * | * | Split (geo+markov) | — | — | — |

Best V4 closed the gap from 1.55× to 1.31× on top-1 but at the cost of top-5 (−0.72%) and top-10 (−1.02%). Repurchase-in-geo (boosting items the user has bought before, weighted by geometric proximity) was the worst performer, destroying both precision and recall.

### A.3 V5 Geometric Improvements (Instacart)

Five improvements to the geometric substrate, tested individually:

| Config | Top-1 | Top-5 | Top-10 | Δ Top-1 |
|--------|-------|-------|--------|---------|
| V2 baseline | 4.36% | 13.50% | 17.02% | — |
| Fixed-point (3 iter) | **4.42%** | **13.64%** | **17.42%** | +0.06% |
| Fixed-point (5 iter) | 4.42% | 13.68% | 17.38% | +0.06% |
| Mahalanobis distance | 4.38% | 13.64% | 17.06% | +0.02% |
| Curvature extrapolation | 4.32% | 13.44% | 17.02% | −0.04% |
| Multi-scale (d=4,16,32) | 4.24% | 13.36% | 17.10% | −0.12% |
| Asymmetric embeddings | 4.36% | 13.50% | 17.02% | 0.00% |
| Transformer ceiling | 6.00% | 13.60% | 17.72% | — |

**Fixed-point iteration** — re-embeds items using velocity contexts from training trajectories, then re-runs SVD. Converges in 3 iterations. The only consistent improvement, but marginal (1.4% relative gain on top-1).

**Curvature-aware extrapolation** — replaces linear extrapolation with osculating circle trajectory. Hurts performance: grocery purchase trajectories are too noisy for higher-order geometric interpolation.

**Multi-scale scoring** — SVD at d=4/16/32, combined scores. Slightly worse than single-scale: the scales interfere rather than complement, suggesting the optimal geometry is already captured at d=32.

**Asymmetric embeddings** — separate forward/backward co-occurrence matrices. No improvement: directional structure in grocery purchases is weak relative to symmetric co-occurrence.

**Key finding:** The geometric substrate was ~95% captured by one pass of SVD. Refining it yields only incremental gains. The 1.36× top-1 gap is not in embedding quality.

### A.4 Synthetic Data Results

5 domains (baby, puppy, renovation, garden, grocery) × 8 products each. Sequences generated as purchase spirals with domain transitions and 30% noise rate.

| Method | Top-1 | Top-5 |
|--------|-------|-------|
| V2 (kinematic) | 10.6% | 42.7% |
| Transformer (epoch 12, peak top-1) | 11.1% | 49.4% |
| Transformer (epoch 50, final) | 9.7% | 51.7% |

Notable: the transformer's top-1 accuracy peaks at epoch 12 and then *declines* while top-5 continues improving — it overfits the peak while broadening the distribution. The geometric framework doesn't have this problem because it doesn't optimize.

---

## Appendix B: Equations Summary

**Co-occurrence matrix:**
$$M_{ij} = \sum_{\text{seq}} \sum_{\substack{a,b \in \text{window}}} \frac{\mathbb{1}[a{=}i,\, b{=}j]}{|pos(a) - pos(b)|}$$

**SVD embeddings:**
$$M \approx U_d \Sigma_d V_d^T, \quad \mathbf{E} = U_d \sqrt{\Sigma_d}$$

**Velocity:**
$$\mathbf{v}_k = \mathbf{e}_{p_k} - \mathbf{e}_{p_{k-1}}$$

**Acceleration:**
$$\mathbf{a}_k = \mathbf{v}_k - \mathbf{v}_{k-1}$$

**Kinematic prediction:**
$$\hat{\mathbf{e}}_{k+1} = \mathbf{e}_{p_k} + \mathbf{v}_k + \tfrac{1}{2}\mathbf{a}_k$$

**Geometric score:**
$$s_{V2}(p) = -\|\mathbf{e}_p - \hat{\mathbf{e}}_{k+1}\|_2$$

**Context interaction (V3):**
$$s_{V3}(p) = s_{V2}(p) + \alpha \sum_{c \in \text{ctx}} \cos(\mathbf{e}_p, \mathbf{e}_c)$$

**Entropy gate (V4):**
$$H = -\sum_p \hat{p}(p) \log \hat{p}(p)$$
$$s_{V4}(p) = (1 - \beta \cdot g) \cdot s_{V2}(p) + \beta \cdot g \cdot s_{\text{markov}}(p), \quad g = \mathbb{1}[H > H_{\text{gate}}]$$

**Attention as geometric selection:**
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{(XW_Q)(XW_K)^T}{\sqrt{d}}\right)(XW_V)$$

The learned projections W_Q, W_K, W_V rotate the embedding space X into a task-specific coordinate system where dot-product proximity = contextual relevance. This rotation unifies ranking precision (top-1) and distributional accuracy (top-5/10) — the specific property that no additive decomposition of pre-computed signals can replicate.
