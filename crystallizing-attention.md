# Crystallizing Attention: Entropy-Gated Learning Reveals Self-Partitioning and Dataset Spectroscopy in Transformers

**Ethan Gill¹, Kevin Ash¹**

¹ Independent

## Abstract

We introduce *crystallizing attention*, a training mechanism where each attention head's gradient is scaled by its own entropy dynamics. Heads that converge to stable, low-entropy attention patterns—crystallized heads—receive near-zero gradient, while high-entropy fluid heads retain full learning capacity. This produces a natural self-partitioning: attention heads spontaneously separate into crystallized (fixed structure) and fluid (contextual computation) populations, with middle layers crystallizing first and interface layers remaining liquid. Across two datasets—character-level Shakespeare (47% crystallization) and Python source code (42% crystallization)—we find that the crystallization fraction differs between datasets while the spatial ordering (middle-out) remains constant. Contrary to our initial hypothesis, code crystallizes *less* than prose despite having more rigid syntax, because its structural patterns are more contextually derivable. This suggests crystallization measures not structural density but *contextual opacity*: the fraction of patterns the model cannot derive from context and must memorize. Combined with learning velocity (which differs 2× between datasets), crystallization dynamics constitute a two-dimensional *spectroscopy* of dataset structure, measurable in the first few hundred training steps. We additionally show that a phase-shift variant exploiting the crystallization diagnostic achieves the best non-standard validation loss (1.4788 vs standard's 1.4664) with a checkpoint stability window 10× wider than standard training.

## 1. Introduction

Standard transformer training applies uniform learning rates across all attention heads, despite growing evidence that heads serve fundamentally different functions (Voita et al., 2019; Michel et al., 2019). Some heads learn fixed structural patterns early in training (positional attention, syntactic relations) while others perform context-dependent computation throughout. Uniform training wastes gradient on converged heads—degrading their learned patterns through overfitting—while under-serving heads that are still actively learning.

We propose a simple modification: scale each head's gradient by its attention entropy relative to its own observed maximum. A head whose attention entropy has stabilized (crystallized) receives minimal gradient. A head whose entropy remains variable (fluid) receives full gradient. No hyperparameters beyond the standard training configuration are introduced—each head self-calibrates against its own entropy range using adaptive EMA decay.

This mechanism reveals a striking self-organization: attention heads spontaneously partition into two populations during training. We term this *crystallizing attention* by analogy to physical crystallization, where a cooling liquid separates into ordered (crystalline) and disordered (amorphous) phases.

Our initial experiments on Shakespeare character-level data showed 47% crystallization with consistent middle-out layer ordering across six mechanistic variants. We hypothesized that this fraction measured "structural density"—that more structured data like code would crystallize more. Testing on Python source code falsified this hypothesis: code crystallizes at 42%, five points *lower* than Shakespeare, despite having more rigid syntax.

The resolution: crystallization measures *contextual opacity*, not structural density. Python's structure is highly predictable from local context (indentation level, bracket depth, keyword patterns). Shakespeare's patterns are more opaque—the model must memorize what it cannot compute from context. The crystallization fraction quantifies this gap between data complexity and the model's contextual capacity.

This reframes crystallizing attention from a training technique to a *measurement instrument*—a spectroscopy of dataset structure that reveals properties invisible to traditional metrics like perplexity or token-level entropy.

## 2. Method

### 2.1 Entropy-Gated Gradient Scaling

For each attention head $h$ in layer $l$, we compute the normalized attention entropy at each position:

$$H_{l,h}(t) = -\sum_j a_{l,h,t,j} \log a_{l,h,t,j}$$

$$\hat{H}_{l,h}(t) = \frac{H_{l,h}(t)}{\log(t+1)}$$

where $a_{l,h,t,j}$ are the attention weights and $\log(t+1)$ is the maximum entropy achievable at position $t$ under the causal mask (the uniform distribution over $t+1$ available positions).

We maintain a running EMA of the mean normalized entropy per head:

$$\bar{H}_{l,h} \leftarrow \alpha_{l,h} \cdot \bar{H}_{l,h} + (1 - \alpha_{l,h}) \cdot \mathbb{E}[\hat{H}_{l,h}]$$

where the decay rate $\alpha_{l,h}$ is adaptive per head:

$$\alpha_{l,h} = 0.95 + 0.045 \cdot \sigma\left(50 \cdot \text{Var}_{l,h}\right)$$

High-variance heads (entropy still changing) get slow decay ($\alpha \approx 0.995$), preserving sensitivity. Low-variance heads (entropy stabilized) get fast decay ($\alpha \approx 0.95$), locking in their crystallized state.

After `loss.backward()` and before `optimizer.step()`, we scale each head's gradient:

$$\nabla_{\theta_{l,h}} \leftarrow \nabla_{\theta_{l,h}} \cdot \frac{\bar{H}_{l,h}}{\max(\bar{H}_{l,h}^{\text{obs}}, \epsilon)}$$

where $\bar{H}_{l,h}^{\text{obs}}$ is the running maximum observed entropy for that head. This normalizes each head against its own range—a head that naturally operates at low entropy is not penalized relative to a globally-defined maximum.

### 2.2 Crystallization Fraction

We define the crystallization fraction for head $(l, h)$ as:

$$c_{l,h} = 1 - \frac{\bar{H}_{l,h}}{\max(\bar{H}_{l,h}^{\text{obs}}, \epsilon)}$$

A fully crystallized head ($c = 1$) has zero entropy relative to its observed range and receives zero gradient. A fully fluid head ($c = 0$) receives full gradient. The mean crystallization across all heads, $\bar{c}$, serves as the system-level crystallization metric.

### 2.3 Dynamic Attention Blending

In addition to gradient gating, the forward pass blends each head's output between the raw query (identity/anchor) and the attention-weighted value:

$$y_{l,h} = \alpha \cdot q + (1 - \alpha) \cdot \text{Attn}(q, k, v)$$

where $\alpha = \sigma(s \cdot \hat{H} + b)$ with learned scale $s$ and bias $b$. High-entropy (uncertain) positions anchor toward the query; low-entropy (confident) positions trust the attention.

### 2.4 Phase-Shift Training

Building on the crystallization diagnostic, we introduce *phase-shift training*: a two-phase regime where Phase 1 runs standard crystallizing attention until the crystal plateau is detected, then Phase 2 locks the entropy EMA and applies differentiated treatment:

- **Gradient redistribution**: Each head's gradient is scaled by $(1 - c_{l,h}) \times B$, where $B$ is computed to conserve total gradient budget: $B = N_{\text{heads}} / \sum(1 - c_{l,h})$.
- **Proportional dropout**: Crystallized heads receive near-zero dropout (deterministic, reliable features). Fluid heads receive elevated dropout (forcing generalization).

Plateau detection: a rolling window of $W=5$ crystallization measurements triggers the shift when $\max - \min < 0.02$.

## 3. Experiments

### 3.1 Setup

All experiments use the same architecture for fair comparison:
- **Model**: GPT-2 style, 6 layers, 6 heads, 384 embedding dimension (10.67M parameters)
- **Training**: AdamW, LR 1e-3 → 1e-4 cosine decay, 5000 max steps, batch size 64, block size 256, dropout 0.2
- **Hardware**: Apple M-series (MPS backend, nondeterministic)
- **Evaluation**: validation loss every 50 steps (200 iterations per eval)

**Datasets**:
- **Shakespeare** (character-level): 1.1M characters, 65-character vocabulary. Early Modern English verse and dialogue.
- **Python** (character-level): 1.2M characters, 96-character vocabulary. Python standard library source code (42 files).

### 3.2 Cross-Dataset Crystallization

| Property | Shakespeare | Python |
|:---|:---:|:---:|
| Crystal plateau | 47% | 42% |
| Plateau step | ~1000 | ~600 |
| Best val loss | 1.50 (step 1650) | 1.18 (step 1100) |
| Learning velocity | Baseline | ~2× faster |
| Vocab size | 65 | 96 |
| Layer leader | L3 (69%) | L3 (55%) |
| Middle-out ordering | ✓ | ✓ |

**Consistent across datasets**: Middle-out spatial ordering (middle layers crystallize first, interface layers stay fluid). Both datasets show L3 leading, with L0 (input) and L5 (output) crystallizing least.

**Different across datasets**: Crystallization fraction (47% vs 42%), learning velocity (Python learns ~2× faster), and the detailed early dynamics (Python showed L4/L1 co-leading before L3 reasserted).

### 3.3 Interpreting the Difference

Our initial hypothesis—that code's rigid syntax would produce higher crystallization—was falsified. We propose instead that crystallization measures *contextual opacity*: the fraction of patterns that exceed the model's ability to derive from local context.

Python's structural patterns (indentation, brackets, keywords, type annotations) are highly predictable from their local context. A `:` after a `def` line, `self` as a first argument, matching brackets—these are derivable by fluid attention without memorization. The model learns Python fast (low val loss) while keeping most heads fluid.

Shakespeare's patterns (iambic meter, archaic vocabulary, character speech patterns) are less locally predictable. "Wherefore art thou" cannot be derived from the surrounding context the way `return self._value` can. The model must crystallize more attention patterns as fixed templates.

**Confound**: The vocabulary size difference (96 vs 65) may account for part of the crystallization gap. A larger vocabulary distributes model capacity across more character embeddings, potentially reducing capacity available for attention crystallization. Controlled experiments (matching vocabulary sizes across datasets) are needed to isolate the contextual opacity effect.

### 3.4 Two-Dimensional Spectroscopy

The combination of crystallization fraction and learning velocity constitutes a two-dimensional characterization of dataset structure:

- **Crystal fraction** (y-axis): contextual opacity. How much of the data's structure must be memorized vs. computed.
- **Learning velocity** (x-axis): total learnable structure. How quickly the model finds patterns regardless of crystallization.

| | Low Crystal | High Crystal |
|:---|:---|:---|
| **Fast Learning** | Code: lots of derivable structure | (untested: formal logic?) |
| **Slow Learning** | (untested: conversational text?) | Shakespeare: opaque patterns, slow to learn |

This two-dimensional space could serve as a *dataset fingerprint*, classifying data types by how they interact with a standard probe architecture. Different datasets would occupy different regions, measurable in the first 300-500 training steps—orders of magnitude cheaper than full training.

### 3.5 Nucleation Mechanism Variants (Shakespeare)

We explored six mechanisms for controlling crystallization dynamics:

| Variant | Best Val Loss | Best Step | Crystal % | Key Finding |
|:---|:---:|:---:|:---:|:---|
| Standard | **1.4664** | 175 | — | Sharp peak, fast overfit |
| Phase Shift | **1.4788** | 2050 | 45% locked | Best non-standard; wide checkpoint window |
| Split Brain† | 1.4854 | — | — | Hemisphere separation helps |
| V1 Natural | 1.5006 | 1650 | 47% | Natural plateau baseline |
| V3 Momentum | 1.5070 | 1750 | 73% | Forced higher crystal, no quality gain |
| V2 Variance | 1.5127 | 1300 | 59% | Premature freeze from fixed threshold |

†Split brain is a separate architecture (dual-hemisphere attention with corpus callosum) included for reference.

**Key findings across variants**:
1. The natural crystallization ceiling (~47% on Shakespeare) is robust to mechanism choice—V1, V4, and V5 all converge on the same ceiling without external pressure.
2. Momentum pressure (V3) can push past the natural ceiling (73%) but does not improve validation loss, confirming the ceiling reflects genuine convergence rather than a mechanism limitation.
3. Phase-shift training achieves the best checkpoint reliability: any checkpoint in steps 1600-2200 outperforms standard training at any step past 300.
4. Self-regulation (continuous entropy-gated scaling) achieves most of what the phase shift does, suggesting the mechanism is less important than the measurement.

### 3.6 Checkpoint Stability

The practical advantage of crystallizing attention:
- **Standard**: Best val loss at step 175. A single-step event; val loss at step 500 is already 1.55+.
- **Phase Shift**: Best val loss at step 2050. Val loss within 0.01 of best for steps 1600-2200 (600-step window).

The crystallization plateau serves as a reliable checkpointing signal: when crystal percentage stops growing, begin saving checkpoints. This eliminates the need for exhaustive evaluation or lucky early stopping.

## 4. Related Work

**Attention entropy in training**: Zhai et al. (2023) track per-head attention entropy and identify "entropy collapse" as a training instability mode, proposing σReparam as prevention. Our work uses entropy constructively—low entropy signals convergence, not failure.

**Head pruning**: Michel et al. (2019) show many heads can be removed post-training. HIES (Choi et al., 2026) combines gradient importance with entropy for post-training pruning. Our approach modulates learning rates during training using entropy, not pruning after.

**Staged learning**: Yüksel et al. (2026) show transformers learn incrementally, transitioning from competitive to cooperative dynamics. Their cooperative phase corresponds to our crystallization—heads settling into specialized roles. Our contribution adds quantitative measurement and the dataset spectroscopy application.

**Per-parameter adaptation**: Adam (Kingma & Ba, 2015) adapts learning rates per-parameter via gradient moments. Our approach operates at head granularity using forward-pass entropy, a semantically meaningful signal about what the head has learned.

## 5. Discussion: Crystallization as Spectroscopy

The most significant finding may not be the training mechanism but its use as a *measurement instrument*. The crystallization dynamics of the first few hundred training steps reveal properties of the dataset that are invisible to token-level statistics:

**What crystal fraction measures**: Not structural density or data quality, but the gap between a dataset's complexity and the model's contextual capacity. High crystallization means the model is memorizing patterns it cannot derive—a signal about the data-architecture interaction, not the data alone.

**Potential applications beyond training**:
- *Data mixture optimization*: Crystal spectra reveal which datasets provide complementary vs. redundant training signal.
- *Model capacity planning*: High crystallization suggests the model is too small for the data's contextual complexity.
- *Dataset fingerprinting*: The two-dimensional spectrum (crystal fraction × learning velocity) could classify datasets by their structural properties in seconds.
- *Curriculum design*: Order training data by crystal profile—high-crystal data first (foundational patterns), low-crystal data later (contextual reasoning).

## 6. Limitations

- **Scale**: All experiments use a 10.67M parameter model. The crystallization dynamics at GPT-2 (124M) or larger scales are unknown.
- **Two datasets**: Shakespeare and Python are insufficient to fully characterize the spectroscopy. Additional datasets (conversational text, formal logic, random sequences) are needed.
- **Vocabulary confound**: The 5-point crystallization difference between Shakespeare (65-char vocab) and Python (96-char vocab) may partially reflect vocabulary size rather than contextual opacity. Controlled experiments with matched vocabularies are needed.
- **Nondeterminism**: MPS backend introduces run-to-run variance. Results from single runs.
- **Architecture specificity**: Only decoder-only GPT with standard multi-head attention tested.

## 7. Future Work

- **Scaling**: Run crystallizing attention on GPT-2 (124M) with OpenWebText. Does the crystallization fraction change with model size?
- **Dataset sweep**: Shakespeare, Python, conversational English, formal proofs, random sequences, DNA. Map the two-dimensional spectroscopy space.
- **Vocabulary control**: Match vocabulary sizes across datasets to isolate the contextual opacity signal.
- **Rapid probing**: Can the crystal spectrum be estimated from gradient magnitudes in the first 1-10 steps, before any meaningful training occurs?
- **Inference efficiency**: Crystallized heads produce near-deterministic attention. Replace with cached static patterns for proportional compute savings.
- **Non-NLP applications**: Apply crystallization probing to sequential data beyond text (genomic sequences, financial time series, sensor data).

## 8. Conclusion

Crystallizing attention reveals that transformer attention heads spontaneously self-partition during training into fixed-structure and contextual-computation populations. This partition follows a consistent middle-out spatial ordering across layers but varies in magnitude across datasets—47% on Shakespeare, 42% on Python—measuring not structural density but *contextual opacity*: how much of the data's pattern exceeds the model's ability to derive from context.

The mechanism is simple (entropy-gated gradient scaling), the observation is robust (consistent across six mechanistic variants), and the practical benefit is clear (10× wider checkpoint stability window). But the most promising direction is the use of crystallization dynamics as a spectroscopy—a fast, cheap characterization of dataset structure through the lens of how a neural network interacts with it.

The primary contribution is the observation, not the mechanism. V1—the simplest variant—was already close to optimal. The journey through six variants taught us that the crystal fraction is a measurement, not a lever.

## References

- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.
- Michel, P., Levy, O., & Neubig, G. (2019). Are Sixteen Heads Really Better than One? NeurIPS.
- Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). Analyzing Multi-Head Self-Attention. ACL.
- Zhai, S., Likhomanenko, T., et al. (2023). Stabilizing Transformer Training by Preventing Attention Entropy Collapse. ICML.
- Choi, M., et al. (2026). Entropy Meets Importance: A Unified Head Importance-Entropy Score for Stable and Efficient Transformer Pruning. arXiv:2510.13832.
- Yüksel, O. K., et al. (2026). Incremental Learning of Sparse Attention Patterns in Transformers. arXiv:2602.19143.

## Appendix A: Reproducibility

All code is based on Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT). Modified files:
- `model.py`: Attention modes `crystallize` (V1) through `crystallize_v5` and `phase_shift`
- `train.py`: Crystallization logging, phase-shift detection, gradient scaling
- `config/`: Per-variant and per-dataset configuration files
- `data/python_char/prepare.py`: Python stdlib character-level dataset preparation

Key implementation details:
- Entropy normalization: per-position maximum $\log(t+1)$ under causal mask, not global constant
- EMA calibration: per-head adaptive decay, not fixed rate
- Phase-shift gradient scaling: per-head slicing of Q/K/V projection weights, not mean-scale across layer
- No additional hyperparameters beyond standard training configuration
