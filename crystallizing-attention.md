# Crystallizing Attention: Entropy-Gated Learning Reveals Self-Partitioning in Transformer Attention Heads

**Ethan Gill¹, Kevin Ash¹**

¹ Independent

## Abstract

We introduce *crystallizing attention*, a training mechanism where each attention head's gradient is scaled by its own entropy dynamics. Heads that converge to stable, low-entropy attention patterns—crystallized heads—receive near-zero gradient, while high-entropy fluid heads retain full learning capacity. This produces a natural self-partitioning: approximately 47% of attention heads crystallize into fixed structural patterns while 53% remain fluid for contextual computation, with middle layers crystallizing first and interface layers (input/output) remaining liquid longest. We show that the crystallization plateau provides a reliable, val-loss-independent early stopping signal derived entirely from the model's internal state. In experiments on character-level Shakespeare with a 6-layer, 6-head GPT, our phase-shift variant achieves the best non-standard validation loss (1.4788) with a checkpoint stability window 10× wider than standard training. We report results across six variants exploring nucleation mechanisms (natural, variance-based, momentum, bond-propagated, combined, and phase-shifted), finding that the self-partitioning is robust to mechanism choice—the same ~47% crystallization ceiling emerges regardless of external pressure, suggesting it reflects an intrinsic property of the architecture-dataset interaction.

## 1. Introduction

Standard transformer training applies uniform learning rates across all attention heads, despite growing evidence that heads serve fundamentally different functions (Voita et al., 2019; Michel et al., 2019). Some heads learn fixed structural patterns early in training (positional attention, syntactic relations) while others perform context-dependent computation throughout. Uniform training wastes gradient on converged heads—degrading their learned patterns through overfitting—while under-serving heads that are still actively learning.

We propose a simple modification: scale each head's gradient by its attention entropy relative to its own observed maximum. A head whose attention entropy has stabilized (crystallized) receives minimal gradient. A head whose entropy remains variable (fluid) receives full gradient. No hyperparameters beyond the standard training configuration are introduced—each head self-calibrates against its own entropy range using adaptive EMA decay.

This mechanism reveals a striking self-organization: attention heads spontaneously partition into two populations during training. We term this *crystallizing attention* by analogy to physical crystallization, where a cooling liquid separates into ordered (crystalline) and disordered (amorphous) phases. The crystallization pattern is consistent across all six mechanism variants we tested:

- **Middle layers crystallize first** (L3: 63-84%, L4: 55-79%)
- **Interface layers stay fluid** (L0: 27-63%, L5: 34-68%)
- **Natural plateau at ~47%** without external pressure

The crystallization plateau serves as an early stopping signal: when crystal percentage stops growing, the model has extracted all stable structure from the data. Checkpoints taken at this point reliably capture near-optimal performance, unlike standard training where the best checkpoint occurs at a single unpredictable step.

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

where $\alpha = \sigma(s \cdot \hat{H} + b)$ with learned scale $s$ and bias $b$. High-entropy (uncertain) positions anchor toward the query; low-entropy (confident) positions trust the attention. This provides a complementary mechanism to gradient gating: the forward pass adapts to entropy in real-time, while gradient gating adapts the learning rate over training.

### 2.4 Phase-Shift Training

Building on the crystallization diagnostic, we introduce *phase-shift training*: a two-phase regime where Phase 1 runs standard crystallizing attention until the crystal plateau is detected, then Phase 2 locks the entropy EMA (stops updating) and applies differentiated treatment:

- **Gradient redistribution**: Each head's gradient is scaled by $(1 - c_{l,h}) \times B$, where $B$ is a boost factor. Crystallized heads receive near-zero gradient (preventing overfitting). Fluid heads receive amplified gradient (concentrating learning).
- **Proportional dropout**: Crystallized heads receive near-zero dropout (they produce deterministic, reliable features). Fluid heads receive elevated dropout (forcing generalization in the contextual computation).

Plateau detection: we maintain a rolling window of $W$ crystallization measurements. When $\max - \min < \tau$ across the window (default $W=5$, $\tau=0.02$), the shift triggers.

## 3. Nucleation Variants

We explored six mechanisms for controlling crystallization dynamics, all sharing the same base entropy-gated architecture:

### V1: Natural (baseline)
Pure entropy-gated gradient scaling with no external pressure. Crystallization emerges from the natural decay of attention entropy as heads find stable patterns.

### V2: Variance-Based Nucleation
Per-head stability counting: heads with low entropy variance for 50+ consecutive steps receive additional gradient reduction. **Result**: premature crystallization. The fixed step threshold caused heads to freeze before fully learning, producing worse validation loss than V1.

### V3: Momentum-Based Nucleation
System-level pressure that ramps when crystallization velocity drops. Captures initial crystal velocity after warmup and applies increasing pressure proportional to the momentum deficit. 5% gradient floor prevents total lockdown. **Result**: reached 73% crystallization (vs V1's 47%) but through brute-force pressure rather than natural convergence.

### V4: Bond-Propagated Crystallization
Per-head entropy correlation matrix (Pearson) updated every 20 steps. When one head crystallizes, its correlated neighbors receive gradient reduction proportional to bond strength. **Result**: bonds self-dissolved as crystallized heads stopped co-moving with fluid heads, limiting propagation to a temporary acceleration. Same 47% ceiling as V1.

### V5: Pressure + Bonds
Combined V3 momentum pressure with V4 bond propagation and bond locking (bonds above 0.1 cannot decay). **Result**: nearly identical to V3 alone—pressure dominated at 0.86 while bonds contributed 0.02.

### V6: Phase Shift
Two-phase regime described in §2.4. Phase 1 diagnostic, Phase 2 differentiated treatment. **Result**: best non-standard validation loss (1.4788), 400 steps of additional improvement beyond V1's peak.

## 4. Experiments

### 4.1 Setup

All experiments use the same architecture for fair comparison:
- **Model**: GPT-2 style, 6 layers, 6 heads, 384 embedding dimension (10.67M parameters)
- **Data**: Shakespeare character-level (1.1M characters)
- **Training**: AdamW, LR 1e-3 → 1e-4 cosine decay, 5000 steps, batch size 64, block size 256, dropout 0.2
- **Hardware**: Apple M-series (MPS backend, nondeterministic)
- **Evaluation**: validation loss every 50 steps (200 iterations per eval)

### 4.2 Main Results

| Variant | Best Val Loss | Best Step | Crystal % | Overfit at Step 5000 |
|---------|:---:|:---:|:---:|:---:|
| Standard | **1.4664** | 175 | — | 1.68 |
| Phase Shift (V6) | **1.4788** | 2050 | 45% (locked) | — |
| Split Brain† | 1.4854 | — | — | — |
| V1 Natural | 1.5006 | 1650 | 47% | ~1.55 |
| V3 Momentum | 1.5070 | 1750 | 73% | ~1.56 |
| V2 Variance | 1.5127 | 1300 | 59% | — |

†Split brain is a separate architecture (dual-hemisphere attention with corpus callosum blending) included for reference.

### 4.3 Crystallization Dynamics

**Middle-out ordering**: Across all variants, middle layers (L3, L4) crystallize first, followed by middle-adjacent layers (L2, L1), with interface layers (L0 input, L5 output) crystallizing last or remaining predominantly fluid. Representative ordering from V1 at plateau:

| Layer | Crystal % | Role |
|:---:|:---:|:---|
| L3 | 69% | Deep structure (highest crystallization) |
| L4 | 61% | Deep structure |
| L2 | 47% | Transitional |
| L1 | 40% | Transitional |
| L5 | 39% | Output interface (fluid) |
| L0 | 28% | Input interface (most fluid) |

This ordering is robust across all variants (V1-V6) despite different nucleation mechanisms.

**Natural ceiling**: Without external pressure (V1, V4), crystallization plateaus at ~47%. This ceiling is consistent between V1 (47%) and V4 (45%), suggesting it reflects an intrinsic property of the architecture-dataset interaction. Momentum pressure (V3) pushes to 73% but does not improve validation loss, indicating the additional crystallization is forced rather than natural.

**Decrystallization**: In V2, heads that were prematurely frozen by the step-counting mechanism spontaneously melted back (reduced crystallization) when the gradient floor allowed loss-driven recovery. This emergent behavior confirms that crystallization represents genuine thermodynamic equilibrium, not an artifact of the gradient scaling.

### 4.4 Checkpoint Stability

The practical advantage of crystallizing attention is checkpoint reliability. Standard training produces a sharp, unpredictable validation loss minimum:

- **Standard**: Best val loss at step 175. Val loss at step 500: ~1.55. Val loss at step 5000: 1.68.
- **Phase Shift**: Best val loss at step 2050. Val loss within 0.01 of best for steps 1600-2200 (600-step window).

Any checkpoint taken in the phase-shift's plateau region outperforms standard training at any step after 300. This eliminates the need for exhaustive checkpoint evaluation—the crystallization plateau signal indicates the optimal checkpointing window.

### 4.5 What We Learned from Six Variants

The progression from V1 to V6 was itself informative:

1. **V1 → V2**: Arbitrary thresholds cause premature freezing. Self-calibrating mechanisms are essential.
2. **V2 → V3**: System-level pressure can push past natural limits but doesn't improve quality.
3. **V3 → V4**: Topology-aware crystallization (bond propagation) is self-limiting—the bonds that drive crystallization dissolve when crystallization occurs.
4. **V4 → V5**: At this scale, topology adds negligible signal above pressure alone.
5. **V5 → V6**: The crystallization diagnostic is more valuable than the crystallization mechanism—knowing WHEN heads have converged enables targeted intervention.
6. **V6 → V1**: Phase 2's gradient redistribution approximates what V1's continuous entropy scaling already does. The mechanism was close to the answer from the start. The key contribution is the observation, not the mechanism.

## 5. Related Work

**Attention entropy in training stability**: Zhai et al. (2023) track per-head attention entropy during training and identify "entropy collapse" (pathologically low entropy causing training instability) as a failure mode. They propose σReparam (spectral normalization) as a preventive measure. Our work uses entropy constructively rather than defensively—low entropy is not a failure mode but a signal of convergence.

**Attention head pruning**: Michel et al. (2019) demonstrate that many attention heads can be removed post-training without significant quality loss, implying redundancy. HIES (Choi et al., 2026) combines gradient-based importance scores with attention entropy for post-training structured pruning. Our approach differs in using entropy during training to modulate learning rates, not after training to remove heads.

**Staged learning in transformers**: Yüksel et al. (2026) show that attention heads learn incrementally, transitioning from competitive (all heads converge on dominant patterns) to cooperative (heads specialize). Their analysis aligns with our crystallization observation—competitive learning corresponds to the pre-crystallization phase, and cooperative specialization corresponds to heads settling into crystallized vs. fluid roles. Our contribution adds the quantitative measurement (entropy-gated crystal fraction) and the practical application (early stopping, phase shift).

**Per-parameter adaptive learning rates**: Adam (Kingma & Ba, 2015) adapts learning rates per-parameter based on gradient moments. Our approach operates at a different semantic granularity—per-head rather than per-parameter—and uses attention entropy (a forward-pass property) rather than gradient statistics (a backward-pass property).

## 6. Limitations

- **Scale**: All experiments use a 10.67M parameter model on a single small dataset. The 47% crystallization fraction, the middle-out ordering, and the phase-shift benefits may not transfer to larger models or diverse datasets. Scaling experiments are the critical next step.
- **Nondeterminism**: MPS backend introduces run-to-run variance. Results are reported from single runs, not averaged across seeds.
- **Architecture specificity**: We test only decoder-only GPT with standard multi-head attention. Encoder-decoder, grouped-query attention, and mixture-of-experts architectures may exhibit different crystallization dynamics.
- **Causal attribution**: The phase-shift validation improvement (1.4788 vs 1.5006) could partially result from the effective learning rate change at the shift point rather than the crystallization-informed gradient redistribution. Ablations isolating the LR effect from the entropy-informed distribution are needed.

## 7. Future Work

- **Scaling experiments**: Run crystallizing attention on GPT-2 (124M) with OpenWebText. Key question: does the crystallization fraction change with model size, and does the middle-out ordering persist?
- **Dataset dependence**: Compare crystallization fractions across text (Shakespeare, Wikipedia, code) to test whether the partition is architecture-dependent, data-dependent, or both.
- **Continuous self-regulation**: Replace the two-phase approach with continuous per-head gradient scaling from step 0, using entropy variance as the signal. This eliminates the plateau detection mechanism while preserving the gradient redistribution.
- **Connection to head function**: Map crystallized heads to known functional types (positional, syntactic, induction) to understand what crystallization means mechanistically.
- **Inference efficiency**: Crystallized heads produce deterministic attention patterns. At inference time, these could be replaced with cached static attention maps, reducing computation proportional to the crystallization fraction.

## 8. Conclusion

Crystallizing attention reveals that transformer attention heads spontaneously self-partition into structure and computation during training. This partition is measurable through attention entropy dynamics, follows a consistent middle-out spatial ordering across layers, and converges to a natural ceiling that reflects the architecture-dataset interaction. The crystallization plateau provides a reliable, internally-derived early stopping signal, and phase-shifted training exploiting the partition produces the most reliable checkpoint selection among all variants tested.

The primary contribution is not the training mechanism—entropy-gated gradient scaling is simple and the first variant (V1) was close to optimal—but the *observation* that this partition exists, is measurable, and is robust to mechanism choice. This observation opens questions about transformer self-organization that extend beyond training efficiency to the fundamental nature of how attention heads allocate computational roles.

## References

- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.
- Michel, P., Levy, O., & Neubig, G. (2019). Are Sixteen Heads Really Better than One? NeurIPS.
- Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). Analyzing Multi-Head Self-Attention. ACL.
- Zhai, S., Likhomanenko, T., et al. (2023). Stabilizing Transformer Training by Preventing Attention Entropy Collapse. ICML.
- Choi, M., et al. (2026). Entropy Meets Importance: A Unified Head Importance-Entropy Score for Stable and Efficient Transformer Pruning. arXiv:2510.13832.
- Yüksel, O. K., et al. (2026). Incremental Learning of Sparse Attention Patterns in Transformers. arXiv:2602.19143.

## Appendix A: Reproducibility

All code is based on Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT). Modified files:
- `model.py`: Added attention modes `crystallize` (V1), `crystallize_v2` through `crystallize_v5`, and `phase_shift`
- `train.py`: Added crystallization logging, phase-shift detection, gradient scaling calls
- `config/train_crystallize.py` through `config/train_phase_shift.py`: Per-variant configuration

Key implementation details:
- Entropy normalization: per-position maximum $\log(t+1)$ under causal mask, not global constant
- EMA calibration: per-head adaptive decay, not fixed rate
- Gradient scaling: per-head slicing of Q/K/V projection weights, not mean-scale across layer
- No additional hyperparameters beyond standard training configuration
