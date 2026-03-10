# Crystallizing Attention: Natural Self-Partitioning of Attention Heads During Transformer Training

**Ethan Gill¹, Kevin Ash¹**

¹ Independent

## Abstract

We report that transformer attention heads naturally partition into two populations during standard training: *crystallized* heads whose attention entropy stabilizes early, and *fluid* heads that maintain high entropy throughout. This self-partitioning occurs without any architectural modification or training intervention—it is an intrinsic property of transformer learning dynamics. On character-level Shakespeare, approximately 43% of heads crystallize by step 1000, with input and middle layers crystallizing first (L0 > L2 > L3) while the second layer (L1) and output layer remain almost entirely fluid. The crystallization plateau and val loss (1.4580) match or exceed standard training baselines, confirming this is a natural convergence pattern, not a pathology.

We additionally show that entropy-gated gradient scaling—an intervention that modulates learning rates based on head entropy—reshapes this natural topology: suppressing input-layer crystallization and promoting middle-layer crystallization, producing a "middle-out" ordering absent in unmodified training. This redistribution slightly degrades validation loss (1.5006 vs 1.4580), suggesting the intervention interferes with the network's preferred specialization order. Across two datasets—Shakespeare (43% crystallization) and Python source code (42%)—the crystallization fraction differs while reflecting each dataset's *contextual opacity*: the proportion of patterns the model must memorize rather than derive from context. Passive entropy tracking provides this diagnostic at zero cost to model performance, functioning as a lightweight spectroscopy of dataset structure.

## 1. Introduction

Transformer attention heads are not interchangeable. Prior work has shown that heads specialize into distinct functional roles (Voita et al., 2019), that many can be pruned without degrading performance (Michel et al., 2019), and that attention entropy dynamics relate to training stability (Zhai et al., 2023). These observations suggest that heads undergo qualitatively different learning trajectories—but the nature and timing of this differentiation during training has not been characterized.

We present evidence that attention heads naturally separate into two populations during standard training. Some heads converge to stable, low-entropy attention patterns within the first few hundred steps—we call these *crystallized*. Others maintain variable, high-entropy patterns throughout training—these remain *fluid*. This partition emerges spontaneously, without modification to the architecture, loss function, or optimizer.

Our finding arose indirectly. We initially developed an entropy-gated gradient scaling mechanism that reduced learning rates for low-entropy heads, hypothesizing this would improve training by preventing gradient waste on converged heads. Across six mechanistic variants, we observed a consistent pattern: approximately 47% of heads crystallized, with middle layers leading—a "middle-out" ordering we initially reported as a natural property of transformer learning.

A subsequent control experiment—standard training with passive entropy tracking but no gradient modification—revealed that the middle-out ordering was partially an artifact of the gating mechanism. The natural ordering is input-first: the layer closest to the embedding (L0) crystallizes fastest, followed by middle layers (L2, L3), with the second layer (L1) and output layer remaining fluid. The gating mechanism had been suppressing L0 crystallization and redistributing it to L1 and middle layers, creating the orderly middle-out pattern we initially observed.

This finding illustrates a general principle: measurement instruments can shape the phenomena they observe. The gated variants are not failures—they reveal how adaptive learning rate mechanisms redistribute specialization pressure across layers. But the natural phenomenon—spontaneous self-partitioning without intervention—is the primary finding.

The crystallization fraction, measured passively, constitutes a zero-cost diagnostic of training dynamics and dataset structure. Different datasets produce different crystallization profiles: Shakespeare (43%) and Python (42%) crystallize to similar levels but with different dynamics, reflecting differences in *contextual opacity*—how much of each dataset's structure exceeds the model's ability to derive from local context.

## 2. Method

### 2.1 Passive Entropy Tracking

For each attention head $h$ in layer $l$, we compute the normalized attention entropy at each position during the forward pass:

$$H_{l,h}(t) = -\sum_j a_{l,h,t,j} \log a_{l,h,t,j}$$

$$\hat{H}_{l,h}(t) = \frac{H_{l,h}(t)}{\log(t+1)}$$

where $a_{l,h,t,j}$ are the attention weights and $\log(t+1)$ normalizes by the maximum entropy achievable at position $t$ under the causal mask.

We maintain a running EMA of the mean normalized entropy per head, updated every training batch:

$$\bar{H}_{l,h} \leftarrow \alpha_{l,h} \cdot \bar{H}_{l,h} + (1 - \alpha_{l,h}) \cdot \mathbb{E}[\hat{H}_{l,h}]$$

The decay rate $\alpha_{l,h}$ adapts per head based on entropy variance:

$$\alpha_{l,h} = 0.95 + 0.045 \cdot \sigma\left(50 \cdot \text{Var}_{l,h}\right)$$

High-variance heads (entropy still changing) get slow decay ($\alpha \approx 0.995$), preserving sensitivity. Low-variance heads (entropy stabilized) get fast decay ($\alpha \approx 0.95$), confirming their crystallized state.

The crystallization fraction for head $(l, h)$ is:

$$c_{l,h} = \text{clamp}\left(1 - \frac{\bar{H}_{l,h}}{\max(\bar{H}_{l,h}^{\text{obs}}, \epsilon)}, 0, 1\right)$$

where $\bar{H}_{l,h}^{\text{obs}}$ is the running maximum observed entropy. A fully crystallized head ($c = 1$) has zero entropy relative to its observed range. The mean $\bar{c}$ across all heads serves as the system-level crystallization metric.

Critically, this tracking modifies no gradients and introduces no learnable parameters. The entropy computation reuses attention weights already computed in the forward pass. The only overhead is the EMA update per head per batch—negligible relative to the attention computation itself.

### 2.2 Entropy-Gated Gradient Scaling (Intervention)

To investigate the effect of actively responding to crystallization, we additionally tested a gradient-scaling intervention. After `loss.backward()` and before `optimizer.step()`, each head's gradient is scaled by its entropy ratio:

$$\nabla_{\theta_{l,h}} \leftarrow \nabla_{\theta_{l,h}} \cdot \frac{\bar{H}_{l,h}}{\max(\bar{H}_{l,h}^{\text{obs}}, \epsilon)}$$

Crystallized heads (low entropy ratio) receive near-zero gradient. Fluid heads receive full gradient. This is intended to prevent gradient waste on converged heads—but as our results show, it also reshapes the crystallization topology in ways that slightly degrade performance.

We tested six variants of this intervention (V1–V5 and phase-shift), differing in how aggressively they gate gradients, whether they include momentum or velocity terms, and whether they lock crystallization fractions at a detected plateau. Details of each variant are provided in Appendix A.

### 2.3 Phase-Shift Training (Intervention Variant)

The phase-shift variant uses crystallization as a diagnostic trigger: Phase 1 runs standard crystallizing attention until the crystal plateau is detected (rolling window of 5 measurements with max-min < 0.02), then Phase 2 locks the entropy EMA and applies differentiated treatment—gradient redistribution scaled by $(1 - c_{l,h})$ with budget conservation, and proportional dropout (near-zero for crystallized heads, elevated for fluid heads).

## 3. Experiments

### 3.1 Setup

All experiments use identical architecture for fair comparison:
- **Model**: GPT-2 style, 6 layers, 6 heads, 384 embedding dimension (10.67M parameters)
- **Training**: AdamW, LR 1e-3 → 1e-4 cosine decay, 5000 max steps, batch size 64, block size 256, dropout 0.2
- **Hardware**: Apple M-series (MPS backend, nondeterministic)
- **Evaluation**: validation loss every 50 steps (200 iterations per eval)

**Datasets**:
- **Shakespeare** (character-level): 1.1M characters, 65-character vocabulary. Early Modern English verse and dialogue.
- **Python** (character-level): 1.2M characters, 96-character vocabulary. Python standard library source code (42 files).

### 3.2 Natural Crystallization (Ungated Control)

The primary experiment: standard transformer training with passive entropy tracking and no gradient modification.

**Crystallization dynamics**: Crystal fraction rises from 0% to ~43% over the first 1000 steps, then plateaus. Growth decelerates from ~5%/50 steps (early) to <0.5%/50 steps by step 1000. The plateau is stable—crystal fraction does not change meaningfully from step 1000 to step 1700 (end of run).

**Layer ordering** (final, stable since step 800):

| Layer | Crystal % | Role |
|:------|:---------:|:-----|
| L0 | 64% | Input — crystallizes fastest and highest |
| L2 | 56% | Middle — early crystallizer |
| L3 | 55% | Middle — early crystallizer |
| L4 | 47% | Middle — moderate |
| L5 | 27% | Output — mostly fluid |
| L1 | 10% | Second layer — almost entirely fluid |

**Ordering evolution**: L2 and L3 led crystallization from steps 100–800, consistent with middle layers having the highest entropy variance early in training. At step 800, L0 overtook the middle layers and continued climbing. The final ordering [L0 > L2 > L3 > L4 > L5 > L1] has been stable since step 800.

**Validation loss**: Best val loss 1.4580 at step 1550, comparable to the standard baseline (1.4664 at step 1750). The passive tracking introduces no measurable performance cost.

**Interpretation**: The input layer (L0) naturally learns fixed patterns first—character embeddings and positional regularities represent the most consistent signal in the data. Middle layers settle next into stable abstract patterns. The second layer (L1) and output layer (L5) remain fluid, suggesting they perform context-dependent computation that resists convergence. This input-first ordering aligns with intuition about gradient signal strength: the most consistent patterns produce the strongest and most stable gradients.

### 3.3 Effect of Entropy-Gated Intervention

The same architecture with entropy-gated gradient scaling active:

| Layer | Ungated (natural) | V1 Gated | Difference |
|:------|:-----------------:|:--------:|:----------:|
| L0 | 64% (1st) | 28% (6th) | −36pp — gating suppressed |
| L2 | 56% (2nd) | 47% (3rd) | −9pp |
| L3 | 55% (3rd) | 69% (1st) | +14pp — gating promoted |
| L4 | 47% (4th) | 61% (2nd) | +14pp — gating promoted |
| L5 | 27% (5th) | 39% (5th) | +12pp |
| L1 | 10% (6th) | 40% (4th) | +30pp — gating massively promoted |

The gating mechanism inverts the periphery ordering: L0 drops from 1st to 6th, L1 rises from 6th to 4th. The middle layers (L2/L3) remain active in both conditions, but the gating promotes L3 and L4 above L0.

**Mechanism**: Entropy-gated gradient scaling reduces learning rates for heads with low entropy. L0 crystallizes naturally first → gating throttles L0's gradient → frees gradient budget for other layers → L1, L3, L4 receive more gradient and crystallize more. The gating acts as an *implicit curriculum*, imposing a learning order by slowing fast learners.

**Performance impact**: This redistribution slightly degrades validation loss:

| Variant | Best Val Loss | Best Step | Crystal % |
|:--------|:------------:|:---------:|:---------:|
| **Ungated (natural)** | **1.4580** | **1550** | **43%** |
| Standard (no tracking) | 1.4664 | 1750 | — |
| Phase Shift | 1.4788 | 2050 | 45% |
| V1 Gated | 1.5006 | 1650 | 47% |
| V3 Momentum | 1.5070 | 1750 | 73% |
| V2 Variance | 1.5127 | 1300 | 59% |

Every gated variant performs worse than both ungated tracking and the standard baseline. The natural crystallization topology—input-first—is what the network prefers. Forcing a different topology via gradient intervention costs performance.

### 3.4 Forced Crystallization and Melting

Variant V3 (momentum-based) pushed crystallization to 73%—well above the natural 43% ceiling. This confirms the gating can force heads to crystallize that would not do so naturally. However, in subsequent training steps, some forcibly crystallized heads *melted back* to fluid states, indicating the network actively resists premature crystallization of heads it needs for ongoing computation.

This melting phenomenon, combined with the val loss degradation, suggests the natural crystallization ceiling represents a genuine equilibrium: the network has crystallized everything it can afford to and needs the remaining fluid heads for context-dependent processing.

### 3.5 Cross-Dataset Crystallization

| Property | Shakespeare | Python |
|:---------|:-----------:|:------:|
| Crystal plateau | 43%* | 42% |
| Plateau step | ~1000 | ~600 |
| Best val loss | 1.46 | 1.18 |
| Learning velocity | Baseline | ~2× faster |
| Vocab size | 65 | 96 |

*Ungated measurement. The gated Shakespeare measurement (47%) is slightly elevated due to the gating's redistribution effect.

**Contextual opacity**: Python crystallizes slightly less than Shakespeare despite having more rigid syntax. Python's structural patterns (indentation, brackets, keywords) are highly derivable from local context—the model can compute them rather than memorize them. Shakespeare's patterns (meter, archaic vocabulary, character speech) are more opaque. The crystallization fraction measures this gap: how much of the data's structure exceeds the model's contextual capacity.

**Confound**: The vocabulary size difference (96 vs 65) may account for part of the crystallization gap. Controlled experiments with matched vocabularies are needed.

### 3.6 Checkpoint Stability

Crystallized models exhibit wider checkpoint windows:
- **Standard**: Best val loss 1.4664 at step 1750. Overfits to ~1.68 by step 5000.
- **Ungated tracking**: Best val loss 1.4580 at step 1550. Similar overfitting trajectory.
- **Phase Shift**: Best val loss 1.4788 at step 2050. Val loss within 0.01 of best for steps 1600–2200 (600-step window).

The crystal plateau provides an internally-derived signal for when to begin saving checkpoints, independent of validation loss. When crystal fraction stops growing, the model's internal structure has stabilized—checkpoints taken in this region are consistently near-optimal.

## 4. Related Work

**Attention entropy in training**: Zhai et al. (2023) track per-head attention entropy and identify "entropy collapse" as a training instability mode, proposing σReparam as prevention. Our work shows that entropy *reduction* in a subset of heads is not collapse but natural convergence—a healthy sign of specialization.

**Head pruning**: Michel et al. (2019) show many heads can be removed post-training without degrading performance. Our crystallization measurement identifies *which* heads have converged during training, potentially providing a principled basis for pruning decisions. The ~43% natural crystallization rate is consistent with their finding that a large fraction of heads are removable.

**Staged learning**: Yüksel et al. (2026) show transformers learn incrementally, transitioning from competitive to cooperative dynamics. Their cooperative phase corresponds to our crystallization—heads settling into specialized roles. Our contribution adds quantitative measurement of the transition and the finding that it follows a spatial ordering across layers.

**Per-parameter adaptation**: Adam (Kingma & Ba, 2015) adapts learning rates per-parameter via gradient moments. Our entropy-gated intervention operates at head granularity using forward-pass entropy. The finding that this intervention degrades performance suggests that per-parameter adaptation (Adam) already handles specialization better than per-head entropy gating.

**Dataset cartography**: Swayamdipta et al. (2020) use training dynamics to characterize per-example difficulty. Our spectroscopy operates at a different level—characterizing dataset structure through per-head internal behavior rather than per-example model confidence.

## 5. Discussion

### 5.1 Crystallization as Natural Phenomenon

The central finding is that attention head crystallization is intrinsic to transformer training. It does not require entropy gating, gradient modification, or any architectural change. A standard transformer, trained with standard optimization, will partition approximately 43% of its attention heads into fixed-pattern specialists within the first thousand steps.

This self-partitioning likely reflects an efficient allocation strategy: once a head has learned a reliable pattern (positional attention, syntactic template, character co-occurrence), there is no benefit to continued gradient updates—and potential harm from overfitting. The network naturally reaches this state through standard gradient dynamics.

### 5.2 Observer Effects in Training Interventions

Our progression from gated to ungated experiments illustrates a methodological lesson. The entropy-gated mechanism was designed to *reveal* crystallization dynamics, but it also *shaped* them—suppressing input-layer crystallization and promoting middle-layer crystallization. The resulting "middle-out" pattern appeared to be a discovery about transformer learning but was partially an artifact of the measurement intervention.

This is analogous to the observer effect in physics: measurement can disturb the system being measured. Training interventions that respond to model internals (adaptive learning rates based on entropy, gradient, or other signals) inevitably alter the dynamics they track. Researchers using such interventions should include unmodified controls to distinguish natural phenomena from intervention artifacts.

### 5.3 Spectroscopy: What Survives

The aggregate crystallization fraction remains a valid and zero-cost diagnostic:
- **Dataset characterization**: Different datasets produce different crystal fractions, reflecting contextual opacity.
- **Training monitoring**: The crystal plateau provides an internally-derived checkpointing signal.
- **Model capacity**: High crystallization may indicate the model is too small for the data's complexity (memorizing patterns it should be computing).

The per-layer ordering, however, is sensitive to training details and intervention. It should not be interpreted as a fixed property of transformer architecture without careful controls.

### 5.4 Implications for Adaptive Learning Rate Research

The finding that entropy-gated gradient scaling *degrades* performance is informative. The mechanism's intuition—"stop wasting gradient on converged heads"—seems sound. But the network's preferred crystallization topology (input-first) reflects an efficient gradient flow that the gating disrupts. Adam's per-parameter adaptation already handles the gradient allocation problem; adding per-head entropy gating introduces a competing signal that fights the optimizer.

This suggests caution for training interventions based on internal model diagnostics. The model may already be optimizing what the intervention targets, and adding explicit control can interfere with implicit optimization.

## 6. Limitations

- **Scale**: All experiments use a 10.67M parameter model. Whether crystallization dynamics hold at GPT-2 (124M) or larger scales is unknown—this is the critical next experiment.
- **Two datasets**: Shakespeare and Python are insufficient to fully characterize the spectroscopy space. Additional datasets are needed.
- **Single runs**: MPS backend introduces nondeterminism. All results are from single runs. Replication across seeds is needed to confirm the crystallization ceiling and ordering.
- **Vocabulary confound**: The crystallization difference between Shakespeare (65-char vocab) and Python (96-char vocab) may partially reflect vocabulary size.
- **Architecture**: Only decoder-only GPT with standard multi-head attention tested. Encoder-only, encoder-decoder, and multi-query attention architectures may exhibit different crystallization dynamics.
- **Val loss gap**: The 0.008 difference between ungated (1.4580) and standard (1.4664) is within noise for a single MPS run. The claim is parity, not improvement.

## 7. Future Work

- **Scaling**: Crystallization dynamics at GPT-2 (124M) with OpenWebText. Does the natural ceiling change with model size? Does input-first ordering hold with more layers?
- **Replication**: Multiple seeds on the same setup to establish confidence intervals for crystal fraction, ordering, and val loss.
- **Dataset sweep**: Shakespeare, Python, conversational English, formal proofs, random sequences, DNA. Map the spectroscopy space.
- **Vocabulary control**: Match vocabulary sizes across datasets to isolate the contextual opacity signal.
- **Fisher information**: Architecture-agnostic crystallization measurement via per-parameter Fisher information. Would enable comparison across transformers, LSTMs, and other architectures.
- **Inference efficiency**: Crystallized heads produce near-deterministic attention. Can they be replaced with cached static patterns for proportional compute savings?
- **Layer-wise gradient analysis**: Compare per-layer gradient norms between gated and ungated training to characterize how gating redistributes gradient budget.

## 8. Conclusion

Transformer attention heads naturally self-partition during training into crystallized and fluid populations. This is not an artifact of any training intervention—it occurs in standard training with standard optimization. Approximately 43% of heads crystallize on character-level Shakespeare, with input layers crystallizing first, middle layers following, and the second and output layers remaining fluid.

Entropy-gated gradient scaling, initially developed to leverage this phenomenon, actually reshapes it: redistributing crystallization from input to middle layers and slightly degrading validation loss. The intervention produces an orderly "middle-out" pattern that is partially artificial—a cautionary example of measurement instruments shaping their observations.

The crystallization fraction, measured passively through entropy tracking, serves as a zero-cost diagnostic of training dynamics and dataset structure. It requires no architectural modification, introduces no performance cost, and provides information invisible to standard metrics. The primary contribution is the observation, validated by the ungated control: transformers crystallize, and watching them do so is both free and informative.

## References

- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.
- Michel, P., Levy, O., & Neubig, G. (2019). Are Sixteen Heads Really Better than One? NeurIPS.
- Swayamdipta, S., Schwartz, R., Lourie, N., Wang, Y., Hajishirzi, H., Smith, N. A., & Choi, Y. (2020). Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics. EMNLP.
- Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). Analyzing Multi-Head Self-Attention. ACL.
- Zhai, S., Likhomanenko, T., et al. (2023). Stabilizing Transformer Training by Preventing Attention Entropy Collapse. ICML.
- Choi, M., et al. (2026). Entropy Meets Importance: A Unified Head Importance-Entropy Score for Stable and Efficient Transformer Pruning. arXiv:2510.13832.
- Yüksel, O. K., et al. (2026). Incremental Learning of Sparse Attention Patterns in Transformers. arXiv:2602.19143.

## Appendix A: Gated Variant Details

Six variants of the entropy-gated intervention were tested on Shakespeare:

| Variant | Best Val Loss | Best Step | Crystal % | Mechanism |
|:--------|:------------:|:---------:|:---------:|:----------|
| V1 Natural | 1.5006 | 1650 | 47% | Entropy-ratio gradient scaling |
| V2 Variance | 1.5127 | 1300 | 59% | Fixed variance threshold |
| V3 Momentum | 1.5070 | 1750 | 73% | Momentum-based crystallization pressure |
| V4 | ~1.50 | ~1650 | 47% | Self-calibrating decay |
| V5 | ~1.50 | ~1650 | 47% | Velocity-aware gating |
| Phase Shift | 1.4788 | 2050 | 45% | Two-phase: detect plateau, then redistribute |

All gated variants share the same entropy tracking (Section 2.1) but differ in how they translate entropy into gradient scaling. Key observations:

1. **Natural ceiling**: V1, V4, and V5 converge on ~47% crystal without external pressure—close to the ungated 43%.
2. **Forced crystallization**: V3's momentum pushes to 73%, but heads melt back and val loss does not improve.
3. **Phase shift**: Best gated variant (1.4788), but still worse than ungated (1.4580) and standard (1.4664).
4. **Consistent finding**: No gated variant outperforms unmodified training.

## Appendix B: Reproducibility

All code is based on Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT). Modified files:
- `model.py`: Attention modes including passive tracking (`ungated_track`) and six gated variants
- `train.py`: Crystallization logging, phase-shift detection, gradient scaling hooks
- `config/`: Per-variant and per-dataset configuration files
- `data/python_char/prepare.py`: Python stdlib character-level dataset preparation

Key implementation details:
- Entropy normalization: per-position maximum $\log(t+1)$ under causal mask
- EMA initialization: `entropy_ema = 1.0`, `entropy_observed_max = 0.1` per head
- EMA update: every training batch (not just at eval intervals—matching measurement cadence is critical)
- Adaptive decay: per-head, variance-driven (Section 2.1)
- No additional hyperparameters beyond standard training configuration

## Appendix C: Dynamic Attention Blending

In the gated variants, the forward pass optionally blends each head's output between the raw query and the attention-weighted value:

$$y_{l,h} = \alpha \cdot q + (1 - \alpha) \cdot \text{Attn}(q, k, v)$$

where $\alpha = \sigma(s \cdot \hat{H} + b)$ with learned scale $s$ and bias $b$. This is part of the intervention, not the passive tracking. It is included for completeness but does not affect the primary (ungated) results.
