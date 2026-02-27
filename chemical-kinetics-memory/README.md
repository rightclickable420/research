# Chemical Kinetics as a Framework for Multi-Agent Memory Management

**Gill, E. & Ash, K. (2026)**

Replacing arbitrary agent memory constants (promotion thresholds, decay rates, capacity limits) with equations from chemical kinetics. At single-agent scale the framework is under-determined; at enterprise scale it uniquely addresses multi-agent dynamics no existing approach formalizes.

## Four Mappings

| Parameter | Chemistry | Derived From |
|-----------|-----------|--------------|
| Promotion threshold | Solubility product (Ksp) | Semantic distance from boot content, boot saturation, access energy |
| Decay rate | Arrhenius (Ea only) | Co-access depth (connection count). Temperature mapping dropped after review. |
| Capacity limit | Solution arithmetic | Context window − boot files − conversation margin. Just math. |
| Holistic promotion | Gibbs free energy (ΔG) | Access energy, system activity, entropy change. Requires 500+ events to validate. |

## Adversarial Review Findings

- **Ksp (strongest):** Semantically novel chunks promote faster than redundant ones. Measurable now.
- **Arrhenius (partial):** Temperature mapping is backwards. Activation energy (co-access depth) is useful.
- **Coupled oscillation risk:** Promotion → saturation → threshold increase → stagnation → batch promotion. Needs simulation.
- **75 events is not statistical mechanics.** Hand-tuned constants likely outperform at this scale.

## Enterprise Scale (where it gets interesting)

Three multi-agent dynamics emerge that the chemical framework addresses:

1. **Phase separation** — organizational domains are immiscible. Ksp must be scoped per domain.
2. **Diffusion** (Fick's Laws) — knowledge propagation follows concentration gradients across agent boundaries.
3. **Reaction kinetics** — one agent's memory restructuring cascades through shared search results.

## Files

```
paper.md    # Full paper with mappings, review, enterprise analysis
paper.pdf   # PDF version
```

This paper is theoretical — it presents the framework, the review, and the scaling analysis. Implementation depends on accumulating sufficient access data (~500+ events for Gibbs validation).
