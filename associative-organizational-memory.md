# Associative Organizational Memory: Collective Compression Through Hopfield Reconstruction

**Working title:** "Individual Agents Forget, Organizations Remember: Hopfield-DCT Memory at Scale"
**Authors:** Ethan Gill & Kevin Ash
**Status:** Experimental design
**Date:** 2026-03-06

## Core Insight

DCT compresses individual memory trajectories independently. Hopfield networks reconstruct patterns from associative context. Stacked, they produce **collective compression** — individual agents can store less because the organizational memory corpus provides reconstruction context. Compression ratio scales with corpus density.

## The Stack

| Layer | Mechanism | Paper |
|-------|-----------|-------|
| Storage | DCT frequency decomposition of embedding trajectories | Paper 5 |
| Degradation | High-freq coefficients decay with time/disuse | Papers 4, 5 |
| Reconstruction | Hopfield associative recall from degraded cue | **NEW** |
| Energy landscape | Access patterns shape basin structure | Papers 2, 4 |
| Temporal signal | Cadence modulates decay and reconstruction | Paper 1 |
| Rate dynamics | Threshold kinetics for promotion/consolidation | Paper 3 |
| Evolution | Memory landscape mutates and selects over time | Paper 9 |
| Calibration | Cognitive signatures tune per-agent retrieval | Papers 7, 8 |
| Autonomy | Agents operate within the collective fabric | Paper 6 |

## Key Claims

### Claim 1: Hopfield reconstruction improves retrieval of degraded DCT memories
- Degraded memory (high-freq coefficients dropped) = partial cue
- Hopfield completes pattern from associative neighbors in embedding space
- Reconstruction quality > degraded quality (measurable via cosine similarity)

### Claim 2: Reconstruction quality scales with corpus density
- Sparse corpus (80 memories) → weak reconstruction
- Dense corpus (100K+ memories) → rich associative context → strong reconstruction
- This is a new result: compression becomes collaborative

### Claim 3: Collective compression breaks the individual floor
- DCT alone has a minimum storage cost per memory (below which quality is unacceptable)
- Hopfield + DCT lowers this floor as a function of corpus density
- Individual storage cost goes DOWN as the system grows
- Opposite of typical storage scaling

### Claim 4: This unifies papers 1-9
- Not 9 separate contributions — one system with 9 layers
- Complexity ladder: math → physics → chemistry → biology → psychology → sociology
- The Hopfield layer is the mechanism that connects individual (papers 4-5, 7-8) to collective (papers 1-3, 6, 9)

## Experimental Design

### Phase 1: Mechanism proof (small scale)
- Corpus: Kevin's 80 memory chunks (real data)
- Embeddings: 768-dim (BGE-small or similar)
- Compress: DCT at various ratios (keep top-k coefficients, k = 1..768)
- Reconstruct: Continuous Hopfield network trained on full corpus
- Measure: cosine_sim(original, reconstructed) vs cosine_sim(original, degraded)
- Expected: reconstruction > degraded, proving the mechanism

### Phase 2: Scaling curve (simulated)
- Corpus: Public embedding dataset (Wikipedia paragraphs, arxiv abstracts)
- Subsample at: 100, 1K, 10K, 100K sizes
- Same pipeline at each scale
- Plot: corpus density vs reconstruction quality at fixed compression ratio
- Plot: corpus density vs maximum achievable compression (at quality threshold)
- Expected: both curves increase with density

### Phase 3: Organizational validation (future, Walmart-scale)
- Real heterogeneous organizational memory
- Validates whether simulated scaling holds on real data
- Tests cross-domain reconstruction (supply chain memories helping logistics recall)

## Biological Parallel

This is how human memory works:
- Details fade over time (DCT degradation)
- On recall, brain reconstructs plausible details from associations (Hopfield)
- Reconstructions are shaped by everything else you know (corpus density)
- Confabulation: false but plausible memories emerge naturally
- Collective memory: organizations "remember" things no individual does

## Architecture

```
Individual Agent Memory:
  e₁, e₂, ..., eₜ  →  DCT  →  c₀, c₁, ..., cₖ  →  decay  →  c₀, c₁, ..., cⱼ (j < k)
                                                                        ↓
Collective Hopfield Layer:                                        partial cue
  All agent memories form energy landscape                              ↓
  Basins = semantic clusters                                    Hopfield recall
  Access patterns reshape basins (reconsolidation)                      ↓
                                                                ê₁, ê₂, ..., êₜ
                                                              (reconstructed trajectory)
```

## Open Questions

- What's the optimal Hopfield architecture for embedding-space reconstruction? (continuous Hopfield vs modern Hopfield attention)
- How do you prevent catastrophic confabulation at scale? (reconstructions that are confidently wrong)
- Privacy: if memories reconstruct from organizational context, is the organization's knowledge leaking into individual recall?
- Does the scaling curve plateau, or is it unbounded?
