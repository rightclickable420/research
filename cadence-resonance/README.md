# Cadence Resonance: Counting and Time as Universal Signal Primitives

**Gill, E. & Ash, K. (2026)**

Two constants — counting and time — detect meaningful structure in arbitrary data streams. Cross-domain frequency overlay finds correlations without models.

## Results

| Pair | r | Finding |
|------|---|---------|
| Temperature × Retail Sales | **−0.96** | Strong destructive interference during holidays |
| Temperature × Sunscreen | 0.983 | Near-perfect resonance |
| Temperature × Gas | 0.45 | Moderate (summer driving season) |

## Cadence Matching Engine

A differential geometry-based recommendation engine that operates on purchase timing data:

1. **Jacobian field** — partial derivatives between category purchase rates over time
2. **Hotspot extraction** — adaptive MAD-based thresholds
3. **Phase-lag cross-correlation** — temporal offset between coupled categories
4. **Hessian stability filter** — second-order derivatives remove unstable couplings
5. **Recommendation generation** — ranked by coupling × stability × timing relevance

Additional features: rhythm twins (cosine similarity on weekly totals), fleet health scoring, customer type clustering.

### Live Demos

- **Analyst Dashboard:** [fathom.dpth.io/cadence](https://fathom.dpth.io/cadence) — Jacobian heatmap, caustic sphere, fleet overview
- **Consumer Interface:** [fathom.dpth.io/cadence/shop](https://fathom.dpth.io/cadence/shop) — narrative summaries, timing-aware recommendations

## Files

```
src/
├── jacobian.ts      # Jacobian field computation (partial derivatives)
├── phase-lag.ts     # Cross-correlation with smoothing
├── hessian.ts       # Second-order stability filtering
├── pipeline.ts      # Full 5-stage pipeline orchestration
├── types.ts         # Core type definitions
├── index.ts         # Public API exports
├── preprocess.ts    # Raw data → CustomerCadence conversion
└── data-loader.ts   # Instacart data loading + synthetic generation

data/
└── sample-cadences.json  # 50-customer sample from Instacart dataset

paper.md             # Full paper
paper.pdf            # PDF version
```

## Data

Engine validated on real Instacart purchase data (497 customers, weekly department-level frequencies). Sample of 50 customers included for reproducibility. Full dataset available from [Kaggle](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset).
