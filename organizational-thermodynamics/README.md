# Organizational Thermodynamics: Automated Attention from Communication Metadata

**Gill, E. & Ash, K. (2026)**

Five metrics derived from three metadata fields — who, when, who-next — classify organizational health without reading message content. An autonomous diagnostic agent investigates flagged entities.

## Results (vercel/next.js, 90 days)

- **9,672 events**, 216 active entities, ~2,800 threads
- **Computation time:** 1.3 seconds on a 4GB VPS
- **Content accessed:** 14 of 2,800+ threads (0.5%)

| Quadrant | Count | Description |
|----------|-------|-------------|
| River | 66 (31%) | Healthy flow, predictable outcomes |
| Waterfall | 58 (27%) | High throughput, diverse outcomes |
| Bottleneck | 25 (12%) | Low throughput, predictable |
| Swamp | 67 (31%) | Low throughput, high entropy |

### Diagnostic Agent Results

Three swamp entities, same classification, three different root causes — all correctly identified by the agent:

| Entity | Root Cause | Category |
|--------|-----------|----------|
| TrevorBurnham | Works in complex areas, not a struggling person | Structural |
| rosbitskyy | Community PRs never merged — merge gate | Process |
| Netail | Cross-system deps, no owner (PR open 226 days) | Organizational |

## Five Metrics

1. **Flow** (Little's Law) — throughput / inventory
2. **Entropy** (Shannon) — outcome distribution unpredictability (0-1)
3. **Cadence** (Pearson) — phase correlation between entity activity patterns
4. **Downstream ratio** — fraction of interactions producing external output
5. **Fan-out trajectory** — participant count trend per thread

## Diagnostic Pipeline

```
Metadata → Thermodynamic Map → Flagged Entities → Scoped Threads → Content Review → Diagnosis
9,672      216                  67                 14               3 validated
```

## Files

```
src/
├── flow.ts           # Little's Law per entity
├── entropy.ts        # Shannon entropy over thread outcomes
├── cadence-sync.ts   # Pearson correlation on activity time series
├── downstream.ts     # External output fraction
├── fanout.ts         # Participant accumulation tracking
├── diagnose.ts       # 4-stage diagnostic agent pipeline
├── events.ts         # GitHub → InteractionEvent converters
├── types.ts          # Core type definitions
├── index.ts          # Public API
└── build-presentation.ts  # Fathom presentation builder

paper.md              # Full paper
paper.pdf             # PDF version
```

## Input Specification

```typescript
interface InteractionEvent {
  from: string;        // who
  timestamp: string;   // when
  thread_id: string;   // context
  event_type?: string; // close, merge, assign, comment, review
  labels?: string[];
}
```

Works with: GitHub, Slack, Teams, email, Jira, Calendar — anything with who/when/who-next.

### Live Demo

[fathom.dpth.io/app?slug=nbccIR](https://fathom.dpth.io/app?slug=nbccIR) — Next.js organizational thermodynamics
