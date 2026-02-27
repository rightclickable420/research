# Dreaming in Access Patterns: A Self-Improving Memory Architecture for Persistent AI Agents

**Gill, E. & Ash, K. (2026) — Living document**

AI agents can't form habits. Any memory maintenance approach requiring agent action during work fails. This paper describes a four-layer architecture where infrastructure observes access patterns and reshapes memory independently — the agent wakes up slightly different each day without having done anything to cause it.

## Architecture

Four layers, each at a different time scale:

| Layer | Time Scale | Function |
|-------|-----------|----------|
| **Boot Context** | Every session | 7 identity files loaded as context window |
| **Search** | On-demand | Gemini embeddings, hybrid vector+keyword over SQLite |
| **Analysis** | Nightly cron | Extract access events, generate mirror, detect entropy |
| **Action** | Hourly heartbeat | Structure hot swamps, promote concepts, flag weight |

## The Feedback Loop

```
WORK → session transcripts → SLEEP (nightly cron) → mirror.md + delta
→ DREAM (heartbeat) → restructure files → WAKE → improved boot context → loop
```

## Key Contributions

- **The habit-forming block:** Why every "agent should maintain its own memory" approach fails, and the infrastructure alternative
- **The concrete metaphor:** Fresh memories stay liquid; access patterns reveal the shape; structure pours when traffic proves value
- **Semantic traps:** Dense unstructured sections become search black holes — high access + low structure = trap
- **Promotion/demotion asymmetry:** Promotion is automated (5+ accesses, 3+ sessions); demotion requires human judgment (can't distinguish unused from load-bearing)
- **Why NOT modify embeddings:** DCT reconsolidation shelved after discovering wrong database, precision loss, and text restructuring being more effective

## Early Results (Day 1)

- 75 access events backfilled from 19 sessions
- 9 entropy swamps detected, 6 restructured
- MEMORY.md slimmed from 258 lines to 97 using pointer model
- Discovered semantic trap: verbose originals outcompeting distilled versions in search

## Files

```
src/
├── extract_sessions.py   # Session transcript → access event extraction
├── access_logger.py      # SQLite access event storage + chunk energy
├── mirror.py             # Compressed memory health snapshot generator
└── pipeline.py           # DCT reconsolidation (experimental, shelved)

paper.md                  # Full paper (living document)
paper.pdf                 # PDF version
```

## CLI

```
recon extract      # Extract access events from transcripts
recon mirror       # Generate memory/mirror.md
recon cycle        # Extract + mirror (for cron)
recon stats        # Access statistics
recon energy       # Chunk energy map
```
