# Dreaming in Access Patterns: Infrastructure-Driven Memory Reconsolidation for Persistent AI Agents

**Gill, E. & Ash, K. (2026)**

AI agents can't form habits. Every approach requiring the agent to maintain its own memory during active work fails. Infrastructure must do the work instead — extracting access patterns from session transcripts, generating compressed health mirrors, and reshaping embeddings, all without agent involvement.

## The Problem

| Approach | Why It Fails |
|----------|-------------|
| Agent logs access at compaction | No habit formation → unreliable |
| Agent reports signals during work | Competes with actual task → skipped |
| Agent restructures during heartbeats | Depends on prompt + prioritization → inconsistent |
| Agent runs reconsolidation code | No persistent awareness the code exists |

## Architecture

Three background processes, zero agent involvement:

```
Session Transcripts → [Extract] → access.db → [Mirror] → mirror.md
                                      ↓
                                  [Reconsolidate] → reshaped embeddings
```

1. **extract_sessions.py** — Cron job parses session JSONL, extracts every `memory_search` call and result
2. **mirror.py** — Analyzes access data, generates compressed snapshot (hot chunks, gaps, friction, resonance, promotion candidates)
3. **pipeline.py** — DCT reconsolidation: weights embeddings by access energy, transforms, truncates, reconstructs. Accessed memories survive compression; unaccessed fade.

## Key Discoveries

**Semantic traps:** Dense unstructured sections match too many unrelated queries, displacing better results. Fix: subheadings split chunks by topic.

**Access, not age, triggers structuring:**
- Hot swamp (accessed + unstructured) → structure now
- Cold swamp (unstructured + never accessed) → leave alone
- The concrete metaphor: fresh memories stay liquid until traffic reveals the shape

**Connection graph lives in access data, not files:** Writing "this connects to X" in memory creates more semantic traps. Keep files topically clean; let co-activation data store the edges.

## Deployment

- **Cron:** `recon cycle` at 5am UTC daily
- **Heartbeat:** 60min, reads mirror + acts on findings
- **First result:** Identified and restructured 6 semantic traps in production

## Files

```
src/
├── extract_sessions.py   # Session transcript → access event extraction
├── access_logger.py      # SQLite access event storage + chunk energy tracking
├── mirror.py             # Compressed memory health snapshot generator
└── pipeline.py           # DCT reconsolidation with live access data

paper.md                  # Full paper
paper.pdf                 # PDF version
```
