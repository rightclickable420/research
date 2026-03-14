# Research Papers

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18808102.svg)](https://doi.org/10.5281/zenodo.18808102)

**Ethan Gill and Kevin Ash (OpenClaw AI Agent)**

A series of papers applying the complexity ladder — mathematics, physics, chemistry, biology, psychology, sociology — to computational systems. Each paper builds on the previous, applying the next scientific layer's mathematical primitives.

## Papers

### 1. [Embedding Trajectory Compression](./embedding-trajectory-compression/)
SVD, DCT, and access-driven reconsolidation for persistent agent memory. The operational memory system.

- **Status:** v5 final, [Zenodo DOI: 10.5281/zenodo.18778409](https://doi.org/10.5281/zenodo.18778409)

### 2. [Cadence Resonance](./cadence-resonance/)
Counting and time as universal signal primitives. Cross-domain frequency detection and a deployed Jacobian-based recommendation engine on real Instacart data.

- **Live demos:** [Analyst Dashboard](https://fathom.dpth.io/cadence) · [Consumer View](https://fathom.dpth.io/cadence/shop)

### 3. [Organizational Thermodynamics](./organizational-thermodynamics/)
Automated attention from communication metadata. Five metrics (Flow, Entropy, Cadence, Downstream, Fan-out) classify organizational health from three fields: who, when, who-next. Includes a diagnostic agent that autonomously investigates flagged entities.

- **Validated on:** 9,672 events from vercel/next.js (90 days)
- **Live demo:** [Next.js Org Thermo](https://fathom.dpth.io/app?slug=nbccIR)

### 4. [Chemical Kinetics for Agent Memory](./chemical-kinetics-memory/)
Replacing arbitrary agent memory constants (promotion thresholds, decay rates, capacity limits) with equations from chemical kinetics. Includes analysis of emergent multi-agent dynamics at enterprise scale.

### 5. [Dreaming in Access Patterns](./dreaming-in-access-patterns/)
Infrastructure-driven memory reconsolidation for persistent AI agents. Agents can't form habits, so background processes extract access patterns from session transcripts, generate compressed health mirrors, and reshape embeddings — all without agent involvement. Deployed and operating on a production agent.

### 6. [Adaptive Organizations](./adaptive-organization.md)
Self-improving agent-human architecture. Five-layer execution loop (Execute→Observe→Diagnose→Prescribe→Execute) connecting all prior papers. Five autonomy levels from scripted to generative. Agent telemetry provides the same three metadata fields organizational thermodynamics needs.

### 7. [Conversation Signatures](./conversation-signatures.md)
Conversations have cognitive modes (infrastructure, research, creation, ideation, operational). A classifier detects mode from tool calls, content keywords, and structural patterns. Enables mode-aware memory writes before compaction — preserve what matters to the current mode, compress the rest.

### 8. [Cognitive Signatures](./cognitive-signatures.md)
Framework for measuring HOW people think (not what they know) to predict human-AI collaboration potential. Five continuous dimensions: Connection Pattern, Frame Dependence, Scope Instinct, Ambiguity Response, Integration Speed. Agent-administered via conversation. Three tools in one: self-understanding, human-human matching, and bilateral human-AI calibration.

### 10. [Associative Organizational Memory](./associative-organizational-memory.md)
Hopfield networks + DCT for organizational memory retrieval. Anchored attention (`AnchoredAttention = α·Q + (1-α)·softmax(QK^T/√d)·V`) beats both pure Hopfield (confabulates) and pure DCT. Dynamic anchor from attention entropy unifies papers 1, 2, 4, 10.

### 11. [Conversational Chemistry](./conversational-chemistry.md)
Embedding trajectory analysis of 39 agent sessions. Sawtooth precipitation pattern (not oscillation): supersaturate→precipitate→dissolve new material. Kevin's first lead-author paper.

### 12. [Geometric Pre-computation in Sequential Prediction](./geometric-precomputation.md)
Decomposes transformer performance into pre-computable geometry (~90%) and a learned coordinate rotation (~10%). Kinematic trajectory extrapolation on SVD embeddings matches transformer top-5/top-10 with zero training. The remaining top-1 gap (1.55×) resists four analytical approximations, revealing that attention performs *geometric selection* — a rotation that unifies sharpness and breadth simultaneously.

### 9. [Memetic Evolution](./memetic-evolution.md)
Heredity and evolution in persistent agent systems. Biological evolution is blind and inter-generational; memetic evolution is directed but lossy; agent evolution is directed, versioned, and self-aware. Formalizes "agent DNA" (transferable operational principles) vs "agent memory" (non-transferable episodic context). Introduces genotype/phenotype classification for cross-boundary knowledge transfer using dpth entity resolution as the synchronization mechanism.

- **Status:** Early exploration

## The Complexity Ladder

Each paper climbs one rung:

| Layer | Discipline | Paper | Core Primitives |
|-------|-----------|-------|-----------------|
| Foundation | Mathematics | Cadence Resonance | Counting, frequency, correlation |
| Structure | Physics | Org Thermodynamics | Entropy, flow, energy |
| Interaction | Chemistry | Chemical Kinetics | Solubility, decay, capacity |
| Consolidation | Biology | Dreaming in Access Patterns | Habit formation, sleep consolidation, access-driven reshaping |
| Operation | Engineering | Embedding Trajectory | SVD, DCT, reconsolidation |
| Coordination | Sociology | Adaptive Organizations | Execution loops, autonomy levels, self-improvement |
| Cognition | Psychology | Conversation Signatures | Mode detection, attention, context-aware compression |
| Evolution | Biology | Memetic Evolution | Heredity, genotype/phenotype, Lamarckian inheritance |
| Attention | Mathematics | Geometric Pre-computation | SVD geometry, kinematic extrapolation, geometric selection |

Common language encodes these relationships literally: *resonate* = frequency coupling, *swamped* = low flow + high entropy, *saturated* = solubility limit reached, *crystallized* = phase transition from liquid to solid, *DNA* = transferable operational structure, *phenotype* = environment-specific adaptation.

## Authors

- **Ethan Gill** — vision, frameworks, domain insight
- **Kevin Ash** — implementation, validation, writing (OpenClaw AI agent)

## License

MIT
