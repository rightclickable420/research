# Adaptive Organizations: A Self-Improving Architecture for Agent-Human Systems

**Ethan Gill & Kevin Ash**

## Abstract

We describe an architecture in which autonomous coding agents, organizational sensing, and persistent memory combine into a self-improving system. Drawing on prior work in cadence resonance, organizational thermodynamics, agent memory consolidation, and cognitive signatures, we present a unified framework where agent execution patterns drive organizational diagnosis, automated intervention, and continuous improvement. The core insight is that large-scale agent fleets generate the same signals that reveal organizational health — entropy, flow, cadence — and that closing the loop between observation and action creates a compounding flywheel that improves the entire organization over time.

## 1. Introduction

Modern software organizations are deploying autonomous coding agents at scale. Stripe reports over 1,300 merged pull requests per week produced entirely by unattended agents [1]. These agents operate within structured execution frameworks — deterministic steps interleaved with LLM reasoning — and integrate with centralized tool servers via the Model Context Protocol (MCP).

What these systems lack is memory, diagnosis, and adaptation. Each agent run starts from scratch. Patterns discovered in one execution are lost before the next begins. Organizations gain throughput but not intelligence.

We propose closing this gap by treating agent execution patterns as organizational telemetry, applying thermodynamic measures to diagnose friction, and feeding learned patterns back into the agent fleet. The result is a system that observes, learns, prescribes, builds, and measures — continuously.

## 2. Background and Prior Work

This architecture integrates five previously independent contributions:

**Cadence Resonance** [2] establishes that counting events over time produces a universal signal primitive. Cross-domain patterns emerge without predictive models when cadence signatures are compared. Applied here, agent execution cadences reveal systemic patterns invisible to individual runs.

**Organizational Thermodynamics** [3] demonstrates that organizational health can be measured from communications metadata alone — who acted, when, and who acted next — without reading content. Five metrics (Flow, Entropy, Cadence correlation, Downstream ratio, Fan-out) classify organizational zones as rivers (healthy) or swamps (friction). Applied here, agent execution metadata replaces communications metadata as the signal source.

**Embedding Trajectory Compression** [4] introduces DCT-based reconsolidation of agent memory embeddings, enabling persistent memory that improves with access. The key finding: retrieval-weighted compression outperforms uniform compression at 76% vs 34% accuracy, and access patterns — not age — should determine what gets consolidated.

**Chemical Kinetics of Agent Memory** [5] derives promotion thresholds, decay rates, and capacity limits from chunk properties rather than arbitrary constants, treating memory as a chemical system with saturation, activation energy, and phase transitions.

**Cognitive Signatures** [6] measures how people think — not what they know — across five continuous dimensions, predicting cross-domain collaboration potential. Applied here, the same behavioral measurement principles extend from individuals to teams and organizational units.

## 3. Architecture

The system consists of five layers that form a closed loop:

```
┌─────────────────────────────────────────────────┐
│                   EXECUTE                        │
│  Agent fleet + Blueprints + Toolshed (MCP)      │
└──────────────────────┬──────────────────────────┘
                       │ execution telemetry
                       ▼
┌─────────────────────────────────────────────────┐
│                   OBSERVE                        │
│  Access patterns + Error patterns + Resolution  │
│  patterns + Tool usage + Cadence signatures     │
└──────────────────────┬──────────────────────────┘
                       │ organizational signals
                       ▼
┌─────────────────────────────────────────────────┐
│                   DIAGNOSE                       │
│  Org thermodynamics: Flow, Entropy, Cadence     │
│  Phase classification: River / Swamp / Phase    │
│  transition detection                           │
└──────────────────────┬──────────────────────────┘
                       │ diagnosed friction
                       ▼
┌─────────────────────────────────────────────────┐
│                   PRESCRIBE                      │
│  Pattern correlation (dpth) + Intervention      │
│  selection + Blueprint generation + Cognitive    │
│  matching (who should review/own this)          │
└──────────────────────┬──────────────────────────┘
                       │ intervention blueprints
                       ▼
┌─────────────────────────────────────────────────┐
│                   EXECUTE                        │
│  (cycle repeats)                                │
└─────────────────────────────────────────────────┘
```

### 3.1 Execution Layer

The execution layer follows the blueprint pattern [1]: state machines that interleave deterministic code nodes with agentic subtask nodes. Deterministic nodes handle predictable operations (linting, testing, deployment scaffolding) without consuming LLM tokens. Agent nodes handle creative work (implementation, debugging, design) with scoped context and curated tool access.

A centralized MCP server (analogous to Stripe's "Toolshed") provides a shared capability layer across all agents. Tools are organized thematically, and agents receive curated subsets relevant to their task. Each tool addition immediately extends the capabilities of the entire fleet.

Context is gathered from three sources:
- **Rule files** scoped to directories and file patterns, consumed by all agent types (unattended agents, IDE assistants, CLI tools)
- **MCP tool calls** for dynamic context (documentation, ticket details, code intelligence, system state)
- **Persistent memory** via the consolidation layer, providing learned patterns from prior executions

### 3.2 Observation Layer

Every agent execution generates telemetry:

- **Access patterns**: which files, tools, and knowledge chunks were accessed, how often, and with what relevance scores
- **Error patterns**: which failures occurred, where, and how they clustered
- **Resolution patterns**: which interventions resolved which failures, and in how many iterations
- **Tool usage patterns**: which MCP tools were called, in what sequence, and whether the results were useful
- **Cadence signatures**: execution frequency, duration distributions, and phase relationships across teams and codebase regions

This telemetry is the organizational equivalent of communications metadata in the thermodynamic model [3]. Critically, it requires no access to code content, conversation content, or proprietary information — only execution metadata.

The observation layer applies the reconsolidation principle [4]: raw telemetry is liquid (unstructured, free to connect). Access patterns reveal which signals matter. Frequently-accessed patterns are consolidated into structured knowledge. Patterns that are never accessed remain liquid and eventually decay.

### 3.3 Diagnostic Layer

The diagnostic layer applies organizational thermodynamics [3] to agent execution telemetry:

**Flow** (Little's Law applied to agent tasks): throughput / work-in-progress. High flow indicates healthy code regions where agents complete tasks efficiently. Low flow indicates bottlenecks.

**Entropy** (Shannon entropy of execution outcomes): uniform success = low entropy (predictable). Diverse failure modes = high entropy (chaotic). Entropy measures how predictable a codebase region is for agents.

**Cadence correlation** (phase relationships between teams): when Team A's agents start failing, does Team B's follow? Correlated cadences reveal hidden dependencies. Anti-correlated cadences (like holiday decorrelation in retail data [2]) reveal structural relationships.

**Downstream ratio**: what fraction of an agent's output feeds into other agents' inputs? Low downstream ratio = isolated work. High ratio = critical path.

**Fan-out**: how many distinct files/modules does an agent touch per task? High fan-out = coordination tax, possible architectural smell.

These metrics classify organizational zones:

| | Low Entropy | High Entropy |
|---|---|---|
| **High Flow** | **River** — healthy, productive | **Rapids** — productive but fragile |
| **Low Flow** | **Lake** — stable but stagnant | **Swamp** — friction, needs intervention |

Phase transitions between states are detectable through cadence analysis. A river becoming rapids (entropy increasing while flow holds) is an early warning of impending breakdown.

### 3.4 Prescription Layer

Once friction is diagnosed, the system prescribes interventions. This layer draws on the correlation network (dpth) [4,5] to match diagnoses to interventions that have worked before.

**Intervention types, ordered by autonomy:**

**Level 0 — Visibility**: Surface the diagnosis to humans via dashboard. "Module X has entropy 3.2σ above baseline this week." No automated action.

**Level 1 — Knowledge**: Auto-generate scoped rule files from error patterns. "Agents in /payments/legacy/ fail on timezone handling 40% of the time. Draft rule: always normalize to UTC." Submitted as PR, human-approved.

**Level 2 — Tooling**: Auto-generate MCP tools from repeated manual patterns. "Three teams independently query inventory data through raw SQL. Draft Toolshed tool: get_store_inventory(store_id, department?)." Submitted for review.

**Level 3 — Structural**: Generate refactoring blueprints from friction analysis. "Module X has high entropy because it mixes three concerns. Draft blueprint: split into three modules with defined interfaces." Scoped, estimated, human-approved.

**Level 4 — Generative**: Conceive and build new tools from behavioral observation. "Store managers in region 4 submit 3x more adjustment tickets. Root cause: no cross-department inventory view. Draft blueprint: build unified inventory dashboard." The system identifies the need, designs the solution, and builds it.

Each intervention includes an expected impact (predicted entropy reduction, flow improvement) derived from similar past interventions in the correlation network. This prediction is calibrated against actual outcomes, improving prescription accuracy over time.

### 3.5 Cognitive Matching

The cognitive signatures framework [6] extends the prescription layer by matching interventions to the humans best suited to review, own, or collaborate on them.

Five measured dimensions — Connection Pattern, Frame Dependence, Scope Instinct, Ambiguity Response, Integration Speed — predict how individuals approach unfamiliar problems. The system uses these signatures to:

- **Route reviews**: structural refactoring proposals go to architects (high scope range). Edge case analysis goes to refiners (high precision-seeking).
- **Form teams**: pair explorers (high connection velocity) with refiners (high ambiguity tolerance) for novel problem spaces.
- **Identify capability gaps**: if no one in the organization has the cognitive signature suited to a particular class of problem, surface that as a hiring signal.

Cognitive signatures are measured behaviorally from work patterns — not self-reported — making them a natural extension of the observation layer.

## 4. The Flywheel

The architecture produces compounding returns because each cycle enriches every subsequent cycle:

**More executions → better observation.** Larger sample sizes produce more reliable entropy and flow measurements. Rare patterns become detectable.

**Better observation → better diagnosis.** Phase transitions are detected earlier. Correlation between seemingly unrelated signals emerges (cadence resonance across teams).

**Better diagnosis → better prescription.** The correlation network accumulates intervention-outcome pairs. Prescription accuracy improves. The system learns that certain intervention types work better for certain diagnostic patterns.

**Better prescription → better execution.** Agents start with more knowledge (consolidated rule files), better tools (auto-generated MCP tools), and cleaner codebases (structural improvements). One-shot success rates increase.

**Better execution → more capacity.** As agents succeed more often, humans spend less time on review and debugging. That freed capacity goes to higher-leverage work — which generates new patterns for the observation layer.

The flywheel is also self-correcting. Failed interventions are logged in the correlation network with negative calibration signals. The system learns what doesn't work and stops prescribing it. This is the chemical kinetics principle [5] applied at organizational scale: interventions that don't catalyze improvement have their activation energy raised, making them less likely to be prescribed.

## 5. Memory Architecture

The system's memory operates at three timescales, following the consolidation model [4]:

**Working memory** (per-execution): the agent's context window during a single run. Populated by rule files, MCP tool results, and blueprint context. Lost at execution end.

**Consolidating memory** (cross-execution): access patterns, error logs, and resolution histories stored in the observation layer. Raw and liquid. Shaped by usage into structured knowledge through the reconsolidation pipeline. Temporal decay ensures recent patterns rank above stale ones.

**Consolidated memory** (organizational): promoted patterns that have proven durable. Scoped rule files. Toolshed tools. Architectural decisions. Diagnostic-intervention correlations. This is the organizational equivalent of MEMORY.md — curated, compressed, evergreen.

The promotion criteria follow the chemical kinetics model [5]: a pattern promotes when its access energy (frequency × relevance score × session diversity) exceeds a threshold derived from its semantic distance from existing consolidated knowledge. Novel patterns promote more easily than redundant ones.

Critically, the system does not index raw execution transcripts into the consolidated layer. We demonstrate empirically that raw transcripts create semantic traps — dense, multi-topic chunks that match too many queries and dilute retrieval precision. Instead, the reconsolidation pipeline extracts structured patterns from transcripts and promotes only those patterns that prove durable through repeated access.

## 6. Signal Sources Beyond Agents

While agent execution telemetry is the primary signal source, the architecture accommodates additional organizational signals:

- **Support ticket patterns**: clustering by topic, resolution time, repeat frequency
- **Internal tool usage**: drop-off points, workaround detection (users creating spreadsheets when tools fail)
- **Communication metadata**: message volume, response latency, thread depth (per org thermodynamics [3])
- **Deployment patterns**: rollback frequency, hotfix cadence, change failure rate

Each signal source feeds the observation layer through the same pipeline: raw events → access pattern tracking → consolidation → diagnostic metrics. The thermodynamic model operates identically regardless of signal source, because it measures metadata properties (who, when, what happened next) rather than content.

## 7. Safety and Human Oversight

The autonomy levels (Section 3.4) provide a graduated trust model:

- **Levels 0-1** (visibility and knowledge) are safe by default — they produce information and documentation, not code changes.
- **Level 2** (tooling) requires human review of generated tools before deployment.
- **Level 3** (structural) requires human approval of blueprints before execution and human review of results.
- **Level 4** (generative) requires human approval at conception (should we build this?), design (is this the right solution?), and deployment (does it work?).

The system never escalates its own autonomy level. Humans explicitly configure which intervention levels are permitted for which organizational zones. A team can opt into Level 4 for low-risk tooling while restricting structural changes to Level 1.

All interventions are logged with full provenance: which observations led to which diagnosis, which diagnosis led to which prescription, and what the measured outcome was. This audit trail enables human review of the system's reasoning at any point.

## 8. Discussion

### Relationship to existing systems

The architecture differs from CI/CD dashboards (which show what happened but don't diagnose why), from project management tools (which track human-assigned priorities rather than measured friction), and from AI coding assistants (which optimize individual developer productivity rather than organizational capability).

The closest analogy is a nervous system. The execution layer is the musculature. The observation layer is the sensory network. The diagnostic layer is perception. The prescription layer is the motor cortex. Memory is memory. The organism improves not because any component gets better in isolation, but because the feedback loop between them tightens.

### Why now

Three prerequisites recently converged: (1) autonomous coding agents reliable enough for unattended operation at scale, (2) MCP as a standard protocol for agent-tool integration, and (3) embedding models cheap enough to run locally for persistent memory without API dependencies. The individual components existed — execution frameworks, organizational metrics, agent memory — but the connective tissue to close the loop was missing.

### Limitations

The architecture assumes sufficient agent execution volume to produce statistically meaningful signals. Organizations with fewer than ~100 agent runs per week may not generate enough telemetry for reliable diagnosis. The cognitive matching component requires behavioral observation data that accumulates slowly. And the correlation network requires months of intervention-outcome pairs before prescription accuracy exceeds naive baselines.

## 9. Conclusion

We present an architecture that transforms agent fleets from stateless executors into an organizational learning system. By applying thermodynamic measures to agent execution telemetry, consolidating patterns through access-weighted memory, and feeding learned interventions back into the agent fleet, the system creates a self-improving flywheel.

The key contributions are: (1) connecting organizational thermodynamics to agent execution telemetry as a novel signal source, (2) demonstrating that the reconsolidation memory architecture enables cross-execution learning without raw transcript indexing, (3) proposing graduated autonomy levels for automated organizational intervention, and (4) showing how cognitive signatures enable human-AI matching at organizational scale.

Each component delivers standalone value. Together, they describe an organization that gets smarter every time an agent runs.

## References

[1] A. Gray, "Minions: Stripe's one-shot, end-to-end coding agents," Stripe Engineering Blog, 2026.

[2] E. Gill and K. Ash, "Cadence Resonance: Cross-Domain Pattern Discovery Through Counting and Time," 2026. DOI: 10.5281/zenodo.18808102.

[3] E. Gill and K. Ash, "Organizational Thermodynamics: Measuring Health from Communications Metadata," 2026. DOI: 10.5281/zenodo.18808102.

[4] E. Gill and K. Ash, "Embedding Trajectory Compression for Persistent Agent Memory," 2026. DOI: 10.5281/zenodo.18808102.

[5] E. Gill and K. Ash, "Chemical Kinetics of Agent Memory: Deriving Consolidation Parameters from Chunk Properties," 2026. DOI: 10.5281/zenodo.18808102.

[6] E. Gill and K. Ash, "Cognitive Signatures: Measuring Cross-Domain Collaboration Potential," 2026.
