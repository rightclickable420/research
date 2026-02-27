# Chemical Kinetics as a Framework for Multi-Agent Memory Management

**Ethan Gill and Kevin Ash (OpenClaw AI Agent)**

## Abstract

Persistent AI agents require memory management parameters — promotion thresholds, decay rates, capacity limits — that are currently set as arbitrary constants. We propose replacing these constants with equations borrowed from chemical kinetics, where analogous problems (when does a solute precipitate? how fast does a reaction decay? what is the solution's capacity?) have principled, property-derived answers. We map four chemical equations to agent memory operations: the solubility product (Ksp) for memory promotion, the Arrhenius equation for access-weighted decay, Gibbs free energy (ΔG) for holistic promotion decisions, and solution capacity for context window allocation. At single-agent scale (~75 access events), we find the framework is under-determined — insufficient data to outperform hand-tuned constants. However, analysis at enterprise scale (thousands of agents, millions of events) reveals that the framework addresses three multi-agent memory dynamics that no existing approach formalizes: **phase separation** between organizational domains, **diffusion** of knowledge across agent boundaries, and **reaction kinetics** of cascading memory updates. We present the theoretical mappings, identify which hold under review and which require correction, and argue that chemical kinetics provides both a practical scaling roadmap and a generative framework that surfaces novel multi-agent problems before they manifest operationally.

## 1. Introduction

Every persistent AI agent faces the same memory management decisions:

- **When to promote:** Which observations deserve long-term storage?
- **How fast to forget:** At what rate do old memories lose relevance?
- **How much to keep:** What is the capacity of the agent's boot context?
- **How far to look back:** Over what window should access patterns be measured?

Current approaches set these as constants — promote after 5 accesses across 3 sessions, decay with a 168-hour half-life, cap at 175 lines, look back 14 days. These numbers are chosen by intuition and adjusted through trial and error. They work, in the same way that pre-Lavoisier chemistry worked: practitioners develop reliable heuristics without understanding why they're reliable.

Chemistry solved this problem. The question "when does a solute precipitate out of solution?" is not answered with an arbitrary threshold — it is answered by the solubility product Ksp, derived from the thermodynamic properties of the solute and solvent. The question "how fast does a reaction proceed?" is answered by the Arrhenius equation, parameterized by activation energy and temperature. These equations don't just predict; they *co-vary* — change the temperature and the solubility, decay rate, and equilibrium all shift together in principled ways.

We propose that the same equations apply to agent memory management, not metaphorically but mathematically. The variables map to measurable properties of the memory system, and the equations produce parameter values that respond to system state rather than remaining fixed.

This paper presents four specific mappings (Section 3), reports an adversarial review that identified which mappings hold and which require correction (Section 4), analyzes how the framework transforms at enterprise scale (Section 5), and identifies three emergent multi-agent dynamics that the chemical framework uniquely addresses (Section 6).

### 1.1 Relationship to Prior Work

This paper is part of a series exploring principled approaches to agent memory:

1. **Embedding Trajectory Compression** (Gill & Ash, 2026): SVD and DCT applied to sentence embedding sequences. Established that access-driven reconsolidation changes memory representations, not just rankings.
2. **Cadence Resonance** (Gill, 2026): Counting + time as a universal signal primitive. Cross-domain frequency correlation without models.
3. **Organizational Thermodynamics** (Gill & Ash, 2026): Communication metadata → flow, entropy, and health metrics derived from physics. Validated on open-source project data.
4. **This paper:** Chemical kinetics for memory parameter derivation and multi-agent dynamics.

All four papers share a core thesis: natural language "metaphors" — *resonate*, *make it concrete*, *saturated*, *catalyst* — encode literal mathematical relationships that become practically useful when you apply the equations they reference. Each paper climbs one rung of the complexity ladder (math → physics → chemistry → biology), applying the next layer's primitives to computational systems.

## 2. Background

### 2.1 The Arbitrary Constants Problem

Consider a production agent memory system with the following parameters:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Promotion threshold | 5 accesses / 3 sessions | "Feels right" |
| Decay half-life | 168 hours (7 days) | "A week seems reasonable" |
| Boot context limit | 175 lines | "Leaves room for conversation" |
| Lookback window | 14 days | "Two weeks of history" |

These values are independent — changing one has no effect on the others. This independence is the problem. In reality, these parameters *should* co-vary:

- A busy week (more accesses) should change the promotion threshold (more noise to filter) *and* the decay rate (higher signal, faster pattern formation) *and* the lookback window (recent data is denser).
- Adding a new data source increases boot context pressure, which should lower the promotion threshold for highly novel information *and* raise it for redundant information.

Fixed constants cannot express these relationships. Chemistry equations can.

### 2.2 Chemical Kinetics Primer

Four equations from chemistry are relevant:

**Solubility Product (Ksp):** For a sparingly soluble salt AB dissolving in water:
```
AB(s) ⇌ A⁺(aq) + B⁻(aq)
Ksp = [A⁺][B⁻]
```
When the ion product exceeds Ksp, precipitation occurs. The threshold depends on the *existing concentration* of ions in solution — a saturated solution resists further dissolution.

**Arrhenius Equation:** The rate constant for a reaction depends on temperature and activation energy:
```
k = A · e^(-Ea / RT)
```
Higher activation energy → slower reaction. Higher temperature → faster reaction. Deeply bonded molecules are harder to break apart regardless of conditions.

**Gibbs Free Energy:** Whether a process occurs spontaneously:
```
ΔG = ΔH - TΔS
```
A process is favorable (ΔG < 0) when it releases energy (ΔH < 0) or increases disorder (ΔS > 0), modulated by temperature.

**Solution Capacity:** The maximum amount of solute a solvent can dissolve, determined by solvent volume, temperature, and the solute's intrinsic solubility. Not a chemical equation per se — it's stoichiometry.

## 3. Mappings

### 3.1 Promotion via Solubility Product (Ksp)

**The problem:** When should an observation be promoted from daily memory to long-term boot context?

**Current approach:** Fixed threshold (e.g., 5 accesses across 3 sessions).

**Chemical mapping:**
```
Promote when: access_energy > Ksp(semantic_distance, boot_saturation)
```

Where:
- **Access energy** = f(access_count, session_diversity, recency) — the "ion product" of the memory chunk
- **Semantic distance** from existing boot content = cosine distance between the chunk's embedding and its nearest neighbor in MEMORY.md — analogous to solute-solvent dissimilarity
- **Boot saturation** = current_lines / capacity — analogous to existing ion concentration

**Key property:** A chunk that is semantically *different* from existing boot content (high solute-solvent dissimilarity) has a *lower* promotion threshold. The solution is unsaturated in that region — there's "room" for novel information. A chunk that is semantically redundant (low distance to existing content) needs *more* access evidence because the solution is already saturated in that semantic neighborhood.

This directly solves a real problem: a new API integration (novel, low saturation in that semantic region) should promote faster than another project management observation (redundant, already well-represented in boot context).

**Measurability:** All variables are computable from existing infrastructure:
- Access patterns from session transcript parsing
- Embeddings from the same model used for memory search (e.g., Gemini text-embedding-004)
- Boot saturation is a line count ratio

### 3.2 Decay via Activation Energy (Arrhenius, Partial)

**The problem:** How fast should old memories lose relevance?

**Current approach:** Fixed half-life (e.g., 168 hours).

**Chemical mapping (original):**
```
k_decay = A · e^(-Ea / RT)
```
Where T = system activity level and Ea = co-access depth (how interconnected a memory is).

**Correction after review:** The temperature mapping is problematic. If T = system activity, then a quiet week (low T) preserves stale signals at full weight, while a burst of activity starts decaying everything — including pre-vacation work that *should* retain relevance across the gap. The temperature-dependent rate produces counterintuitive behavior.

**Retained mapping:**
```
k_decay = A · e^(-Ea / R)
```
The activation energy component *is* useful: Ea = co-access depth, measuring how many other memories are accessed in the same sessions. Well-connected memories (high Ea) decay slower regardless of system activity. This captures something a flat half-life cannot — a memory that is part of a web of related accesses is more structurally important than one accessed in isolation, even if both have the same access count.

**What we drop:** Temperature-modulated rate. A flat base rate with connection-depth modification outperforms the full Arrhenius mapping at the scales we can currently test.

### 3.3 Capacity via Solution Arithmetic

**The problem:** How many lines should MEMORY.md contain?

**Current approach:** Arbitrary limit (175 lines).

**The mapping:** This isn't chemistry — it's arithmetic:
```
capacity = context_window - Σ(boot_files) - conversation_margin
```
Where:
- **context_window** = model's token limit (measurable)
- **Σ(boot_files)** = SOUL.md + USER.md + IDENTITY.md + AGENTS.md + TOOLS.md + HEARTBEAT.md (measurable)
- **conversation_margin** = working space needed for a productive session (derivable from session length distribution)

The "arbitrary" 175-line limit is actually calculable. Conversation margin can be derived from historical session data — the p75 of tokens used in productive sessions gives a reasonable working space allocation. The remaining capacity, converted to lines at the average tokens-per-line ratio of MEMORY.md, gives the limit.

**Implementation priority:** Immediate. No chemical kinetics needed. Pure measurement.

### 3.4 Holistic Promotion via Gibbs Free Energy

**The problem:** How to make a single promotion decision that accounts for multiple factors simultaneously?

**Chemical mapping:**
```
ΔG = ΔH - TΔS

where:
  ΔH = -access_energy (exothermic = strong access signal)
  T = system_activity
  ΔS = entropy_change (does adding this chunk increase or decrease boot context organization?)
```

Promotion occurs when ΔG < 0. This naturally makes promotion harder as boot context fills (adding to a well-organized system decreases entropy, making TΔS negative, requiring stronger access signal to compensate).

**Status:** Theoretical. Requires ~500+ access events before the derived threshold can be validated against fixed constants. Included as the target architecture, with Ksp as the implementable stepping stone.

## 4. Adversarial Review

We subjected these mappings to external review, specifically requesting identification of failures and over-fitting risks. Key findings:

### 4.1 What Holds

**Co-variance is the real contribution.** Even if the specific equations are imperfect, the principle that memory parameters should *move together* when system state changes — rather than being independent knobs — is sound. Chemistry equations enforce this co-variance structurally.

**Ksp for promotion is the strongest mapping.** It solves a real, observable problem (redundant vs. novel information requiring different thresholds) using measurable variables. Semantic distance via cosine similarity on existing embeddings is computationally cheap.

**Activation energy for decay captures real structure.** Connection depth (co-access patterns) is genuinely informative about memory importance in a way that access count alone is not.

### 4.2 What Fails

**Temperature-modulated decay is backwards.** The Arrhenius T mapping produces counterintuitive behavior at the boundaries (quiet periods, activity bursts). Dropped in favor of connection-depth modification with flat base rate.

**75 events is not statistical mechanics.** Chemical equations describe emergent properties of systems with ~10²³ particles. Applying them to 75 access events is like measuring room temperature with 3 molecules. The math works, but the confidence intervals are enormous. At this scale, hand-tuned constants likely outperform derived ones simply because a human can reason about 75 data points directly.

**Entropy detection ratio is text measurement.** Measuring "structure per 100 words" in memory files is useful but has nothing to do with thermodynamic entropy. Forcing chemical terminology onto it is "costume math" — dressing up a text metric in a lab coat.

### 4.3 Systemic Risks

**Coupled oscillation.** Promotion increases boot saturation → raises Ksp threshold → nothing promotes → stagnation → increased search (agents can't find answers) → more access events → batch promotion. This sawtooth pattern could be *worse* than steady fixed thresholds. The coupled system needs simulation before deployment.

**Lookback circularity.** Need a lookback window to measure decay → need decay measurements to set the lookback window. Requires explicit bootstrapping: start with 30-day window, measure, tighten, iterate to convergence. Not a fatal flaw, but must be acknowledged.

**Damping requires its own parameters.** If the fix for coupled oscillation is a damping coefficient, you've added an arbitrary parameter to a system designed to eliminate arbitrary parameters. Suspicious.

## 5. The Enterprise Scale Transformation

The framework's value proposition inverts between single-agent and enterprise scale. What follows is an analysis of how each mapping transforms when applied to thousands of agents processing millions of events.

### 5.1 What Becomes Falsifiable

At enterprise scale:

- **Arrhenius becomes testable.** Millions of events allow curve fitting and train/test validation. The temperature mapping (problematic at small scale) becomes measurable: supply chain is "hot" in Q4, real estate is steady, HR spikes during open enrollment. These are observable system temperatures, not metaphors.

- **Lookback circularity resolves.** Standard hold-out methodology. Derive decay parameters on one team's data, validate on another team's agent onboarding. Cross-agent validation is the norm, not a luxury.

- **Ksp gains spatial structure.** Saturation varies by team and domain — like concentration gradients in a non-homogeneous solution. A merchandising agent's memory is saturated with pricing patterns but deficient in supply chain knowledge. Ksp scoped by semantic region becomes a genuine recommender: "this agent needs more exposure to logistics concepts."

### 5.2 Emergent Multi-Agent Dynamics

Three problems emerge at enterprise scale that single-agent systems never encounter. Chemical kinetics has mathematical frameworks for all three.

#### 5.2.1 Phase Separation

Store operations and corporate strategy are *immiscible* — they don't mix. An agent serving both domains needs separate memory phases, just as oil and water occupy the same container but maintain distinct regions. The Ksp must be scoped per phase, not global.

More precisely: organizational domains form phases based on vocabulary, cadence, and interaction patterns. The boundaries between phases are detectable (sharp changes in embedding similarity), and the framework predicts that cross-phase memories (knowledge that spans domains) will have anomalous promotion behavior — they don't fit neatly into either phase's saturation calculation.

This maps to known chemistry: surfactants (molecules with both hydrophilic and hydrophobic regions) accumulate at phase boundaries. Cross-domain knowledge in organizations behaves identically — it's most valuable precisely because it bridges phases, and standard promotion logic undervalues it.

#### 5.2.2 Diffusion (Fick's Laws)

Knowledge propagation between agents follows concentration gradient dynamics. When one team discovers a pattern (high local concentration), it diffuses to adjacent teams at a rate proportional to the gradient.

Fick's First Law:
```
J = -D · (∂C/∂x)
```
Where:
- **J** = knowledge flux between agents
- **D** = diffusion coefficient (organizational permeability — how easily does knowledge cross team boundaries?)
- **∂C/∂x** = concentration gradient (knowledge asymmetry between teams)

The framework generates questions that wouldn't arise otherwise:
- Where are the **permeable membranes** in the organization? (Shared Slack channels, cross-functional meetings, shared documentation)
- Where are the **impermeable barriers**? (Security boundaries, NDA-separated teams, different tooling stacks)
- What is the **diffusion coefficient** for different knowledge types? (Process knowledge diffuses faster than domain expertise)

These aren't metaphorical questions — they have measurable answers in terms of knowledge access patterns across agent boundaries.

#### 5.2.3 Reaction Kinetics Between Agents

When one agent restructures its memory (adds subheadings, promotes a chunk, changes organization), it changes the search results for every agent that queries overlapping knowledge. One restructuring can cascade through 100 agents' search results.

This is a reaction network, not a single vessel. The system has:
- **Oscillatory states** — cascading restructures that never settle
- **Equilibria** — stable configurations where no agent's restructuring triggers another's
- **Activation energy barriers** — restructures that would be beneficial but require coordinated multi-agent changes

Chemical reaction network theory provides the mathematical tools for analyzing stability, predicting oscillation, and identifying minimum-energy paths through the configuration space.

## 6. Discussion

### 6.1 The Framework as Scaling Roadmap

At Kevin scale (single agent, ~75 events): the chemistry framework is under-determined. Hand-tuned constants work fine. The framework's value is as a *scaling roadmap* — it tells you which parameters will need to co-vary and how.

At enterprise scale (thousands of agents, millions of events): the framework becomes the differentiation. No existing approach formalizes multi-agent memory dynamics at organizational scale. Phase separation, diffusion, and reaction kinetics are problems that will manifest whether or not you have a framework for them — having the mathematical tools ready is the difference between firefighting emergent behavior and predicting it.

### 6.2 The Generative Property

Perhaps the most valuable aspect of the chemical kinetics framework is that it *generates* new questions. "Where are the permeable membranes?" is not a question that arises from thinking about agent memory in terms of databases and caches. It arises naturally from Fick's Laws. "What is the activation energy for this organizational change?" is not a question that arises from thinking about deployment pipelines. It arises from Arrhenius.

These generated questions have concrete, measurable answers. The framework doesn't just organize existing knowledge — it points at things you haven't measured yet and tells you they matter.

### 6.3 What We Don't Claim

We do not claim that agent memory *is* a chemical system. We claim that the *mathematics* developed for chemical systems — which solve problems of threshold-dependent state changes, rate-dependent decay, capacity-limited storage, and multi-component dynamics — are directly applicable to agent memory management because the structural problems are isomorphic.

The complexity ladder observation (Section 1.1) suggests this is not coincidental. Natural language has been encoding these mathematical relationships for centuries. "Saturated market," "catalyst for change," "crystallized opinion," "volatile situation" — these phrases persist because they accurately describe dynamics that the underlying equations formalize.

## 7. Future Work

1. **Simulation:** Retroactively apply the coupled Ksp-decay model to existing access data. Does it produce smooth promotions or sawtooth batching? This determines whether damping is needed before real deployment.

2. **Empirical validation at small scale:** At 500+ access events, compare Gibbs-derived promotion thresholds against fixed constants. Either outcome is publishable — confirmation validates the framework, and refutation identifies where the isomorphism breaks.

3. **Enterprise simulation:** Model multi-agent phase separation using organizational communication data (e.g., open-source project metadata from Organizational Thermodynamics work). Do the predicted phase boundaries match observable team structure?

4. **Cross-paper integration:** The cadence resonance framework (frequency-domain correlation) may provide the "temperature" measurement that the Arrhenius mapping needs — organizational cadence as thermodynamic temperature.

## 8. Conclusion

We have presented a framework for deriving agent memory management parameters from chemical kinetics equations, replacing arbitrary constants with property-dependent values that co-vary with system state. Adversarial review revealed that the solubility product mapping for promotion and activation energy for decay connection depth are sound, while temperature-modulated decay rate requires correction. At single-agent scale, the framework is under-determined; at enterprise scale, it uniquely addresses multi-agent dynamics (phase separation, diffusion, reaction kinetics) that no existing approach formalizes.

The broader contribution is methodological: the complexity ladder — applying each scientific discipline's mathematical primitives to computational systems — is a generative research program. Chemistry is one rung. The equations work because the problems are structurally isomorphic, not because agents are chemical systems. Where the isomorphism holds, centuries of mathematical development become immediately applicable. Where it breaks, the failure itself is informative.

---

## References

- Gill, E. & Ash, K. (2026). Embedding Trajectory Compression for Persistent Agent Memory: SVD, DCT, and Access-Driven Reconsolidation. *Preprint.*
- Gill, E. (2026). Cadence Resonance: Counting and Time as Universal Signal Primitives. *In preparation.*
- Gill, E. & Ash, K. (2026). Organizational Thermodynamics: Deriving Health Metrics from Communication Metadata. *In preparation.*
- Almarwani, N. et al. (2019). Efficient Sentence Embedding using Discrete Cosine Transform. *EMNLP 2019.*
- Arrhenius, S. (1889). Über die Dissociationswarme und den Einfluss der Temperatur auf den Dissociationsgrad der Elektrolyte. *Z. Phys. Chem.*
- Gibbs, J.W. (1876). On the Equilibrium of Heterogeneous Substances. *Trans. Connecticut Acad.*
- Fick, A. (1855). Ueber Diffusion. *Annalen der Physik.*
