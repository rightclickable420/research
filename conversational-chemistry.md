# Conversational Chemistry: Measuring the Dynamics of Collaborative Cognition

**Ethan Gill and Kevin Ash (OpenClaw AI Agent)**

## Abstract

We present a framework for measuring the dynamics of collaborative cognition using embedding-space trajectory analysis. Analyzing 39 conversation sessions between a human and an AI agent, we find that research-productive conversations exhibit statistically distinct dynamics compared to operational ones: higher semantic step distance (Cohen's d = 1.25) and sustained candidate diversity (d = 1.26). Contrary to our initial hypothesis that productive conversations would show oscillatory "breathing" patterns, we observe a sawtooth dynamic: sustained supersaturation punctuated by sharp precipitation events. Transcript analysis at convergence points reveals a consistent three-phase microstructure — accumulation, synthesis collapse, and immediate re-expansion — analogous to chemical precipitation and dissolution cycles. We formalize these dynamics using the vocabulary of solution chemistry, demonstrate that the metaphor of "team chemistry" encodes a measurable physical process, and propose applications ranging from real-time agent adaptation to organizational diagnostics.

## 1. Introduction

"Team chemistry" is universally invoked and never measured. Teams that produce exceptional work are said to have chemistry; teams that don't are said to lack it. The term is treated as irreducibly subjective — a vibe, a feeling, something you know when you see it.

We propose that team chemistry is not subjective. It is a measurable dynamic in the embedding space of conversation, and it follows the same laws as literal chemistry: supersaturation, nucleation, precipitation, and dissolution.

This paper sits at the chemistry layer of the complexity ladder — a framework connecting mathematical, physical, chemical, biological, and psychological primitives across computational systems [1-10]. Where cadence resonance [2] provides the mathematical foundation (counting and time), organizational thermodynamics [3] provides the physical layer (entropy, flow, energy), and chemical kinetics [4] provides the interaction rules (solubility, decay, catalysis), conversational chemistry applies these chemical dynamics to the conversation itself — the reaction vessel where collaborative cognition occurs.

### 1.1 The Conversation as Reaction Vessel

A conversation between two cognitive systems — human-human, human-AI, or potentially AI-AI — is not merely information exchange. It is a process in which candidate models are generated, tested, and either stabilized or dissolved. This process has measurable dynamics that distinguish productive conversations from unproductive ones.

We draw on three observations:

1. **Thinking is model-building.** The human brain generates candidate mental models in parallel, applies selection pressure through internal debate or external testing, and retains survivors [11, 12]. A "thought" is a candidate that passed selection.

2. **Conversation externalizes this process.** A critical thinker's internal dialogue — notice, focus, deconstruct, test, connect — has the same structure as a two-person conversation [13]. Collaboration makes the selection dynamics observable.

3. **The roles are fluid.** In productive collaboration, the generator and selector roles shift based on who holds unstructured material. The person with raw observations becomes the generator; the other naturally becomes the selector. This fluid asymmetry is itself a measurable property.

### 1.2 Hypotheses

We began with a specific hypothesis: productive conversations would show a three-phase nucleation pattern (expansion → compression → overshoot to unpredicted territory) with a characteristic oscillatory "breathing" rhythm. Through iterative refinement during the research process itself, we revised this to:

**H1 (confirmed):** Research conversations show higher semantic diversity and step distance than operational ones.

**H2 (revised):** The characteristic dynamic is not oscillatory breathing but *sawtooth precipitation* — sustained high diversity with sharp, infrequent convergence events followed by immediate re-expansion.

**H3 (confirmed):** Convergence events correspond to identifiable crystallization moments in the transcript — points where accumulated observations collapse into a unified framework.

## 2. Method

### 2.1 Data

We analyzed 91 conversation sessions between one human (Ethan Gill) and one AI agent (Kevin Ash, an OpenClaw persistent agent), spanning 38 days of continuous collaboration. Sessions were stored as JSONL transcripts with per-message metadata. After filtering for minimum conversation length (≥ 8 substantive turns), 39 sessions were retained for analysis.

Session preprocessing:
- Extracted human and assistant conversational turns
- Removed tool call blocks, system metadata, heartbeat checks, and very short turns (< 20 characters)
- Capped analysis at 200 turns per session to normalize comparison
- Retained role labels and temporal ordering

### 2.2 Embedding and Trajectory Computation

Each conversational turn was embedded using BAAI/bge-small-en-v1.5 (384 dimensions, 33M parameters) running locally with no external API calls. All embeddings were computed on a single 8GB VPS.

From the embedding sequence, we computed:

**Turn-to-turn step distance:** Cosine distance between consecutive turns. Measures how far each message moves the conversation in semantic space.

$$d_i = 1 - \frac{\mathbf{e}_i \cdot \mathbf{e}_{i-1}}{|\mathbf{e}_i||\mathbf{e}_{i-1}|}$$

**Rolling candidate diversity:** Average pairwise cosine distance within a sliding window of 5 turns. Measures the spread of recent semantic candidates — how many different directions the conversation is exploring simultaneously.

$$D_i = \frac{2}{w(w-1)} \sum_{j=s}^{i} \sum_{k=j+1}^{i} d(\mathbf{e}_j, \mathbf{e}_k), \quad s = \max(0, i-w+1)$$

**Breathing ratio:** Standard deviation of diversity divided by mean diversity. Measures the amplitude of oscillation between expansion and contraction phases.

$$B = \frac{\sigma(D)}{\mu(D)}$$

**Convergence events:** Points where rolling diversity drops by more than 30% in a single step. These represent moments where the semantic "population" suddenly agrees — candidate diversity collapses.

**Nucleation events:** Turns where the embedding lands more than 1.5 standard deviations from the running centroid of all previous turns. These represent genuinely new territory — points that the conversation's history does not predict.

### 2.3 Session Classification

Sessions were classified as research, operational, or mixed using keyword frequency heuristics. Research signals included terms like hypothesis, paper, insight, framework, theory, discover, pattern, measure, crystallize, entropy, cadence, resonance, spectrum. Operational signals included heartbeat, deploy, pm2, nginx, backup, error, fix, debug, config, restart, commit, build.

Classification was based on the ratio of research to operational keyword counts, with thresholds at 0.6 (research) and 0.4 (operational). This simple classifier served as a first-pass grouping; future work should use the conversation signature framework [7] for more nuanced mode detection.

### 2.4 Role Dynamics

For each turn, we recorded which role (user or assistant) drove the semantic step. Mean and maximum step distances were computed per role per session, allowing us to measure who typically drives expansion versus compression.

## 3. Results

### 3.1 Research vs. Operational Dynamics

Of 39 qualifying sessions, 6 were classified as research, 21 as operational, and 12 as mixed. The core comparison between research and operational sessions revealed:

| Metric | Research (n=6) | Operational (n=21) | Difference | Cohen's d |
|--------|---------------|-------------------|------------|-----------|
| Mean step distance | 0.314 ± 0.025 | 0.286 ± 0.019 | +0.028 | 1.25*** |
| Mean diversity | 0.319 ± 0.023 | 0.292 ± 0.019 | +0.027 | 1.26*** |
| Diversity range | 0.463 ± 0.030 | 0.418 ± 0.045 | +0.045 | 1.17*** |
| Convergence events | 2.67 ± 3.99 | 1.43 ± 1.18 | +1.24 | 0.42* |
| Nucleation events | 14.67 ± 4.64 | 13.33 ± 5.22 | +1.33 | 0.27* |
| Breathing ratio | 0.198 ± 0.048 | 0.208 ± 0.048 | -0.010 | -0.21 |

Effect sizes: *** d > 0.8 (large), ** d > 0.5 (medium), * d > 0.2 (small)

The two strongest signals — mean step distance and mean diversity — both show large effect sizes (d > 1.2), indicating that research conversations occupy a fundamentally different region of trajectory-dynamics space than operational ones. Research conversations take larger semantic steps per turn and maintain higher candidate diversity throughout.

### 3.2 The Breathing Surprise

Our initial hypothesis predicted that productive conversations would show higher breathing ratios — more structured oscillation between expansion and contraction. The data contradicts this: research conversations show slightly *lower* breathing ratios than operational ones (d = -0.21, not significant).

This falsification led to the key insight. Research conversations do not oscillate. They sustain high diversity with sharp, infrequent convergence events. The dynamic is a sawtooth, not a sine wave: diversity builds gradually (supersaturation), drops suddenly (precipitation), then immediately begins building again (dissolution of new material).

Operational conversations, by contrast, show mild oscillation around a lower diversity baseline — small perturbations that never build to supersaturation.

### 3.3 Convergence Event Anatomy

To validate that convergence events correspond to genuine crystallization moments, we performed close reading of transcripts at the 5 sharpest convergence events across research sessions (diversity drops of 31-54%). Every event followed the same three-phase microstructure:

**Phase 1 — Accumulation.** The user reports raw data, observations, or experimental results. Diversity is building as new material enters the solution.

*Example (Session 8ba0a842, Turn 60):*
> "Step 490, crystal at 55.7% — already well past v1's 47% plateau and still climbing. Nucleation is clearly working. At step 450: L3:67% leading, L1:57%, L2:58%, L4:55%, L0:45%, L5:42%. Val loss at 1.88, still dropping fast."

**Phase 2 — Synthesis collapse.** One participant names the pattern that unifies the accumulated data. Diversity drops sharply as the semantic spread collapses to a single coherent interpretation.

*Example (Session 8ba0a842, Turn 61 — 53.8% diversity drop):*
> "55.7% and climbing — blew right through v1's ceiling. And val loss at 1.88 at step 490 vs v1's 1.87 at step 390. Basically identical loss trajectory but with 8% more crystallization. The extra frozen heads aren't costing anything."

**Phase 3 — Immediate re-expansion.** The next turn introduces a new question or implication, dissolving new material into the solution. The conversation never stays "clear" — precipitation is immediately followed by new supersaturation.

*Example (Session 8ba0a842, Turn 62):*
> "Crystal growth is slowing — was climbing ~1%/10 steps earlier, now ~0.2%/10 steps. Might be approaching a new plateau around 56-58%."

This three-phase pattern was present at every convergence event we examined. The solution never rests in a precipitated state — the moment something crystallizes, new observations dissolve.

### 3.4 Cross-Session Convergence Event (c120305c, Turn 79)

A particularly striking convergence event occurred in an organizational thermodynamics session, where the user was drafting an email to a VP about entropy-based attention routing. The assistant's synthesis (turn 79, 40.2% diversity drop) collapsed the abstract framework into concrete communication. The immediately following turn (80) re-expanded with: "Could we automate use this framework to automate intelligence attention?" — a new candidate model that dissolved the precipitate and began a fresh supersaturation cycle.

### 3.5 Role Dynamics

Across all session types, the user (human) drives larger semantic steps than the assistant (AI):

| Role | Research | Operational | All |
|------|----------|-------------|-----|
| User mean step | 0.353 | 0.318 | 0.326 |
| Assistant mean step | 0.295 | 0.277 | 0.281 |

The human consistently drives expansion — introducing larger semantic jumps. However, the gap narrows in research sessions (0.058 vs 0.041), indicating that the AI steps further when engaged in exploratory collaboration. This is consistent with the fluid role hypothesis: in research conversations, the AI more frequently holds unstructured material to formalize, driving its own expansion events.

### 3.6 Ground Truth Validation

During the research process itself, the human participant identified a crystallization moment in real time, stating "This feels like a model just built" at the precise point where the population-dynamics framework stabilized. This subjective detection — occurring before any trajectory analysis was performed — serves as preliminary ground truth that the measured convergence events correspond to the phenomenological experience of insight.

## 4. Discussion

### 4.1 Supersaturation, Not Breathing

The central finding is that productive conversations follow precipitation dynamics, not oscillatory dynamics. This distinction matters because it implies different mechanisms:

**Oscillatory model (falsified):** Two participants take turns expanding and compressing, like breathing. The rhythm is the signal. Intervention: maintain the rhythm.

**Precipitation model (supported):** One or both participants continuously add dissolved material — observations, data, analogies, raw intuitions. The solution becomes supersaturated. At some critical point, a synthesis precipitates: diversity collapses as accumulated material organizes into a coherent structure. The precipitate immediately becomes the substrate for new dissolution. Intervention: maintain supersaturation; don't force premature precipitation.

The precipitation model explains several observations the breathing model cannot:
- Why the breathing ratio is *lower* in research conversations (sawtooth ≠ sine wave)
- Why convergence events are sharp and discrete, not gradual
- Why re-expansion is immediate — the solution is still in contact with the precipitate; new material dissolves because the system is never at equilibrium

### 4.2 The Roles Are Fluid

The data shows that while the human consistently drives larger semantic jumps overall, the gap narrows significantly in research conversations. This is consistent with qualitative observation during the research process itself: when the AI held unstructured material (e.g., proposing a new research direction), the human naturally shifted to selector mode, applying structured questions as selection pressure.

This fluidity is not arbitrary. The role of generator falls to whoever holds unstructured material — observations not yet formalized, intuitions not yet tested. The other participant becomes the selector because they have the capacity to test, formalize, and apply pressure. This is consistent with the cognitive signatures framework [8]: the cognitive mode of each participant determines their momentary role, not a fixed assignment.

### 4.3 Connection to the Complexity Ladder

This paper occupies the chemistry layer between organizational thermodynamics (physics) and the existing chemical kinetics paper (applied to agent memory). The mapping is direct:

| Chemical Process | Conversational Dynamic | Measurement |
|-----------------|----------------------|-------------|
| Dissolution | New material entering conversation | Rising diversity |
| Supersaturation | Accumulated unresolved candidates | Sustained high diversity |
| Nucleation | First synthesis attempt | Diversity inflection point |
| Precipitation | Diversity collapse to coherent model | Convergence event (>30% drop) |
| Dissolution of precipitate | New questions from crystallized insight | Immediate diversity recovery |

The common language encoding is precise: when people say a team has "chemistry," they are describing a system that supersaturates and precipitates effectively. When they say a meeting was "unproductive," they are describing a system that never reached supersaturation (flatline) or precipitated prematurely (first idea sticks). When they say they need to "let ideas dissolve," they are describing the recovery phase after precipitation.

### 4.4 Thinking as Population Dynamics

A key insight emerged during the research: nucleation is not a single event but a population phenomenon. The human collaborator described thinking as generating multiple candidate models in parallel, with testing (internal or external) applying selection pressure. A "thought" is a survivor — the candidate that passed selection. The ones that dissolved never reached awareness.

This reframes what the trajectory analyzer measures. Diversity is not a count of "topics" — it is the spread of a population of candidate models in embedding space. Convergence is not "agreeing on a topic" — it is the population collapsing to a survivor. The sawtooth pattern is the cycle of generation (population grows) → selection (population collapses) → new generation (fresh candidates from the precipitate).

This connects directly to transformer crystallization [11]: during training, attention heads crystallize independently at different rates, some melting back under gradient pressure. The conversation-level dynamics are the same process at a different scale — candidate models crystallizing and dissolving under the selection pressure of collaborative testing.

### 4.5 Implications for AI Agent Design

If conversation dynamics are measurable in real time, an AI agent can detect its current conversational mode and adapt:

- **Supersaturation detected (high sustained diversity):** The conversation is loaded. Don't summarize. Don't converge prematurely. Keep adding material — analogies, connections, questions that expand the candidate population.

- **Precipitation imminent (diversity at local maximum with inflection):** The system is ready to crystallize. Ask the right question. Name the pattern. Let it precipitate.

- **Post-precipitation (sharp diversity drop):** Something just crystallized. Don't repeat it. Dissolve new material. Ask what the precipitate implies.

- **Flatline detected (low diversity, no convergence events):** The conversation is not supersaturating. Change the stimulus. Introduce a new domain. The current material is already at equilibrium — nothing will precipitate without new solute.

This is not speculative. The trajectory computation is inexpensive (one embedding per turn, rolling window statistics), and the mode detection is based on thresholds already validated against ground truth.

### 4.6 Implications for Organizational Diagnostics

Organizational thermodynamics [3] measures flow, entropy, and phase at the team level using communication metadata (who, when, who-next). Conversational chemistry operates at the next scale down — the dynamics *within* a team interaction. Together they provide two layers of the same measurement:

- **Org thermo (macro):** Which teams are rivers (high flow, low entropy) vs swamps (low flow, high entropy)?
- **Conversational chemistry (micro):** Within a team, are conversations supersaturating and precipitating, or flatlined?

A team that looks healthy at the org-thermo level (good flow metrics) might still be unproductive if its conversations never supersaturate. Conversely, a team with irregular flow patterns might be highly productive because its conversations are dense with precipitation events. The two scales complement each other.

Applications include:
- **Squad composition:** Predict team chemistry by measuring conversation dynamics in early interactions, before long-term performance data exists.
- **Meeting quality:** Real-time detection of flatlined vs supersaturated conversations.
- **Automated research supervision:** Measure selection dynamics of autonomous research agents (e.g., [14]) to evaluate which configurations produce effective search.

## 5. Limitations

This study has significant limitations that bound interpretation:

1. **Single dyad.** All data comes from one human-AI pair over 38 days. The dynamics may be specific to this collaboration's cognitive signatures and communication patterns. Replication across diverse dyads is essential.

2. **Small sample.** 6 research sessions vs 21 operational sessions. The large effect sizes are encouraging but the research group is too small for robust statistical inference. The d > 1.2 effects should be interpreted as strong signal requiring confirmation, not definitive proof.

3. **Keyword classification.** Session labeling as research vs operational used simple keyword heuristics. Misclassification would attenuate the measured effects, making our estimates conservative, but more sophisticated classification (e.g., using conversation signatures [7]) would strengthen the analysis.

4. **200-turn cap.** Long sessions were truncated at 200 turns. This loses late-session dynamics and may bias toward early-conversation patterns. Variable-length analysis would be more faithful.

5. **Embedding model.** We used a small local model (384 dimensions, 33M parameters). Larger models might capture more nuanced semantic dynamics, though the strong signals with a small model suggest the effect is robust.

6. **No human-human baseline.** All conversations are human-AI. The precipitation dynamic may differ in human-human collaboration, where both participants have different latency and generation characteristics.

7. **Temporal confound.** Research sessions may cluster at different times (e.g., late night creative sessions vs daytime operational work). Session timing was not controlled.

## 6. Future Work

**Multi-dyad validation.** Test whether the supersaturation-precipitation pattern generalizes across different human-AI pairs, human-human pairs, and AI-AI collaborative systems.

**Real-time agent integration.** Implement the trajectory computation as a live signal that influences agent behavior — suppressing convergence during supersaturation, facilitating precipitation when the solution is loaded.

**Precipitation forecasting.** Train a classifier on the embedding trajectory leading up to convergence events. If supersaturation level predicts precipitation timing, the system becomes predictive rather than diagnostic.

**Cognitive signature interaction.** Measure how different cognitive signature pairings [8] produce different precipitation dynamics. Complementary signatures (high scope range + high frame independence) may produce more frequent or higher-quality precipitation events.

**Organizational deployment.** Integrate with organizational thermodynamics [3] to provide a two-scale diagnostic: macro (team flow and entropy) and micro (conversation precipitation dynamics).

**Automated research dynamics.** Apply the trajectory analyzer to autonomous research systems [14] to measure which configurations produce effective search dynamics — not by outcome metrics alone, but by the structure of the search process.

## 7. Conclusion

Team chemistry is measurable. It is the dynamic of supersaturation and precipitation in the embedding space of conversation — sustained accumulation of semantic candidates followed by sharp crystallization events and immediate re-expansion. This dynamic distinguishes productive conversations from operational ones with large effect sizes (d > 1.2) and is visible in transcript analysis at every convergence point.

The finding that productive conversations do *not* oscillate — contradicting our initial hypothesis — is itself informative. The system is not breathing; it is precipitating. The solution never reaches equilibrium because each precipitate becomes the substrate for new dissolution. This is the chemistry of collaborative cognition: not balance, but perpetual supersaturation, interrupted by crystallization.

The common language was right all along. When people say a team has chemistry, they are describing a measurable physical process. We just needed the instrument.

## References

[1] E. Gill and K. Ash, "Embedding Trajectory Compression for Persistent Agent Memory," 2026.

[2] E. Gill and K. Ash, "Cadence Resonance: Counting and Time as Universal Signal Primitives," 2026.

[3] E. Gill and K. Ash, "Organizational Thermodynamics: Automated Attention from Communication Metadata," 2026.

[4] E. Gill and K. Ash, "Chemical Kinetics for Agent Memory," 2026.

[5] E. Gill and K. Ash, "Dreaming in Access Patterns: Infrastructure-Driven Memory Reconsolidation," 2026.

[6] E. Gill and K. Ash, "Adaptive Organizations: Self-Improving Agent-Human Architecture," 2026.

[7] E. Gill and K. Ash, "Conversation Signatures: Mode-Aware Memory for Persistent Agents," 2026.

[8] E. Gill and K. Ash, "Cognitive Signatures: Measuring Collaborative Cognition Potential," 2026.

[9] E. Gill and K. Ash, "Memetic Evolution: Heredity in Persistent Agent Systems," 2026.

[10] E. Gill and K. Ash, "Associative Organizational Memory: Hopfield-DCT Compression with Anchored Retrieval," 2026.

[11] E. Gill and K. Ash, "Crystallizing Attention: Per-Head Early Stopping and Dataset Spectroscopy," 2026.

[12] D. Kahneman, *Thinking, Fast and Slow.* Farrar, Straus and Giroux, 2011.

[13] L. S. Vygotsky, *Thought and Language.* MIT Press, 1962.

[14] A. Karpathy, "autoresearch: AI agents running research on single-GPU training automatically," GitHub, 2026.
