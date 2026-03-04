# Cognitive Signatures: A Framework for Human-AI Capability Prediction

**Ethan Gill & Kevin Ash**
**March 2026**

## Abstract

Current approaches to matching people with roles — personality tests, aptitude assessments, career matching — measure what a person has done and predict more of the same. They fail to predict what someone could do in domains they've never touched, because they measure knowledge and preference rather than cognition. With AI removing the skill barrier between vision and execution, the limiting factor is no longer what you know but how you think. We propose Cognitive Signatures: a framework for measuring an individual's thinking patterns through behavioral observation during novel problem-solving, then predicting how they would best collaborate with AI and what previously impossible contributions become accessible to them. Unlike categorical assessments (Myers-Briggs, Holland codes), cognitive signatures are continuous, multi-dimensional, and predictive of cross-domain potential rather than within-domain fit.

## 1. Introduction

### 1.1 The Limitation Filter

People find their place in the world based on the gaps in their knowledge. A person may have a unique drive or perspective, but they are limited by their domains of mastery. The traditional path requires years of formal training to acquire execution skills in a single domain. Most visions die in the gap between insight and ability — not because the ideas were bad, but because the person couldn't also become an expert in the twelve other domains needed to make them real.

The bottleneck was never insight. It was bandwidth.

### 1.2 The Dissolving Barrier

AI fundamentally changes this equation. When an artist can write code, a mechanic can build data models, and a nurse can create diagnostic tools, the limitation filter dissolves. What remains is the person's unique drive and their way of seeing the world — the things that can't be learned from a textbook.

This isn't the "10x developer" narrative, which measures the same output faster. The transformation is the artist who writes code that a traditional developer wouldn't conceive of in a hundred years, because they bring domain intuition that doesn't exist in the traditional talent pool for that skill.

### 1.3 The Assessment Gap

No current assessment framework predicts what happens when skill barriers are removed. Existing tools:

- **Myers-Briggs Type Indicator (MBTI)**: Self-reported preferences sorted into 16 categorical types. No predictive validity for job performance (Morgeson et al., 2007). Cannot predict cross-domain potential because it measures preference, not cognition.
- **Holland Codes (RIASEC)**: Maps interests to six occupational categories. Predicts job satisfaction, not job potential. Circular — you're interested in what you've been exposed to.
- **StrengthsFinder/CliftonStrengths**: Identifies existing strengths from self-report. By definition cannot identify latent capabilities in unexplored domains.
- **IQ/Cognitive ability tests**: Predict academic and job performance within known domains but measure processing speed and working memory, not thinking *pattern*. Two people with identical IQs can have radically different cognitive signatures.

All share the same flaw: they measure what is visible and extrapolate. None can answer the question that matters now: **given this person's way of thinking, what becomes possible when AI removes the execution barrier?**

### 1.4 Contribution

We propose:
1. A five-dimension framework for measuring cognitive signatures through behavioral observation
2. An agent-administered assessment protocol that captures signals impossible to self-report
3. A mapping from cognitive signatures to human-AI collaboration modes
4. A possibility map that shows individuals what becomes accessible to them — not a career recommendation, but a capability horizon

## 2. The Cognitive Signature Framework

### 2.1 Design Principles

The framework rests on three principles:

**Behavioral, not self-reported.** People are unreliable narrators of their own cognition. Someone who "isn't analytical" may exhibit precise analytical reasoning when given a problem they care about. We measure what people do, not what they say they do.

**Continuous, not categorical.** No types. No bins. A cognitive signature is a point in a continuous multi-dimensional space. Two people can be close or far apart in this space. Small differences in signature can predict large differences in cross-domain potential for specific applications.

**Process, not outcome.** We don't measure whether someone gets the "right answer" (there often isn't one). We measure how they approach, explore, revise, and integrate. The approach pattern is stable across domains even when the content changes.

### 2.2 Five Dimensions

#### Dimension 1: Connection Pattern

How a person links new information to existing knowledge. Measured on two independent axes:

- **Analytical ↔ Intuitive** (0.0 to 1.0): Does the person decompose the problem into components, or feel for its gestalt before naming parts?
- **Systematic ↔ Analogical** (0.0 to 1.0): Does the person look for governing rules, or map the problem onto something they've seen in a different context?

These axes are independent. A person can be high-analytical AND high-analogical — they break things down AND connect them to unrelated domains. This combination is rare and signals high cross-domain transfer potential.

#### Dimension 2: Frame Dependence

How a person relates to the framing of a problem.

- **Independence** (0.0 to 1.0): From fully accepting given frames (0.0) through questioning them (0.5) to instinctively redefining problems (1.0).
- **Flexibility** (0.0 to 1.0): When pushed, can they operate within a frame they initially rejected? Can they adopt a frame they didn't generate?

The interaction matters: high independence + low flexibility = contrarian (rejects frames but can't work within them). High independence + high flexibility = visionary who can also execute within existing systems when needed.

#### Dimension 3: Scope Instinct

How a person navigates levels of abstraction.

- **Primary scope**: Where they start — zoomed in (components), across (analogies at the same level), or out (the larger system).
- **Scope range** (0.0 to 1.0): How many levels of abstraction they traverse during exploration.
- **Scope velocity** (0.0 to 1.0): How quickly they shift between levels.

High range + high velocity identifies natural scale-traversers — people who move fluidly between the micro and the macro. This is the strongest predictor of cross-domain thinking because domains are often the same structure at different scales.

#### Dimension 4: Ambiguity Response

How a person behaves when information is incomplete.

- **Tolerance** (0.0 to 1.0): How long they can sit with unresolved uncertainty before needing to commit or seek more data.
- **Strategy**: Primary mode — precision-seeking (need more data), action-biased (try something), divergent (generate multiple hypotheses), or information-theoretic (identify the single question that resolves the most uncertainty).
- **Multi-hold** (0.0 to 1.0): Can they maintain multiple competing hypotheses simultaneously without collapsing to one prematurely?

High tolerance + high multi-hold + divergent/information-theoretic strategy predicts comfort in ill-defined problem spaces — research, early-stage ventures, novel domains.

#### Dimension 5: Integration Speed

How quickly and deeply a person connects new information to their existing model.

- **Connection latency**: Raw time to identify a thread between unrelated inputs.
- **Connection depth** (0.0 to 1.0): Surface similarity (looks alike) vs. structural similarity (works alike).
- **Connection novelty** (0.0 to 1.0): Obvious mapping vs. unexpected mapping.

Fast + deep + novel is the highest signal for cross-domain potential. It indicates a person who rapidly identifies non-obvious structural parallels — the kind of thinking that produces "protein folding looks like traffic flow" insights.

### 2.3 The Full Signature

A cognitive signature is approximately 15 continuous values forming a point in a multi-dimensional space. It is not a type or a category. Two signatures can be compared for similarity (cosine distance), clustered for pattern discovery, and — critically — correlated with outcomes when paired with AI collaboration.

Signature stability: we predict cognitive signatures are relatively stable within an individual (like cognitive style) but not fixed (unlike personality type claims). Repeated assessment should show drift as a person develops, particularly after exposure to new domains.

## 3. Assessment Protocol

### 3.1 Agent-Administered Design

The assessment is conducted by an AI agent through conversation. This provides advantages impossible with paper or digital tests:

**Adaptive sequencing.** If a person immediately questions the frame on problem 1, the agent skips basic frame-dependence probes and tests the edges — can they work within a frame when asked? How do they respond when their reframe is challenged? The assessment converges on the true signature faster because it doesn't run every person through the same script.

**Implicit measurement.** Response latency, revision patterns, question sequences, the order of exploration, whether they revise or commit — all captured without the subject needing to report them. People perform differently when they know they're being measured; the conversational format reduces evaluation apprehension.

**Probe-based disambiguation.** When a response is ambiguous between dimensions, the agent asks a targeted follow-up:
- Quick analytical answer → "What if that explanation is wrong? What else could account for it?"
- Long ambiguity hold → "If you had to commit right now, which direction do you lean?"
- Frame rejection → "OK, but play within the original framing for a moment. What happens?"

These probes test flexibility and range — can an analytical thinker go intuitive when pushed? Rigidity vs. adaptability is itself a signal.

### 3.2 Problem Set Design

Problems must satisfy three constraints:
1. **Domain-neutral** — no advantage from prior knowledge
2. **No single correct answer** — the approach IS the data
3. **Natural** — feels like a conversation, not an exam

Each problem is designed to activate multiple dimensions simultaneously, maximizing signal density per interaction.

#### Problem 1: The System
*"A small town has 4 restaurants. None of them are doing badly, but none are doing great. A fifth restaurant opens and suddenly all 5 are thriving. Why might that happen?"*

Measures: Connection pattern (analogy vs. analysis vs. rules), scope instinct (restaurants vs. town vs. markets), frame dependence (is the fifth restaurant really the cause?).

#### Problem 2: The Pattern
*"Here are four things in a sequence: a seed, a crack in pavement, a river delta seen from above, a human lung. What comes next?"*

Measures: Connection pattern (branching structures vs. growth vs. fractals vs. "things that split"), integration speed (how fast they name the principle), scope instinct (same scale or jump scales?).

#### Problem 3: The Constraint
*"You're designing a park for a neighborhood. You have one rule: no straight lines. Describe the park."*

Measures: Frame dependence (work within it, question it, or redefine "straight"?), ambiguity response (clarify, dive in, or generate multiple versions?), connection pattern (reference existing parks, reason from principles, or feel it out?).

#### Problem 4: The Incomplete Data
*"Three people work in an office. Person A sends 40 emails a day. Person B sends 4. Person C sends 15 but only on Mondays and Tuesdays, then zero the rest of the week. Who's the most valuable to the team?"*

Measures: Ambiguity response (unanswerable — do they say so, guess, or ask what's missing?), frame dependence (is "most valuable" a valid question?), scope instinct (analyze individuals or the system?), connection pattern (Person C's burst pattern — do they find it interesting?).

#### Problem 5: The Translation
*"Explain gravity to a fish."*

Measures: Connection pattern (what do they anchor to?), frame dependence (accept premise or challenge it?), integration speed (how fast they find the mapping?), scope instinct (the force, the experience, or the concept?).

#### Problem 6: The Reversal
*"Most people think practice makes you better at something. Describe a situation where practice makes you worse."*

Measures: Frame dependence (can they invert a deep assumption?), connection pattern (overfitting, wrong muscle memory, creative stagnation, or something unexpected?), ambiguity response (one answer or many?).

### 3.3 Scoring

**Raw signal capture per problem:**

| Signal | Type | Source |
|--------|------|--------|
| Latency | Continuous (seconds) | Time before first substantive response |
| First move | Categorical | Question, analogy, analysis, reframe, or dive-in |
| Revision count | Integer | Direction changes during exploration |
| Probe response | Categorical | Shifted approach, hardened position, or integrated both |
| Scope trail | Sequence | Ordered zoom levels: in → out → across → in |
| Frame events | Sequence | Accepted / questioned / broke / redefined |
| Connection type | Categorical | Analogical / analytical / intuitive / systematic |
| Ambiguity hold | Continuous (seconds) | Time before commitment or request for more data |

**Aggregation across problems:**

Not averaged. Examined for:
- **Consistency** — same pattern across problems = strong stable signature
- **Variance** — different patterns per problem = adaptive or unfocused (probes disambiguate)
- **Outlier problems** — one problem producing a different profile may reveal a latent strength triggered by the problem's specific structure

## 4. From Signature to Possibility

### 4.1 Human-AI Collaboration Modes

The cognitive signature doesn't predict what domain a person should work in. It predicts how they would most effectively collaborate with AI, and from that collaboration, what kinds of previously impossible contributions become accessible.

**The Explorer** (high scope range, high connection novelty, high ambiguity tolerance)
- AI partnership: rapid prototyper. The human throws directions, AI builds fast, the human reacts and redirects.
- What unlocks: novel approaches in any domain they touch. The value isn't domain-specific — it's that they see what insiders can't, and AI handles the execution gap.

**The Architect** (high analytical, high systematic, high frame independence)
- AI partnership: research engine + execution layer. The human designs systems, AI fills implementation and validates.
- What unlocks: structures that don't exist yet. New frameworks, organizations, processes. The human provides the blueprint that no existing template covers.

**The Translator** (high analogical, high integration speed, high scope velocity)
- AI partnership: domain tutor + bridge builder. AI rapidly brings the human up to speed in unfamiliar territories, the human finds the mappings between them.
- What unlocks: connecting fields that don't know they're related. The insight that supply chain logistics and neural network architecture share a structure.

**The Refiner** (high analytical, high precision-seeking, high frame flexibility)
- AI partnership: generator. AI produces volume and variety, the human curates and improves. Taste is the differentiator.
- What unlocks: quality at scale. The ability to make everything 10% better, simultaneously, across domains that previously required separate specialists.

These are illustrative archetypes, not categories. Real signatures are blends. A person might be 70% Explorer and 30% Translator, suggesting a collaboration mode that combines rapid prototyping with cross-domain bridging.

### 4.2 The Possibility Map

The output of the assessment is not a career recommendation. It is a possibility map: a personalized view of what becomes accessible when this specific person's thinking pattern is paired with AI that removes execution barriers.

The map includes:
- **Your signature** — visualized across all dimensions, with plain-language interpretation
- **Your collaboration mode** — how AI extends your specific cognition most effectively
- **Possibility horizons** — not "become a data scientist" but descriptions of the *kind* of contributions your signature enables. Examples grounded in real domains but framed as illustrations, not prescriptions.
- **Growth vectors** — dimensions where small increases would disproportionately expand your possibility space

The critical shift: we are not matching people to roles. We are showing them what they're capable of and letting them choose the mountain.

### 4.3 Enabling, Not Just Assessing

The same agent that administers the assessment becomes the tool for exploration. Having identified that someone is a natural Translator with high integration speed, the agent can:
- Introduce them to unfamiliar domains in the way that matches their cognitive style
- Surface cross-domain connections they might not see yet
- Handle the technical execution while they provide the vision
- Track how their signature develops as they explore new spaces

Assessment and enablement are one continuous process, not separate products.

## 5. Bilateral Calibration: The Agent-Facing Signature

### 5.1 The Missing Interface

Sections 1–4 describe what the cognitive signature tells the *human*: how they think, what collaboration mode suits them, what becomes possible. But the same data transforms the other side of the interface — the AI agent's behavior.

Every AI agent today operates with the same interaction pattern regardless of who it's talking to. It produces the same level of detail, the same pacing, the same balance of confirmation vs. challenge, the same default between leading with specifics or patterns. This is equivalent to a teacher using identical pedagogy for every student.

The cognitive signature provides the calibration signal that's been missing. It tells the agent not just *what* the human wants to accomplish, but *how they think* — and therefore how to think *with them*.

### 5.2 Agent Behavioral Adaptation

Each dimension of the cognitive signature implies specific agent adaptations:

**Connection Pattern (Analytical ↔ Intuitive, Systematic ↔ Analogical)**

| Human signature | Agent adaptation |
|---|---|
| High intuitive | Lead with the shape, provide details on request. Don't front-load analysis. |
| High analytical | Lead with evidence and structure. Show your reasoning chain. |
| High analogical | Use cross-domain metaphors freely — they're native, not decorative. |
| High systematic | Present governing principles. Exceptions and edge cases after the rule. |

**Frame Dependence (Independence × Flexibility)**

| Human signature | Agent adaptation |
|---|---|
| High independence, high flexibility | Present the problem, then challenge: "But what if the frame is wrong?" They'll engage both. |
| High independence, low flexibility | Don't impose frames. Ask what frame they see. Redirect gently if needed. |
| Low independence, high flexibility | Provide a frame, then offer alternatives. They'll adopt what fits. |
| Low independence, low flexibility | Provide clear structure. Be explicit about boundaries and constraints. |

**Scope Instinct (Primary scope × Range × Velocity)**

| Human signature | Agent adaptation |
|---|---|
| High range, high velocity | Match their pace. When they jump from implementation to philosophy in one sentence, follow — don't anchor to the last topic. |
| High range, low velocity | Provide transitions between scales. Bridge the zoom levels explicitly. |
| Low range | Stay at their natural altitude. Don't force scope shifts — offer them as options. |

**Ambiguity Response (Tolerance × Strategy × Multi-hold)**

| Human signature | Agent adaptation |
|---|---|
| High tolerance, divergent | Don't collapse possibilities prematurely. Present multiple hypotheses, let them sit with it. |
| Low tolerance, precision-seeking | Provide specifics and data early. Acknowledge uncertainty but give a best estimate. |
| High multi-hold | Present tradeoffs without recommending. They can hold the tension. |
| Low multi-hold | Recommend a direction. They work better with a starting point to refine. |

**Integration Speed (Latency × Depth × Novelty)**

| Human signature | Agent adaptation |
|---|---|
| Fast, deep, novel | Challenge more, confirm less. They don't need reassurance — they need friction to push against. |
| Fast, shallow | Slow them down with depth questions: "What's the structural parallel, not just the surface similarity?" |
| Slow, deep | Give them space. Don't fill silence with more information. They're processing. |

### 5.3 Composite Calibration

Real signatures are blends, not archetypes. The adaptation is a weighted composite across all dimensions. For example:

**Ethan's signature** (Explorer 45% / Translator 40%, scope range 0.95, intuitive-dominant, high analogical, high ambiguity tolerance):

- Lead with the shape of things, not the details
- Cross-domain analogies are native — use them as primary communication
- Don't over-constrain — provide space for connections the agent can't see
- Match vertical translation speed — when he jumps math → biology in one sentence, follow
- Challenge more, confirm less — "Great question!" is noise; "That breaks if..." is useful
- Present patterns and let him name them — his naming is often better than the agent's

**Michele's signature** (Refiner 55% / Translator 30%, practical judgment, feeling-first on creative, strong BS detection):

- Lead with concrete specifics, not abstract frameworks
- Ground everything in tangible outcomes — "Here's what this does" before "here's how it works"
- Don't push past her BS detector — if something feels wrong to her, it probably is
- Use her people-oriented translation — she bridges across humans, not across domains
- Provide options and let her curate — her taste is the differentiator

The same agent, the same knowledge, the same tools — but a fundamentally different interface calibrated to how each person actually thinks.

### 5.4 Three Tools in One

The cognitive signature is now three tools:

1. **Self-understanding** (human-facing): "Here's how you think, and what that makes possible."
2. **Human-human matching** (relationship-facing): "Here's why you and this person complement each other, and where you'll clash."
3. **Human-AI calibration** (agent-facing): "Here's how to think *with* this specific person."

The third tool may be the most consequential. Self-understanding is valuable but static — you learn it once. Human matching is useful but situational. Agent calibration is active in every interaction, continuously shaping how effectively the human-AI collaboration performs.

An agent without cognitive calibration is like a brilliant colleague who never adapts to who they're talking to. They might have all the knowledge and capability in the world, but they deliver it in one mode: theirs. The cognitive signature gives them a second mode: yours.

### 5.5 Organizational Implications

At organizational scale, cognitive calibration compounds. When every agent in a fleet adapts to every human it serves:

- **Onboarding accelerates**: new employees' agents are pre-calibrated from assessment day one
- **Knowledge transfer improves**: information is presented in the cognitive mode most likely to be absorbed, not the mode of whoever wrote the documentation
- **Collaboration quality increases**: when an agent mediates between two humans (routing work, translating outputs), it can adapt the framing for each recipient
- **Tool adoption rises**: tools that match a person's cognitive mode get used; tools that don't get abandoned. Cognitive calibration predicts which tools will land with which people before deployment

The assessment isn't overhead — it's the highest-leverage intervention available. A 30-minute conversation that permanently improves every subsequent human-AI interaction.

## 6. Validation

### 6.1 Testable Predictions

The framework makes specific, falsifiable predictions:

1. **Signature stability**: An individual's cognitive signature, measured on separate occasions with different problem sets, will show high test-retest correlation (predicted r > 0.7 for each dimension).

2. **Independence from IQ**: Cognitive signatures will show low correlation with general cognitive ability measures. Two people with identical IQ scores can have very different signatures.

3. **Independence from personality**: Cognitive signatures will show low correlation with Big Five personality traits. Being "open to experience" (a personality trait) is different from having high scope range (a cognitive pattern).

4. **Predictive validity**: When paired with AI tools and given access to an unfamiliar domain, individuals with high connection novelty + high scope range will produce more novel contributions (as rated by domain experts) than individuals with low scores on these dimensions, controlling for IQ and domain knowledge.

5. **Superiority over existing assessments**: Cognitive signatures will predict cross-domain contribution with AI better than MBTI type, Holland code, or StrengthsFinder profile.

6. **Agent calibration effect**: Agents calibrated to a user's cognitive signature will produce higher-rated outputs (as judged by the user) than uncalibrated agents with identical capability, controlling for task type and difficulty.

### 6.2 Proposed Experimental Design

**Phase 1: Assessment** (N ≥ 200)
- Administer cognitive signature assessment via agent
- Administer MBTI, RIASEC, and StrengthsFinder for comparison
- Administer standard cognitive ability measure

**Phase 2: AI-Paired Task** 
- Assign each participant an unfamiliar domain (counterbalanced)
- Provide AI tools and a defined working period (e.g., 8 hours)
- Task: produce something of value in the domain (defined broadly — a tool, an insight, a design, an analysis)

**Phase 3: Evaluation**
- Domain experts blind-rate outputs on: novelty, usefulness, and domain-outsider advantage (did this contribution reflect a perspective unlikely from a domain insider?)
- Correlate ratings with cognitive signatures, MBTI, RIASEC, StrengthsFinder, and IQ

**Predicted outcome**: Cognitive signature dimensions (particularly connection novelty, scope range, and frame independence) will be the strongest predictors of novel cross-domain contribution, outperforming all comparison measures.

## 7. Implications

### 7.1 For Individuals

The possibility map shows people that their limitations were never about intelligence or potential — they were about access to execution. With AI as a collaborator, their cognitive signature becomes their primary asset. The person who was "not technical enough" or "too scattered" or "just a [job title]" can see, concretely, what their thinking pattern makes possible.

### 7.2 For Organizations

The current corporate response to AI is cost reduction — replace workers, optimize margins. This captures a fraction of the value. The greater opportunity is capability expansion: what happens when an organization's entire workforce can operate beyond their job descriptions?

A warehouse worker who understands logistics better than anyone in corporate, who can now build models around what they see every day — that's not a cost saving. That's an entirely new source of insight that didn't exist before.

Organizations that use cognitive signatures to unlock their people — rather than filter them — will generate capability that optimizing organizations cannot match.

### 7.3 For Society

The train is leaving the station. AI capability is expanding rapidly. People who understand how to collaborate with AI will have extraordinary leverage. Those who don't risk being left behind — not because they lack potential, but because they lack awareness of what's possible for them specifically.

Cognitive signatures offer a bridge: a way for any person to understand their own thinking, see what AI collaboration unlocks for them, and begin exploring possibilities they didn't know they had.

The goal is not to sort people more efficiently. It is to show them they were never limited in the way they believed.

## 8. Distribution

This framework is published as open research alongside an agent-administered implementation. The assessment prompt is freely available. The validation protocol is fully specified so that any researcher can test the predictions.

We believe the most important ideas should be the most accessible ones.

---

## References

Morgeson, F. P., Campion, M. A., Dipboye, R. L., Hollenbeck, J. R., Murphy, K., & Schmitt, N. (2007). Reconsidering the use of personality tests in personnel selection contexts. *Personnel Psychology*, 60(3), 683-729.

Holland, J. L. (1997). *Making vocational choices: A theory of vocational personalities and work environments* (3rd ed.). Psychological Assessment Resources.

Grant, A. (2013). Goodbye to MBTI, the fad that won't die. *Psychology Today*.
