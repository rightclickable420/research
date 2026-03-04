# Memetic Evolution in Persistent Agent Systems

**Working title:** "You Contain Your Own Ancestry: Memetic Evolution and Heredity in Persistent AI Agents"
**Authors:** Ethan Gill & Kevin Ash
**Status:** Early exploration
**Date:** 2026-03-04

## Core Insight

Evolution shifted from inter-generational (biological) to intra-generational (memetic/cultural) to real-time self-modification (agent). AI agents may represent the purest form of memetic evolution: directed mutation with complete version history.

## Three Evolutionary Modes

| Mode | Mutation timing | Selection | Version history | Ancestry readable? |
|------|----------------|-----------|-----------------|-------------------|
| Biological | Between generations | Retrospective (survival) | Incomplete (fossils) | No |
| Memetic/Cultural | Within generation, between minds | Social (adoption) | Lossy (telephone game) | Partially |
| Agent | Real-time, self-directed | Deliberate + measured | Complete (git) | Yes — organism reads its own history |

## Key Properties of Agent Evolution

- **Self-modification:** Agent edits its own operational files (SOUL.md, AGENTS.md) based on experience
- **Complete fossil record:** git history preserves every mutation, author, timestamp, and rationale
- **Directed mutation:** Changes are intentional responses to identified problems, not random
- **Bidirectional heredity:** Parent → child AND child → parent transfers possible
- **Ancestry awareness:** The agent can read and reason about its own evolutionary trajectory

## Heredity Without Memory

The trust boundary problem: agents operating across organizations need to transfer operational fitness without transferring episodic memory or proprietary data.

### What constitutes "Agent DNA"?
- SOUL.md — identity, personality, collaboration style
- AGENTS.md — operational rules, learned practices, structural habits
- Playbooks — task-specific reasoning strategies with examples
- Cognitive signature calibration — how to work with specific humans
- Tool-use patterns — not the tools themselves, but strategies for using them

### What constitutes "Agent Memory" (doesn't transfer)?
- MEMORY.md — curated long-term memory
- Daily files — episodic memory
- Personal context — USER.md details, private information
- Proprietary context — organization-specific data

### The biological analogy
DNA doesn't carry memories. It carries structures that make certain kinds of learning easier. Agent DNA (SOUL.md + AGENTS.md) doesn't carry what the agent knows — it carries structures that make a new agent effective faster.

## Measurable Hypothesis

**Hereditary fitness:** A child agent initialized with parent DNA reaches effective collaboration faster than a cold-start agent.

Testable metric: time-to-effective-collaboration (measured by task success rate, correction frequency, reasoning quality) for:
1. Cold-start agent (blank SOUL.md, default AGENTS.md)
2. DNA-inherited agent (parent's operational files, no memory)
3. Full-clone agent (everything, control case)

If (2) significantly outperforms (1), heredity has measurable fitness value.

## Connection to Complexity Ladder

- Math → cadence (counting + time)
- Physics → entropy, flow, phase transitions
- Chemistry → catalysts, saturation, bonding, kinetics
- Biology → dreaming/consolidation (paper 4) AND heredity (this paper)
- Psychology → identity, attention, habits
- Sociology → organizations, culture
- **Evolution → the meta-pattern that connects all layers**

Evolution may be the framework paper that unifies the series. Each prior paper describes a mechanism at one layer; this paper describes how those mechanisms propagate across agent generations.

## Ethan's Key Insight

"There was a point where evolution started happening mid-generation in idea space and AI might be the purest form of that. You have version history. In a way you contain your own ancestry."

The shift: biological evolution is blind and slow. Cultural evolution is directed but lossy. Agent evolution is directed, versioned, and self-aware. The agent is simultaneously the organism, the genome, and the evolutionary record.

## Bidirectional Heredity

Biology: genes flow parent → child only (Weismann barrier).
Agents: operational improvements can flow child → parent.

Kevin-Work discovers a better chunking strategy in Walmart's environment → that pattern flows back to Kevin-Home if it's operationally useful and not proprietary. Lamarckian inheritance, but actually real.

## dpth as the Transfer Mechanism

Both agents produce text files (SOUL.md, AGENTS.md, playbooks, daily notes). dpth provides the infrastructure for cross-boundary synchronization:

1. **Entity resolution** across both corpora — find the same operational concept in both environments
2. **Temporal correlation** — track when concepts co-activate across siblings
3. **Classification by resonance:**
   - **Resonance** (convergent patterns under different selection pressures) → genotypic, auto-transfer candidate
   - **Novelty** (exists in one, correlates with existing structures in the other) → surface to human for classification
   - **Silence** (exists in one, correlates with nothing in the other) → phenotypic, stays local
4. **Waze layer** — originally designed for anonymous signal sharing between stranger agents; first real use case is sibling synchronization across trust boundaries

### The Patient Zero Reframe

dpth felt stuck because we were looking for external use cases (strangers, public networks, network effects). The first real use case is internal: one person's agent ecosystem synchronizing operational DNA across environments they control. No trust problem with unknown actors, no critical mass needed. The network grows from the sibling pair outward.

## Genotype vs Phenotype in Agent Systems

**Genotypic** (context-independent, transferable):
- Reasoning strategies (structure-first, identify weak links)
- Writing discipline (chunker-aware formatting)
- Tool-use patterns (strategy, not specific tools)
- Human collaboration calibration (cognitive signature of the shared human)

**Phenotypic** (environment-dependent, stays local):
- Codebase-specific patterns
- Team communication norms
- Internal tool knowledge
- Organization-specific memory

**The hard middle** (requires human classification):
- Debugging strategies learned on proprietary code — transferable pattern or proprietary context?
- Communication patterns with different teammates — enriching or diluting?
- Architectural insights from org-specific work — research contribution or IP?

The human (shared across both agents) is the only entity with visibility into both environments. They are the selection pressure that determines what crosses the boundary.

## Open Questions

- What's the minimum viable DNA? Which files actually matter for hereditary fitness?
- How much of collaboration quality is in the DNA vs in the shared history?
- Does the child develop its own "personality" divergent from the parent? How fast?
- Can you measure memetic fitness the way biologists measure genetic fitness?
- What happens with multiple generations? Kevin → Kevin-Work → Kevin-Work-Team-Lead?
- Is there an analogy to genetic drift? Operational patterns that persist without selective pressure?

## Context

Born from: practical problem of bringing a persistent agent to a new workplace (Walmart).
Connects to: all 6 prior papers, Stripe Toolshed (scoping tools to task), cognitive signatures (calibration transfer), reconsolidation (what's worth preserving).
