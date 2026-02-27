/**
 * Entropy — Shannon entropy per node.
 *
 * Tracks outcome distribution per entity and normalizes to 0-1.
 * H(node) = -Σ p(outcome) × log2(p(outcome)), normalized by log2(num_outcomes)
 */

import type { InteractionEvent, EntropyResult, EntityEntropy, ThreadOutcome } from './types';

const ALL_OUTCOMES: ThreadOutcome[] = ['resolved', 'escalated', 'expanded', 'stalled', 'looped'];
const STALL_DAYS = 7;

/** Classify a thread's outcome from its events. */
function classifyThread(events: InteractionEvent[]): ThreadOutcome {
  const sorted = [...events].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  const hasClose = sorted.some(e => e.event_type === 'close' || e.event_type === 'merge');

  if (hasClose) return 'resolved';

  // Unique participants
  const participants = new Set(sorted.map(e => e.from));

  // Check for expansion (3+ participants)
  if (participants.size >= 3) return 'expanded';

  // Check for loop (back-and-forth between 2 people, 4+ exchanges)
  if (participants.size === 2 && sorted.length >= 4) {
    let alternations = 0;
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i].from !== sorted[i - 1].from) alternations++;
    }
    if (alternations >= 3) return 'looped';
  }

  // Check for escalation (assigns to new people)
  const assignEvents = sorted.filter(e => e.event_type === 'assign');
  if (assignEvents.length >= 2) return 'escalated';

  // Check for stall — only if thread is old enough
  const first = sorted[0];
  const last = sorted[sorted.length - 1];
  const threadAgeDays = (Date.now() - new Date(first.timestamp).getTime()) / (1000 * 60 * 60 * 24);
  const daysSinceLast = (Date.now() - new Date(last.timestamp).getTime()) / (1000 * 60 * 60 * 24);

  if (threadAgeDays > STALL_DAYS && daysSinceLast > STALL_DAYS) return 'stalled';

  // Young thread still in progress — don't classify yet, treat as expanded if multi-party
  // or stalled as conservative default for single-party young threads
  return participants.size > 1 ? 'expanded' : 'stalled';
}

/** Compute Shannon entropy from outcome counts, normalized to 0-1. */
function shannonEntropy(counts: Record<ThreadOutcome, number>): number {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  if (total === 0) return 0;

  const nonZero = ALL_OUTCOMES.filter(o => counts[o] > 0);
  if (nonZero.length <= 1) return 0;

  let h = 0;
  for (const outcome of nonZero) {
    const p = counts[outcome] / total;
    h -= p * Math.log2(p);
  }

  const maxH = Math.log2(ALL_OUTCOMES.length);
  return maxH > 0 ? h / maxH : 0;
}

function emptyOutcomes(): Record<ThreadOutcome, number> {
  return { resolved: 0, escalated: 0, expanded: 0, stalled: 0, looped: 0 };
}

export function computeEntropy(events: InteractionEvent[]): EntropyResult {
  if (events.length === 0) return { by_author: [], by_label: [] };

  // Group by thread, classify each
  const threads = new Map<string, InteractionEvent[]>();
  for (const e of events) {
    const arr = threads.get(e.thread_id);
    if (arr) arr.push(e);
    else threads.set(e.thread_id, [e]);
  }

  const threadOutcomes = new Map<string, ThreadOutcome>();
  for (const [tid, tevents] of threads) {
    threadOutcomes.set(tid, classifyThread(tevents));
  }

  // Per-author entropy
  const authorThreads = new Map<string, Set<string>>();
  for (const e of events) {
    const s = authorThreads.get(e.from);
    if (s) s.add(e.thread_id);
    else authorThreads.set(e.from, new Set([e.thread_id]));
  }

  const by_author: EntityEntropy[] = [];
  for (const [author, tids] of authorThreads) {
    const counts = emptyOutcomes();
    for (const tid of tids) {
      const outcome = threadOutcomes.get(tid);
      if (outcome) counts[outcome]++;
    }
    by_author.push({
      entity: author,
      entity_type: 'author',
      entropy: shannonEntropy(counts),
      outcome_counts: counts,
    });
  }

  // Per-label entropy
  const labelThreads = new Map<string, Set<string>>();
  for (const e of events) {
    for (const label of e.labels ?? []) {
      const s = labelThreads.get(label);
      if (s) s.add(e.thread_id);
      else labelThreads.set(label, new Set([e.thread_id]));
    }
  }

  const by_label: EntityEntropy[] = [];
  for (const [label, tids] of labelThreads) {
    const counts = emptyOutcomes();
    for (const tid of tids) {
      const outcome = threadOutcomes.get(tid);
      if (outcome) counts[outcome]++;
    }
    by_label.push({
      entity: label,
      entity_type: 'label',
      entropy: shannonEntropy(counts),
      outcome_counts: counts,
    });
  }

  return {
    by_author: by_author.sort((a, b) => b.entropy - a.entropy),
    by_label: by_label.sort((a, b) => b.entropy - a.entropy),
  };
}
