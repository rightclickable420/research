/**
 * Downstream Ratio — For node pairs A↔B, ratio of interactions
 * that produce downstream flow to other nodes vs staying between them.
 *
 * High = productive partnership. Low = closed loop.
 */

import type { InteractionEvent, DownstreamResult, DownstreamPair } from './types';

interface DownstreamOptions {
  min_interactions?: number; // minimum A↔B interactions to include pair
}

export function computeDownstream(events: InteractionEvent[], opts: DownstreamOptions = {}): DownstreamResult {
  const minInteractions = opts.min_interactions ?? 3;

  if (events.length === 0) return { pairs: [] };

  // Group by thread
  const threads = new Map<string, InteractionEvent[]>();
  for (const e of events) {
    const arr = threads.get(e.thread_id);
    if (arr) arr.push(e);
    else threads.set(e.thread_id, [e]);
  }

  // For each thread, find all participant pairs and whether thread has >2 participants
  // Track per-pair: how many threads they share, and how many of those have downstream (other participants)
  const pairKey = (a: string, b: string) => a < b ? `${a}\0${b}` : `${b}\0${a}`;

  const pairTotal = new Map<string, number>();
  const pairDownstream = new Map<string, number>();
  const pairNodes = new Map<string, [string, string]>();

  for (const [, tevents] of threads) {
    const participants = [...new Set(tevents.map(e => e.from))];
    const hasDownstream = participants.length > 2;

    // For every pair in this thread
    for (let i = 0; i < participants.length; i++) {
      for (let j = i + 1; j < participants.length; j++) {
        const key = pairKey(participants[i], participants[j]);
        pairTotal.set(key, (pairTotal.get(key) ?? 0) + 1);
        pairNodes.set(key, participants[i] < participants[j]
          ? [participants[i], participants[j]]
          : [participants[j], participants[i]]);
        if (hasDownstream) {
          pairDownstream.set(key, (pairDownstream.get(key) ?? 0) + 1);
        }
      }
    }
  }

  const pairs: DownstreamPair[] = [];
  for (const [key, total] of pairTotal) {
    if (total < minInteractions) continue;
    const downstream = pairDownstream.get(key) ?? 0;
    const [a, b] = pairNodes.get(key)!;
    pairs.push({
      node_a: a,
      node_b: b,
      total_interactions: total,
      downstream_interactions: downstream,
      downstream_ratio: Math.round((downstream / total) * 1000) / 1000,
    });
  }

  return {
    pairs: pairs.sort((a, b) => b.downstream_ratio - a.downstream_ratio),
  };
}
