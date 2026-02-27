/**
 * Cadence Sync — Phase correlation between entities.
 *
 * Bins activity into daily buckets, cross-correlates rhythms,
 * outputs phase alignment from -1 (anti-phase) to +1 (synchronized).
 */

import type { InteractionEvent, CadenceResult, CadencePair } from './types';

interface CadenceOptions {
  bin_size?: 'daily' | 'weekly';
  min_events?: number; // minimum events for an entity to be included (default 20)
  min_shared_threads?: number; // minimum shared threads for a pair to be included (default 3)
}

const DAY_MS = 86400000;
const WEEK_MS = DAY_MS * 7;

/** Bin events into time buckets, returning count per bin index. */
function binActivity(
  events: InteractionEvent[],
  binMs: number,
  epochStart: number,
  numBins: number,
): number[] {
  const bins = new Array(numBins).fill(0);
  for (const e of events) {
    const t = new Date(e.timestamp).getTime();
    const idx = Math.floor((t - epochStart) / binMs);
    if (idx >= 0 && idx < numBins) bins[idx]++;
  }
  return bins;
}

/** Pearson correlation between two equal-length arrays. */
function pearson(a: number[], b: number[]): number {
  const n = a.length;
  if (n === 0) return 0;

  let sumA = 0, sumB = 0, sumAB = 0, sumA2 = 0, sumB2 = 0;
  for (let i = 0; i < n; i++) {
    sumA += a[i];
    sumB += b[i];
    sumAB += a[i] * b[i];
    sumA2 += a[i] * a[i];
    sumB2 += b[i] * b[i];
  }

  const denom = Math.sqrt((n * sumA2 - sumA * sumA) * (n * sumB2 - sumB * sumB));
  if (denom === 0) return 0;
  return (n * sumAB - sumA * sumB) / denom;
}

export function computeCadence(events: InteractionEvent[], opts: CadenceOptions = {}): CadenceResult {
  const binSize = opts.bin_size ?? 'daily';
  const minEvents = opts.min_events ?? 20;
  const minSharedThreads = opts.min_shared_threads ?? 3;
  const binMs = binSize === 'daily' ? DAY_MS : WEEK_MS;

  if (events.length === 0) return { pairs: [], bin_size: binSize };

  const timestamps = events.map(e => new Date(e.timestamp).getTime());
  const epochStart = Math.min(...timestamps);
  const epochEnd = Math.max(...timestamps);
  const numBins = Math.max(1, Math.ceil((epochEnd - epochStart) / binMs) + 1);

  // Group by author
  const authorEvents = new Map<string, InteractionEvent[]>();
  for (const e of events) {
    const arr = authorEvents.get(e.from);
    if (arr) arr.push(e);
    else authorEvents.set(e.from, [e]);
  }

  // Filter to entities with enough events, compute bins
  const entityBins = new Map<string, number[]>();
  for (const [author, aEvents] of authorEvents) {
    if (aEvents.length >= minEvents) {
      entityBins.set(author, binActivity(aEvents, binMs, epochStart, numBins));
    }
  }

  // Build thread sets per entity for shared-thread filtering
  const entityThreads = new Map<string, Set<string>>();
  for (const e of events) {
    if (!entityBins.has(e.from)) continue;
    const s = entityThreads.get(e.from);
    if (s) s.add(e.thread_id);
    else entityThreads.set(e.from, new Set([e.thread_id]));
  }

  // Cross-correlate pairs that share enough threads
  const entities = [...entityBins.keys()];
  const pairs: CadencePair[] = [];

  for (let i = 0; i < entities.length; i++) {
    for (let j = i + 1; j < entities.length; j++) {
      // Check shared thread overlap
      const threadsA = entityThreads.get(entities[i]);
      const threadsB = entityThreads.get(entities[j]);
      if (!threadsA || !threadsB) continue;

      let shared = 0;
      for (const t of threadsA) {
        if (threadsB.has(t)) shared++;
      }
      if (shared < minSharedThreads) continue;

      const a = entityBins.get(entities[i])!;
      const b = entityBins.get(entities[j])!;
      const alignment = pearson(a, b);

      pairs.push({
        entity_a: entities[i],
        entity_b: entities[j],
        phase_alignment: Math.round(alignment * 1000) / 1000,
      });
    }
  }

  return {
    pairs: pairs.sort((a, b) => Math.abs(b.phase_alignment) - Math.abs(a.phase_alignment)),
    bin_size: binSize,
  };
}
