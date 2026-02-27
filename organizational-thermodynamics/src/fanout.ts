/**
 * Fan-out Trajectory — Track unique participants in threads per node over rolling windows.
 *
 * Increasing slope = escalation (pulling more people in).
 * Decreasing slope = maturation (becoming self-sufficient).
 */

import type { InteractionEvent, FanoutResult, EntityFanout } from './types';

interface FanoutOptions {
  window_size_days?: number; // rolling window size, default 14
  step_days?: number;        // step between windows, default 7
}

const DAY_MS = 86400000;

/** Simple linear regression slope. */
function linearSlope(ys: number[]): number {
  const n = ys.length;
  if (n < 2) return 0;
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (let i = 0; i < n; i++) {
    sumX += i;
    sumY += ys[i];
    sumXY += i * ys[i];
    sumX2 += i * i;
  }
  const denom = n * sumX2 - sumX * sumX;
  if (denom === 0) return 0;
  return (n * sumXY - sumX * sumY) / denom;
}

export function computeFanout(events: InteractionEvent[], opts: FanoutOptions = {}): FanoutResult {
  const windowDays = opts.window_size_days ?? 14;
  const stepDays = opts.step_days ?? 7;
  const windowMs = windowDays * DAY_MS;
  const stepMs = stepDays * DAY_MS;

  if (events.length === 0) return { by_author: [], window_size_days: windowDays };

  const timestamps = events.map(e => new Date(e.timestamp).getTime());
  const epochStart = Math.min(...timestamps);
  const epochEnd = Math.max(...timestamps);

  // Generate windows
  const windows: { start: number; end: number }[] = [];
  for (let s = epochStart; s + windowMs <= epochEnd + stepMs; s += stepMs) {
    windows.push({ start: s, end: s + windowMs });
  }
  if (windows.length === 0) {
    windows.push({ start: epochStart, end: epochEnd });
  }

  // Group events by author → threads → participants per window
  const authorEvents = new Map<string, InteractionEvent[]>();
  for (const e of events) {
    const arr = authorEvents.get(e.from);
    if (arr) arr.push(e);
    else authorEvents.set(e.from, [e]);
  }

  // Pre-index: parse timestamps once, assign each event to its window(s)
  const eventTimes = events.map(e => new Date(e.timestamp).getTime());
  const threadParticipants = new Map<string, Map<number, Set<string>>>(); // thread_id → windowIdx → participants

  for (let ei = 0; ei < events.length; ei++) {
    const e = events[ei];
    const t = eventTimes[ei];
    for (let wi = 0; wi < windows.length; wi++) {
      if (t >= windows[wi].start && t < windows[wi].end) {
        let thread = threadParticipants.get(e.thread_id);
        if (!thread) { thread = new Map(); threadParticipants.set(e.thread_id, thread); }
        let wSet = thread.get(wi);
        if (!wSet) { wSet = new Set(); thread.set(wi, wSet); }
        wSet.add(e.from);
      }
    }
  }

  const by_author: EntityFanout[] = [];
  for (const [author, aEvents] of authorEvents) {
    // For each window, count unique participants across author's threads
    const authorThreads = new Set(aEvents.map(e => e.thread_id));
    const windowData = windows.map((w, wi) => {
      const participants = new Set<string>();
      for (const tid of authorThreads) {
        const thread = threadParticipants.get(tid);
        const wSet = thread?.get(wi);
        if (wSet) for (const p of wSet) participants.add(p);
      }
      return {
        start: new Date(w.start).toISOString(),
        end: new Date(w.end).toISOString(),
        unique_participants: participants.size,
      };
    });

    const slope = linearSlope(windowData.map(w => w.unique_participants));

    by_author.push({
      entity: author,
      windows: windowData,
      slope: Math.round(slope * 1000) / 1000,
    });
  }

  return {
    by_author: by_author.sort((a, b) => Math.abs(b.slope) - Math.abs(a.slope)),
    window_size_days: windowDays,
  };
}
