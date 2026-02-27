/**
 * Flow — Little's Law metrics per entity.
 *
 * throughput = resolved threads per window
 * inventory  = open threads at window end
 * flow_rate  = throughput / inventory
 */

import type { InteractionEvent, FlowResult, FlowMetrics, EntityFlow } from './types';

interface FlowOptions {
  window_start?: string; // ISO, default: earliest event
  window_end?: string;   // ISO, default: latest event
}

/** Compute flow for a set of threads (grouped events). */
function computeFlow(
  threadGroups: Map<string, InteractionEvent[]>,
  windowStart: number,
  windowEnd: number,
): FlowMetrics {
  let resolved = 0;
  let open = 0;

  for (const events of threadGroups.values()) {
    const inWindow = events.filter(e => {
      const t = new Date(e.timestamp).getTime();
      return t >= windowStart && t <= windowEnd;
    });
    if (inWindow.length === 0) continue;

    const hasClose = inWindow.some(e => e.event_type === 'close' || e.event_type === 'merge');
    if (hasClose) {
      resolved++;
    } else {
      open++;
    }
  }

  return {
    throughput: resolved,
    inventory: open,
    flow_rate: open > 0 ? resolved / open : 0,
  };
}

/** Group events by thread_id. */
function groupByThread(events: InteractionEvent[]): Map<string, InteractionEvent[]> {
  const map = new Map<string, InteractionEvent[]>();
  for (const e of events) {
    const arr = map.get(e.thread_id);
    if (arr) arr.push(e);
    else map.set(e.thread_id, [e]);
  }
  return map;
}

/** Compute per-entity flow (author or label). */
function perEntityFlow(
  events: InteractionEvent[],
  entityType: 'author' | 'label',
  windowStart: number,
  windowEnd: number,
): EntityFlow[] {
  const buckets = new Map<string, InteractionEvent[]>();

  for (const e of events) {
    const keys = entityType === 'author'
      ? [e.from]
      : (e.labels ?? []);

    for (const key of keys) {
      const arr = buckets.get(key);
      if (arr) arr.push(e);
      else buckets.set(key, [e]);
    }
  }

  const results: EntityFlow[] = [];
  for (const [entity, entityEvents] of buckets) {
    const threads = groupByThread(entityEvents);
    results.push({
      entity,
      entity_type: entityType,
      flow: computeFlow(threads, windowStart, windowEnd),
    });
  }

  return results.sort((a, b) => b.flow.throughput - a.flow.throughput);
}

export function computeFlowMetrics(events: InteractionEvent[], opts: FlowOptions = {}): FlowResult {
  if (events.length === 0) {
    return {
      overall: { throughput: 0, inventory: 0, flow_rate: 0 },
      by_author: [],
      by_label: [],
      window_start: opts.window_start ?? new Date().toISOString(),
      window_end: opts.window_end ?? new Date().toISOString(),
    };
  }

  const timestamps = events.map(e => new Date(e.timestamp).getTime());
  const windowStart = opts.window_start ? new Date(opts.window_start).getTime() : Math.min(...timestamps);
  const windowEnd = opts.window_end ? new Date(opts.window_end).getTime() : Math.max(...timestamps);

  const threads = groupByThread(events);
  const overall = computeFlow(threads, windowStart, windowEnd);

  return {
    overall,
    by_author: perEntityFlow(events, 'author', windowStart, windowEnd),
    by_label: perEntityFlow(events, 'label', windowStart, windowEnd),
    window_start: new Date(windowStart).toISOString(),
    window_end: new Date(windowEnd).toISOString(),
  };
}
