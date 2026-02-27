/**
 * Organizational Thermodynamics — Public API
 *
 * Computes 5 org health metrics from interaction event metadata:
 * Flow, Entropy, Cadence, Downstream, Fan-out
 */

export type {
  InteractionEvent,
  OrgThermoResult,
  FlowResult,
  EntropyResult,
  CadenceResult,
  DownstreamResult,
  FanoutResult,
  EntityFlow,
  EntityEntropy,
  CadencePair,
  DownstreamPair,
  EntityFanout,
  ThreadOutcome,
  OrgQuadrant,
  EntityClassification,
} from './types';

export { issuestoEvents, prsToEvents, commentsToEvents, reviewsToEvents, bulkCommentsToEvents, bulkReviewsToEvents } from './events';
export { computeFlowMetrics } from './flow';
export { computeEntropy } from './entropy';
export { computeCadence } from './cadence-sync';
export { computeDownstream } from './downstream';
export { computeFanout } from './fanout';

import type { InteractionEvent, OrgThermoResult, OrgQuadrant, EntityClassification } from './types';
import { computeFlowMetrics } from './flow';
import { computeEntropy } from './entropy';
import { computeCadence } from './cadence-sync';
import { computeDownstream } from './downstream';
import { computeFanout } from './fanout';

/**
 * Run all 5 organizational thermodynamics metrics on a set of interaction events.
 */
export function computeOrgThermo(events: InteractionEvent[]): OrgThermoResult {
  return {
    flow: computeFlowMetrics(events),
    entropy: computeEntropy(events),
    cadence: computeCadence(events),
    downstream: computeDownstream(events),
    fanout: computeFanout(events),
  };
}

/**
 * Classify entities into the Flow × Entropy 2×2 quadrant.
 *
 * Uses median flow_rate and median entropy as thresholds (data-driven,
 * not arbitrary cutoffs). Each entity lands in one of:
 * - river (high flow, low entropy)
 * - waterfall (high flow, high entropy)
 * - bottleneck (low flow, low entropy)
 * - swamp (low flow, high entropy)
 */
export function classifyEntities(result: OrgThermoResult): EntityClassification[] {
  // Build a map of entity → flow_rate
  const flowMap = new Map<string, number>();
  for (const ef of result.flow.by_author) {
    flowMap.set(ef.entity, ef.flow.flow_rate);
  }

  // Build a map of entity → entropy
  const entropyMap = new Map<string, number>();
  for (const ee of result.entropy.by_author) {
    entropyMap.set(ee.entity, ee.entropy);
  }

  // Only classify entities that appear in both AND have meaningful activity
  // Filter out one-shot contributors (zero flow AND zero entropy)
  const entities = [...flowMap.keys()].filter(e => {
    if (!entropyMap.has(e)) return false;
    const fr = flowMap.get(e)!;
    const ent = entropyMap.get(e)!;
    return fr > 0 || ent > 0; // must have at least some measurable activity
  });
  if (entities.length === 0) return [];

  // Use log-scale for flow (highly skewed distribution) and linear for entropy (0-1 bounded)
  const logFlow = (v: number) => Math.log1p(v); // log(1+x) to handle 0
  const flowValues = entities.map(e => logFlow(flowMap.get(e)!)).sort((a, b) => a - b);
  const entropyValues = entities.map(e => entropyMap.get(e)!).sort((a, b) => a - b);

  const median = (arr: number[]) => {
    const mid = Math.floor(arr.length / 2);
    return arr.length % 2 === 0 ? (arr[mid - 1] + arr[mid]) / 2 : arr[mid];
  };

  const flowMedian = median(flowValues);
  const entropyMedian = median(entropyValues);

  return entities.map(entity => {
    const fr = flowMap.get(entity)!;
    const ent = entropyMap.get(entity)!;
    const logFr = logFlow(fr);

    let quadrant: OrgQuadrant;
    if (logFr >= flowMedian && ent < entropyMedian) quadrant = 'river';
    else if (logFr >= flowMedian && ent >= entropyMedian) quadrant = 'waterfall';
    else if (logFr < flowMedian && ent < entropyMedian) quadrant = 'bottleneck';
    else quadrant = 'swamp';

    return {
      entity,
      entity_type: 'author' as const,
      quadrant,
      flow_rate: fr,
      entropy: ent,
    };
  });
}
