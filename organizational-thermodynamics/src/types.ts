/**
 * Organizational Thermodynamics — Shared Types
 *
 * All computations use ONLY metadata (who, when, thread_id, event_type) — never content.
 */

// ── Input ──

export interface InteractionEvent {
  timestamp: string;       // ISO date
  from: string;           // author/login
  to?: string;            // respondee or assignee
  thread_id: string;      // issue/PR number as grouping key
  event_type: 'open' | 'comment' | 'review' | 'merge' | 'close' | 'assign' | 'label';
  thread_type: 'issue' | 'pr';
  labels?: string[];
}

// ── Outcome classification for entropy ──

export type ThreadOutcome = 'resolved' | 'escalated' | 'expanded' | 'stalled' | 'looped';

// ── Flow (Little's Law) ──

export interface FlowMetrics {
  throughput: number;      // resolved per window
  inventory: number;       // open at end of window
  flow_rate: number;       // throughput / inventory (0 if inventory=0)
}

export interface EntityFlow {
  entity: string;          // author, label, etc.
  entity_type: 'author' | 'label';
  flow: FlowMetrics;
}

export interface FlowResult {
  overall: FlowMetrics;
  by_author: EntityFlow[];
  by_label: EntityFlow[];
  window_start: string;
  window_end: string;
}

// ── Entropy (Shannon) ──

export interface EntityEntropy {
  entity: string;
  entity_type: 'author' | 'label';
  entropy: number;         // 0-1 normalized
  outcome_counts: Record<ThreadOutcome, number>;
}

export interface EntropyResult {
  by_author: EntityEntropy[];
  by_label: EntityEntropy[];
}

// ── Cadence (Phase Correlation) ──

export interface CadencePair {
  entity_a: string;
  entity_b: string;
  phase_alignment: number; // -1 to +1
}

export interface CadenceResult {
  pairs: CadencePair[];
  bin_size: 'daily' | 'weekly';
}

// ── Downstream Ratio ──

export interface DownstreamPair {
  node_a: string;
  node_b: string;
  total_interactions: number;
  downstream_interactions: number;
  downstream_ratio: number; // 0-1
}

export interface DownstreamResult {
  pairs: DownstreamPair[];
}

// ── Fan-out Trajectory ──

export interface EntityFanout {
  entity: string;
  windows: { start: string; end: string; unique_participants: number }[];
  slope: number;           // positive = escalation, negative = maturation
}

export interface FanoutResult {
  by_author: EntityFanout[];
  window_size_days: number;
}

// ── Quadrant Classification ──

/**
 * Flow × Entropy 2×2:
 * - river:      high flow, low entropy (healthy, ordered throughput)
 * - waterfall:  high flow, high entropy (productive but chaotic/fragile)
 * - bottleneck: low flow, low entropy (predictable failure point)
 * - swamp:      low flow, high entropy (energy trapped, unpredictable)
 */
export type OrgQuadrant = 'river' | 'waterfall' | 'bottleneck' | 'swamp';

export interface EntityClassification {
  entity: string;
  entity_type: 'author' | 'label';
  quadrant: OrgQuadrant;
  flow_rate: number;
  entropy: number;
}

// ── Combined ──

export interface OrgThermoResult {
  flow: FlowResult;
  entropy: EntropyResult;
  cadence: CadenceResult;
  downstream: DownstreamResult;
  fanout: FanoutResult;
}
