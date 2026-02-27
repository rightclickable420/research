/**
 * Org-Thermo Diagnostic Agent
 * 
 * Pipeline: detect swamps → scope threads → read content → generate action plan
 * 
 * Content is ONLY read for threads flagged by the thermodynamic map.
 * This is surgical, justified, scoped access — not surveillance.
 */

import type {
  InteractionEvent,
  OrgThermoResult,
  EntityClassification,
  ThreadOutcome,
} from './types';

// ── Types ──

export interface SwampDiagnosis {
  entity: string;
  quadrant: 'swamp' | 'bottleneck';
  flow_rate: number;
  entropy: number;
  total_threads: number;
  stuck_threads: number;
  scoped_threads: ScopedThread[];
  diagnosis?: string;       // LLM-generated
  action_plan?: string[];   // LLM-generated action items
}

export interface ScopedThread {
  thread_id: string;
  thread_type: 'issue' | 'pr';
  outcome: ThreadOutcome;
  participants: string[];
  event_count: number;
  age_days: number;
  last_activity_days: number;
  // Content (fetched separately, only for flagged threads)
  title?: string;
  content_summary?: string;
  labels?: string[];
}

export interface DiagnosticReport {
  source: string;
  generated_at: string;
  total_entities: number;
  flagged_entities: number;
  diagnoses: SwampDiagnosis[];
  summary?: string;
}

// ── Thread outcome classification (same logic as entropy.ts) ──

const STALL_DAYS = 7;

function classifyThreadOutcome(events: InteractionEvent[]): ThreadOutcome {
  const sorted = [...events].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  const hasClose = sorted.some(e => e.event_type === 'close' || e.event_type === 'merge');
  if (hasClose) return 'resolved';

  const participants = new Set(sorted.map(e => e.from));
  if (participants.size >= 3) return 'expanded';

  if (participants.size === 2 && sorted.length >= 4) {
    let alternations = 0;
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i].from !== sorted[i - 1].from) alternations++;
    }
    if (alternations >= 3) return 'looped';
  }

  const assignEvents = sorted.filter(e => e.event_type === 'assign');
  if (assignEvents.length >= 2) return 'escalated';

  const first = sorted[0];
  const last = sorted[sorted.length - 1];
  const threadAgeDays = (Date.now() - new Date(first.timestamp).getTime()) / (1000 * 60 * 60 * 24);
  const daysSinceLast = (Date.now() - new Date(last.timestamp).getTime()) / (1000 * 60 * 60 * 24);

  if (threadAgeDays > STALL_DAYS && daysSinceLast > STALL_DAYS) return 'stalled';
  return participants.size > 1 ? 'expanded' : 'stalled';
}

// ── Swamp Detection ──

export interface DetectOptions {
  /** Max swamps to analyze (default 10) */
  maxEntities?: number;
  /** Max threads per swamp to scope (default 5) */
  maxThreadsPerEntity?: number;
  /** Include bottlenecks too (default true) */
  includeBottlenecks?: boolean;
}

/**
 * Detect swamps and scope the specific threads that need investigation.
 * Returns the threads that should have their content fetched.
 */
export function detectAndScope(
  events: InteractionEvent[],
  result: OrgThermoResult,
  classified: EntityClassification[],
  opts: DetectOptions = {},
): SwampDiagnosis[] {
  const maxEntities = opts.maxEntities ?? 10;
  const maxThreads = opts.maxThreadsPerEntity ?? 5;
  const includeBottlenecks = opts.includeBottlenecks ?? true;

  // Get swamps (and optionally bottlenecks), sorted by entropy descending
  const targets = classified
    .filter(c => c.quadrant === 'swamp' || (includeBottlenecks && c.quadrant === 'bottleneck'))
    .sort((a, b) => b.entropy - a.entropy)
    .slice(0, maxEntities);

  // Group all events by thread
  const threadEvents = new Map<string, InteractionEvent[]>();
  for (const e of events) {
    const arr = threadEvents.get(e.thread_id);
    if (arr) arr.push(e);
    else threadEvents.set(e.thread_id, [e]);
  }

  const now = Date.now();

  return targets.map(target => {
    // Find all threads this entity participates in
    const entityThreadIds = new Set(
      events.filter(e => e.from === target.entity).map(e => e.thread_id)
    );

    // Classify and score each thread
    const threadScores: Array<{ threadId: string; thread: ScopedThread; score: number }> = [];

    for (const threadId of entityThreadIds) {
      const tevents = threadEvents.get(threadId);
      if (!tevents) continue;

      const outcome = classifyThreadOutcome(tevents);
      if (outcome === 'resolved') continue; // skip healthy threads

      const sorted = [...tevents].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
      const participants = [...new Set(sorted.map(e => e.from))];
      const firstTime = new Date(sorted[0].timestamp).getTime();
      const lastTime = new Date(sorted[sorted.length - 1].timestamp).getTime();
      const ageDays = (now - firstTime) / (1000 * 60 * 60 * 24);
      const lastActivityDays = (now - lastTime) / (1000 * 60 * 60 * 24);
      const labels = sorted.find(e => e.labels)?.labels;

      // Score: higher = more worth investigating
      // Stalled threads with many participants and high age are most interesting
      let score = 0;
      if (outcome === 'stalled') score += 3;
      if (outcome === 'looped') score += 4;  // active conflict/confusion
      if (outcome === 'expanded') score += 2;
      if (outcome === 'escalated') score += 3;
      score += Math.min(participants.length, 5); // more people = more impact
      score += Math.min(ageDays / 7, 5); // older = more stuck
      score += tevents.length * 0.5; // more activity without resolution = worse

      threadScores.push({
        threadId,
        thread: {
          thread_id: threadId,
          thread_type: sorted[0].thread_type,
          outcome,
          participants,
          event_count: tevents.length,
          age_days: Math.round(ageDays),
          last_activity_days: Math.round(lastActivityDays),
          labels,
        },
        score,
      });
    }

    // Take the worst threads
    threadScores.sort((a, b) => b.score - a.score);
    const scopedThreads = threadScores.slice(0, maxThreads).map(t => t.thread);

    const flow = result.flow.by_author.find(f => f.entity === target.entity);

    return {
      entity: target.entity,
      quadrant: target.quadrant as 'swamp' | 'bottleneck',
      flow_rate: target.flow_rate,
      entropy: target.entropy,
      total_threads: entityThreadIds.size,
      stuck_threads: threadScores.length,
      scoped_threads: scopedThreads,
    };
  });
}

// ── Content Fetcher Interface ──

export interface ContentFetcher {
  /**
   * Fetch the title and content summary for a specific thread.
   * Implementation depends on data source (GitHub, Slack, etc.)
   */
  fetchThreadContent(threadId: string, threadType: 'issue' | 'pr'): Promise<{
    title: string;
    content_summary: string;
  } | null>;
}

/**
 * Enrich scoped threads with content from the data source.
 * Only fetches content for threads that were flagged by the thermodynamic map.
 */
export async function enrichWithContent(
  diagnoses: SwampDiagnosis[],
  fetcher: ContentFetcher,
): Promise<SwampDiagnosis[]> {
  for (const diag of diagnoses) {
    for (const thread of diag.scoped_threads) {
      const content = await fetcher.fetchThreadContent(thread.thread_id, thread.thread_type);
      if (content) {
        thread.title = content.title;
        thread.content_summary = content.content_summary;
      }
    }
  }
  return diagnoses;
}

// ── Diagnostic Prompt Builder ──

/**
 * Build a prompt for an LLM to diagnose a swamp entity and generate an action plan.
 * The LLM only sees the scoped threads — not the entire communication history.
 */
export function buildDiagnosticPrompt(diagnosis: SwampDiagnosis): string {
  const threadDescriptions = diagnosis.scoped_threads.map((t, i) => {
    const lines = [
      `Thread ${i + 1}: ${t.title ?? t.thread_id}`,
      `  Type: ${t.thread_type} | Outcome: ${t.outcome} | Age: ${t.age_days} days | Last activity: ${t.last_activity_days} days ago`,
      `  Participants: ${t.participants.join(', ')} (${t.participants.length})`,
      `  Events: ${t.event_count}`,
    ];
    if (t.labels?.length) lines.push(`  Labels: ${t.labels.join(', ')}`);
    if (t.content_summary) lines.push(`  Summary: ${t.content_summary}`);
    return lines.join('\n');
  }).join('\n\n');

  return `You are analyzing organizational health data. A contributor has been flagged by automated thermodynamic analysis.

## Entity Profile
- **Name:** ${diagnosis.entity}
- **Classification:** ${diagnosis.quadrant} (${diagnosis.quadrant === 'swamp' ? 'low flow, high entropy — energy enters and gets trapped' : 'low flow, low entropy — predictable blockage'})
- **Flow Rate:** ${diagnosis.flow_rate.toFixed(3)} (resolved/open ratio)
- **Entropy:** ${diagnosis.entropy.toFixed(3)} (outcome unpredictability, 0-1)
- **Total Threads:** ${diagnosis.total_threads} | **Stuck:** ${diagnosis.stuck_threads}

## Flagged Threads (scoped — only these were read)
${threadDescriptions}

## Task
Based on the profile and flagged threads:

1. **Diagnose** — What pattern do you see? Is this person overloaded, blocked by dependencies, gatekeeping, or struggling? Be specific.
2. **Root Cause** — What's the most likely underlying cause of the swamp forming around this person?
3. **Action Plan** — Provide 3-5 specific, actionable steps to restore flow. Focus on systemic fixes (routing, responsibility, process) not individual behavior.

Be concise and direct. No platitudes.`;
}

/**
 * Generate a full diagnostic report (without LLM — just the structured data).
 * LLM diagnosis can be added by calling an LLM with buildDiagnosticPrompt() for each diagnosis.
 */
export function generateReport(
  diagnoses: SwampDiagnosis[],
  source: string,
): DiagnosticReport {
  return {
    source,
    generated_at: new Date().toISOString(),
    total_entities: 0, // filled by caller
    flagged_entities: diagnoses.length,
    diagnoses,
  };
}
