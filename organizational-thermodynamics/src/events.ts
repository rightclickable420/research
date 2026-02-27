/**
 * Events — Transform GitHub API data into InteractionEvent[].
 *
 * Accepts raw JSON arrays from GitHub's REST API and produces
 * the flat event log used by all metric computations.
 */

import type { InteractionEvent } from './types';

// ── GitHub API shapes (minimal, metadata only) ──

interface GHUser {
  login: string;
}

interface GHLabel {
  name: string;
}

interface GHIssue {
  number: number;
  user: GHUser;
  created_at: string;
  state: string;
  closed_at?: string | null;
  labels?: GHLabel[];
  pull_request?: unknown; // present if it's actually a PR
  assignees?: GHUser[];
}

interface GHComment {
  user: GHUser;
  created_at: string;
  issue_url?: string; // contains issue number
}

interface GHReview {
  user: GHUser;
  submitted_at: string;
  state: string; // APPROVED, CHANGES_REQUESTED, COMMENTED, DISMISSED
}

interface GHPullRequest {
  number: number;
  user: GHUser;
  created_at: string;
  state: string;
  merged_at?: string | null;
  closed_at?: string | null;
  labels?: GHLabel[];
  assignees?: GHUser[];
}

// ── Transformers ──

function extractLabels(labels?: GHLabel[]): string[] | undefined {
  if (!labels || labels.length === 0) return undefined;
  return labels.map(l => l.name);
}

/** Transform issues into events. */
export function issuestoEvents(issues: GHIssue[]): InteractionEvent[] {
  const events: InteractionEvent[] = [];

  for (const issue of issues) {
    if (issue.pull_request) continue; // skip PRs listed as issues

    const labels = extractLabels(issue.labels);
    const threadId = `issue-${issue.number}`;

    events.push({
      timestamp: issue.created_at,
      from: issue.user.login,
      thread_id: threadId,
      event_type: 'open',
      thread_type: 'issue',
      labels,
    });

    if (issue.closed_at) {
      events.push({
        timestamp: issue.closed_at,
        from: issue.user.login,
        thread_id: threadId,
        event_type: 'close',
        thread_type: 'issue',
        labels,
      });
    }

    for (const assignee of issue.assignees ?? []) {
      events.push({
        timestamp: issue.created_at,
        from: issue.user.login,
        to: assignee.login,
        thread_id: threadId,
        event_type: 'assign',
        thread_type: 'issue',
        labels,
      });
    }
  }

  return events;
}

/** Transform PRs into events. */
export function prsToEvents(prs: GHPullRequest[]): InteractionEvent[] {
  const events: InteractionEvent[] = [];

  for (const pr of prs) {
    const labels = extractLabels(pr.labels);
    const threadId = `pr-${pr.number}`;

    events.push({
      timestamp: pr.created_at,
      from: pr.user.login,
      thread_id: threadId,
      event_type: 'open',
      thread_type: 'pr',
      labels,
    });

    if (pr.merged_at) {
      events.push({
        timestamp: pr.merged_at,
        from: pr.user.login,
        thread_id: threadId,
        event_type: 'merge',
        thread_type: 'pr',
        labels,
      });
    } else if (pr.closed_at) {
      events.push({
        timestamp: pr.closed_at,
        from: pr.user.login,
        thread_id: threadId,
        event_type: 'close',
        thread_type: 'pr',
        labels,
      });
    }

    for (const assignee of pr.assignees ?? []) {
      events.push({
        timestamp: pr.created_at,
        from: pr.user.login,
        to: assignee.login,
        thread_id: threadId,
        event_type: 'assign',
        thread_type: 'pr',
        labels,
      });
    }
  }

  return events;
}

/** Transform issue/PR comments into events (single thread). */
export function commentsToEvents(
  comments: GHComment[],
  threadId: string,
  threadType: 'issue' | 'pr',
  labels?: string[],
): InteractionEvent[] {
  return comments.map(c => ({
    timestamp: c.created_at,
    from: c.user.login,
    thread_id: threadId,
    event_type: 'comment' as const,
    thread_type: threadType,
    labels,
  }));
}

/** Bulk comment shape from our API fetch (has issue_number). */
interface BulkComment {
  issue_number: number;
  author: string;
  created_at: string;
  updated_at?: string;
}

/**
 * Transform bulk comments (from /repos/{owner}/{repo}/issues/comments endpoint)
 * into events. Automatically maps issue_number → thread_id.
 * Uses prNumbers set to determine if a thread is a PR or issue.
 */
export function bulkCommentsToEvents(
  comments: BulkComment[],
  prNumbers: Set<number>,
  labelMap?: Map<number, string[]>,
): InteractionEvent[] {
  return comments
    .filter(c => c && c.author && c.issue_number)
    .map(c => {
      const isPr = prNumbers.has(c.issue_number);
      return {
        timestamp: c.created_at,
        from: c.author,
        thread_id: `${isPr ? 'pr' : 'issue'}-${c.issue_number}`,
        event_type: 'comment' as const,
        thread_type: (isPr ? 'pr' : 'issue') as 'issue' | 'pr',
        labels: labelMap?.get(c.issue_number),
      };
    });
}

/** Bulk review shape from our API fetch. */
interface BulkReview {
  pr_number: number;
  author: string;
  state: string;
  submitted_at: string;
}

/**
 * Transform bulk reviews into events.
 */
export function bulkReviewsToEvents(
  reviews: BulkReview[],
  labelMap?: Map<number, string[]>,
): InteractionEvent[] {
  return reviews
    .filter(r => r && r.author && r.pr_number)
    .map(r => ({
      timestamp: r.submitted_at,
      from: r.author,
      thread_id: `pr-${r.pr_number}`,
      event_type: 'review' as const,
      thread_type: 'pr' as const,
      labels: labelMap?.get(r.pr_number),
    }));
}

/** Transform PR reviews into events. */
export function reviewsToEvents(
  reviews: GHReview[],
  threadId: string,
  labels?: string[],
): InteractionEvent[] {
  return reviews.map(r => ({
    timestamp: r.submitted_at,
    from: r.user.login,
    thread_id: threadId,
    event_type: 'review' as const,
    thread_type: 'pr' as const,
    labels,
  }));
}
