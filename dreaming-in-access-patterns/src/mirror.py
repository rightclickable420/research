#!/usr/bin/env python3
"""
Mirror Generator — Produces memory/mirror.md from access patterns and session transcripts.

Nightly cron job. Analyzes:
1. Access energy — what memory chunks are hot/cold
2. Gap detection — what I search for that isn't in boot context
3. Friction — repeated searches, tool failures
4. Co-access resonance — chunks that activate together across sessions

Output: compressed shorthand in memory/mirror.md, loaded on heartbeat.

Usage:
  python mirror.py generate
  python mirror.py generate --dry-run
"""

import json
import sqlite3
import sys
import time
import os
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict

WORKSPACE = Path(os.environ.get("WORKSPACE", "/home/clawd/clawd"))
ACCESS_DB = Path(__file__).parent / "access.db"
MIRROR_PATH = WORKSPACE / "memory" / "mirror.md"
SESSIONS_DIR = Path(os.path.expanduser("~/.openclaw/agents/main/sessions"))

# Boot context files — things loaded every session
BOOT_FILES = {"MEMORY.md", "SOUL.md", "USER.md", "IDENTITY.md", "TOOLS.md", "AGENTS.md", "HEARTBEAT.md"}


def load_access_data(days: int = 14) -> dict:
    """Load access events from the last N days."""
    if not ACCESS_DB.exists():
        return {"events": [], "chunks": {}}

    db = sqlite3.connect(str(ACCESS_DB))
    cutoff = time.time() - (days * 86400)

    events = db.execute(
        "SELECT timestamp, session_id, query, results, n_results, top_score "
        "FROM access_events WHERE timestamp > ? ORDER BY timestamp",
        (cutoff,)
    ).fetchall()

    chunks = db.execute(
        "SELECT chunk_key, total_accesses, total_score, last_accessed, first_accessed "
        "FROM chunk_energy ORDER BY total_accesses DESC"
    ).fetchall()

    db.close()

    return {
        "events": [
            {"ts": e[0], "session": e[1], "query": e[2],
             "results": json.loads(e[3]) if e[3] else [], "n_results": e[4], "top_score": e[5]}
            for e in events
        ],
        "chunks": {
            c[0]: {"accesses": c[1], "score": c[2], "last": c[3], "first": c[4]}
            for c in chunks
        }
    }


def analyze_hot_cold(data: dict, top_n: int = 10) -> dict:
    """Identify hottest and coldest chunks."""
    chunks = data["chunks"]
    if not chunks:
        return {"hot": [], "cold": []}

    sorted_chunks = sorted(chunks.items(), key=lambda x: x[1]["accesses"], reverse=True)
    hot = sorted_chunks[:top_n]

    # Cold = chunks that exist in boot files but never get accessed
    # We can infer these from the access log — boot file chunks not in the energy map
    return {
        "hot": [{"key": k, "accesses": v["accesses"], "score": round(v["score"], 2)} for k, v in hot],
        "cold_boot": []  # filled in by gap analysis
    }


def analyze_gaps(data: dict) -> list:
    """Find queries that returned 0 results or low scores — gaps in memory."""
    gaps = []
    query_counts = Counter()

    for event in data["events"]:
        if event["n_results"] == 0 or event["top_score"] == 0:
            query_counts[event["query"]] += 1

    # Queries that repeatedly fail
    for query, count in query_counts.most_common(15):
        gaps.append({"query": query, "misses": count})

    return gaps


def analyze_friction(data: dict) -> list:
    """Find repeated searches in the same session — retrieval friction."""
    session_queries = defaultdict(list)

    for event in data["events"]:
        sid = event["session"] or "unknown"
        session_queries[sid].append(event["query"])

    friction = []
    for sid, queries in session_queries.items():
        # Look for similar queries in same session (fuzzy: same first 3 words)
        seen = defaultdict(int)
        for q in queries:
            key = " ".join(q.lower().split()[:3])
            seen[key] += 1
        for key, count in seen.items():
            if count >= 2:
                friction.append({"pattern": key, "repeats": count, "session": sid[:8]})

    # Deduplicate across sessions
    pattern_counts = Counter()
    for f in friction:
        pattern_counts[f["pattern"]] += f["repeats"]

    return [{"pattern": p, "total_repeats": c} for p, c in pattern_counts.most_common(10)]


def analyze_resonance(data: dict) -> list:
    """Find chunks that co-access across sessions."""
    session_chunks = defaultdict(set)

    for event in data["events"]:
        sid = event["session"] or "unknown"
        for r in event["results"]:
            key = f"{r.get('file', '?')}:{r.get('lines', '?')}"
            session_chunks[sid].add(key)

    # Count co-occurrences
    cooccur = Counter()
    for sid, chunks in session_chunks.items():
        chunk_list = sorted(chunks)
        for i in range(len(chunk_list)):
            for j in range(i + 1, len(chunk_list)):
                pair = (chunk_list[i], chunk_list[j])
                cooccur[pair] += 1

    # Only pairs that co-occur in 2+ sessions
    resonant = [(pair, count) for pair, count in cooccur.most_common(20) if count >= 2]

    return [{"a": p[0], "b": p[1], "sessions": c} for p, c in resonant]


def analyze_tool_failures(days: int = 14) -> list:
    """Scan recent session transcripts for tool call failures."""
    import re
    failures = Counter()
    cutoff = time.time() - (days * 86400)

    if not SESSIONS_DIR.exists():
        return []

    for sf in sorted(SESSIONS_DIR.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True):
        if sf.stat().st_mtime < cutoff:
            break
        try:
            with open(sf) as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        msg = obj.get("message", {})
                        # Look for toolResult messages with error indicators
                        if msg.get("role") != "toolResult":
                            continue
                        content = msg.get("content", [])
                        if not isinstance(content, list):
                            continue
                        for b in content:
                            if not isinstance(b, dict) or b.get("type") != "text":
                                continue
                            text = b.get("text", "")
                            # Match "exited with code N" where N != 0
                            m = re.search(r'(?:Process |Command )exited with code (\d+)', text)
                            if m and m.group(1) != "0":
                                failures[f"exit:{m.group(1)}"] += 1
                            elif "Command timed out" in text:
                                failures["timeout"] += 1
                            elif '"status": "error"' in text[:100]:
                                failures["tool-error"] += 1
                    except (json.JSONDecodeError, KeyError):
                        pass
        except IOError:
            continue

    return [{"type": t, "count": c} for t, c in failures.most_common(5) if c >= 2]


def generate_mirror(dry_run: bool = False) -> str:
    """Generate the mirror file content."""
    data = load_access_data(days=14)

    if not data["events"]:
        return "# mirror — no access data yet\n"

    hot_cold = analyze_hot_cold(data)
    gaps = analyze_gaps(data)
    friction = analyze_friction(data)
    resonance = analyze_resonance(data)
    tool_fails = analyze_tool_failures(days=14)

    now = datetime.now(timezone.utc)
    lines = [f"# mirror [{now.strftime('%Y-%m-%d')}]", ""]

    # Hot chunks — compressed
    if hot_cold["hot"]:
        hot_strs = []
        for h in hot_cold["hot"][:8]:
            key = h["key"]
            # Compress: MEMORY.md:51 → M:51, memory/2026-02-07.md:1 → m/0207:1
            if key.startswith("MEMORY.md:"):
                short = f"M:{key.split(':')[1]}"
            elif key.startswith("memory/"):
                parts = key.replace("memory/", "").split(":")
                date_part = parts[0].replace("2026-", "").replace("-", "").replace(".md", "")
                short = f"m/{date_part}:{parts[1]}" if len(parts) > 1 else f"m/{date_part}"
            else:
                short = key.replace(".md", "").replace(":", "→")
                if len(short) > 15:
                    short = short[:15]
            hot_strs.append(f"{short}({h['accesses']}x)")
        lines.append(f"hot: {' '.join(hot_strs)}")

    # Gaps — queries that fail
    if gaps:
        gap_strs = [f"\"{g['query'][:40]}\"({g['misses']}x)" for g in gaps[:5]]
        lines.append(f"gaps: {' | '.join(gap_strs)}")

    # Friction — repeated searches
    if friction:
        fric_strs = [f"{f['pattern']}({f['total_repeats']}x)" for f in friction[:5]]
        lines.append(f"friction: {' | '.join(fric_strs)}")

    # Resonance — co-accessed chunks
    if resonance:
        lines.append("resonance:")
        for r in resonance[:5]:
            a = r["a"].replace("MEMORY.md", "M").replace("memory/", "m/").replace(".md", "")
            b = r["b"].replace("MEMORY.md", "M").replace("memory/", "m/").replace(".md", "")
            lines.append(f"  {a} ↔ {b} ({r['sessions']}s)")

    # Tool failures
    if tool_fails:
        fail_strs = [f"{f['type']}({f['count']}x)" for f in tool_fails[:5]]
        lines.append(f"errors: {' '.join(fail_strs)}")

    # Promotion candidates — high access, broad sessions, not in boot context
    boot_files = {"MEMORY.md", "SOUL.md", "USER.md", "IDENTITY.md", "TOOLS.md", "AGENTS.md", "HEARTBEAT.md"}
    session_chunks = defaultdict(set)
    for event in data["events"]:
        sid = event["session"] or "unknown"
        for r in event["results"]:
            key = f"{r.get('file', '?')}:{r.get('lines', '?')}"
            session_chunks[key].add(sid)

    promotions = []
    for chunk_key, sessions in session_chunks.items():
        file_part = chunk_key.split(":")[0]
        if file_part in boot_files:
            continue  # already in boot context
        accesses = data["chunks"].get(chunk_key, {}).get("accesses", 0)
        if accesses >= 5 and len(sessions) >= 3:
            promotions.append({
                "key": chunk_key,
                "accesses": accesses,
                "sessions": len(sessions),
            })

    if promotions:
        promotions.sort(key=lambda x: x["accesses"], reverse=True)
        promo_strs = [f"{p['key']}({p['accesses']}x/{p['sessions']}s)" for p in promotions[:5]]
        lines.append(f"promote: {' | '.join(promo_strs)}")

    # Stats
    total_events = len(data["events"])
    unique_queries = len(set(e["query"] for e in data["events"]))
    sessions_with_search = len(set(e["session"] for e in data["events"] if e["session"]))
    lines.append("")
    lines.append(f"stats: {total_events}ev/{unique_queries}uq/{sessions_with_search}sess/14d")

    content = "\n".join(lines) + "\n"

    if not dry_run:
        MIRROR_PATH.parent.mkdir(parents=True, exist_ok=True)
        MIRROR_PATH.write_text(content)

    return content


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv

    content = generate_mirror(dry_run=dry_run)
    print(content)

    if not dry_run:
        print(f"→ Written to {MIRROR_PATH}")
