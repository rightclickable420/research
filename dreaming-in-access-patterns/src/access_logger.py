#!/usr/bin/env python3
"""
Access Logger — Extracts memory_search access events from session transcripts.

Called during compaction flush. Parses the session transcript for memory_search
tool calls and their results, logs each as an access event.

The access log is the raw signal for reconsolidation. Every memory_search call
tells us: what was the query, which chunks resonated, how strongly.

Usage:
  python access_logger.py log-session <transcript_json>
  python access_logger.py log-event <query> <results_json>
  python access_logger.py stats
  python access_logger.py dump [--limit N]
"""

import json
import sqlite3
import sys
import time
import re
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "access.db"


def get_db() -> sqlite3.Connection:
    db = sqlite3.connect(str(DB_PATH))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""
        CREATE TABLE IF NOT EXISTS access_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            session_id TEXT,
            query TEXT NOT NULL,
            results TEXT NOT NULL,  -- JSON: [{file, content, lines, score}]
            n_results INTEGER NOT NULL,
            top_score REAL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS chunk_energy (
            chunk_key TEXT PRIMARY KEY,  -- file:line_start
            total_accesses INTEGER DEFAULT 0,
            total_score REAL DEFAULT 0.0,
            last_accessed REAL,
            first_accessed REAL
        )
    """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_access_timestamp ON access_events(timestamp)
    """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunk_accesses ON chunk_energy(total_accesses DESC)
    """)
    db.commit()
    return db


def log_event(query: str, results: list[dict], session_id: str = None, timestamp: float = None):
    """Log a single memory_search access event."""
    ts = timestamp or time.time()
    db = get_db()

    top_score = max((r.get("score", 0) for r in results), default=0)

    db.execute(
        "INSERT INTO access_events (timestamp, session_id, query, results, n_results, top_score) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (ts, session_id, query, json.dumps(results), len(results), top_score)
    )

    # Update chunk energy for each result
    for r in results:
        chunk_key = f"{r.get('file', r.get('path', '?'))}:{r.get('lines', r.get('line', '?'))}"
        score = r.get("score", 0.5)

        existing = db.execute(
            "SELECT total_accesses FROM chunk_energy WHERE chunk_key = ?",
            (chunk_key,)
        ).fetchone()

        if existing:
            db.execute(
                "UPDATE chunk_energy SET total_accesses = total_accesses + 1, "
                "total_score = total_score + ?, last_accessed = ? WHERE chunk_key = ?",
                (score, ts, chunk_key)
            )
        else:
            db.execute(
                "INSERT INTO chunk_energy (chunk_key, total_accesses, total_score, last_accessed, first_accessed) "
                "VALUES (?, 1, ?, ?, ?)",
                (chunk_key, score, ts, ts)
            )

    db.commit()
    db.close()


def extract_from_transcript(transcript: str, session_id: str = None) -> int:
    """
    Parse a session transcript for memory_search calls and log them.
    
    Handles both structured JSON transcripts and text-format transcripts.
    Returns number of access events logged.
    """
    count = 0

    # Try JSON format first (array of messages)
    try:
        messages = json.loads(transcript)
        if isinstance(messages, list):
            return _extract_from_messages(messages, session_id)
    except (json.JSONDecodeError, TypeError):
        pass

    # Text format: look for memory_search patterns
    # Pattern 1: tool call blocks
    search_pattern = re.compile(
        r'memory_search.*?["\']query["\']\s*[:=]\s*["\']([^"\']+)["\']',
        re.DOTALL | re.IGNORECASE
    )
    result_pattern = re.compile(
        r'(?:snippets|results|matches).*?(\[[\s\S]*?\])',
        re.DOTALL
    )

    # Find all memory_search queries in the text
    queries = search_pattern.findall(transcript)
    results_blocks = result_pattern.findall(transcript)

    # Pair them up best-effort
    for i, query in enumerate(queries):
        results = []
        if i < len(results_blocks):
            try:
                results = json.loads(results_blocks[i])
            except json.JSONDecodeError:
                pass

        if query:  # Log even without results — the query itself is signal
            log_event(query, results if isinstance(results, list) else [], session_id)
            count += 1

    return count


def _extract_from_messages(messages: list, session_id: str = None) -> int:
    """Extract from structured message format."""
    count = 0
    pending_query = None

    for msg in messages:
        # Look for tool_use with memory_search
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use" and block.get("name") == "memory_search":
                            pending_query = block.get("input", {}).get("query", "")
                        elif block.get("type") == "tool_result" and pending_query:
                            # Try to parse the result
                            result_content = block.get("content", "")
                            if isinstance(result_content, str):
                                try:
                                    parsed = json.loads(result_content)
                                    if isinstance(parsed, dict) and "snippets" in parsed:
                                        results = parsed["snippets"]
                                    elif isinstance(parsed, list):
                                        results = parsed
                                    else:
                                        results = []
                                except json.JSONDecodeError:
                                    results = []
                            else:
                                results = []
                            log_event(pending_query, results, session_id)
                            pending_query = None
                            count += 1
            elif isinstance(content, str) and "memory_search" in content:
                # Text content mentioning memory_search
                match = re.search(r'query["\s:=]+["\']([^"\']+)', content)
                if match:
                    pending_query = match.group(1)

    return count


def stats() -> dict:
    """Return access statistics."""
    db = get_db()

    total_events = db.execute("SELECT COUNT(*) FROM access_events").fetchone()[0]
    total_chunks = db.execute("SELECT COUNT(*) FROM chunk_energy").fetchone()[0]
    
    earliest = db.execute("SELECT MIN(timestamp) FROM access_events").fetchone()[0]
    latest = db.execute("SELECT MAX(timestamp) FROM access_events").fetchone()[0]

    top_chunks = db.execute(
        "SELECT chunk_key, total_accesses, total_score, last_accessed "
        "FROM chunk_energy ORDER BY total_accesses DESC LIMIT 15"
    ).fetchall()

    # Unique queries
    unique_queries = db.execute("SELECT COUNT(DISTINCT query) FROM access_events").fetchone()[0]

    # Recent queries
    recent = db.execute(
        "SELECT query, n_results, top_score, timestamp FROM access_events "
        "ORDER BY timestamp DESC LIMIT 10"
    ).fetchall()

    db.close()

    return {
        "total_access_events": total_events,
        "unique_queries": unique_queries,
        "tracked_chunks": total_chunks,
        "earliest_event": earliest,
        "latest_event": latest,
        "hottest_chunks": [
            {"chunk": r[0], "accesses": r[1], "total_score": round(r[2], 3), "last_accessed": r[3]}
            for r in top_chunks
        ],
        "recent_queries": [
            {"query": r[0], "results": r[1], "top_score": r[2], "timestamp": r[3]}
            for r in recent
        ]
    }


def dump(limit: int = 50) -> list:
    """Dump recent access events."""
    db = get_db()
    rows = db.execute(
        "SELECT timestamp, session_id, query, n_results, top_score "
        "FROM access_events ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    ).fetchall()
    db.close()
    return [
        {"timestamp": r[0], "session": r[1], "query": r[2], "results": r[3], "top_score": r[4]}
        for r in rows
    ]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: access_logger.py <log-session|log-event|stats|dump>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "stats":
        s = stats()
        print(json.dumps(s, indent=2))

    elif cmd == "dump":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        for event in dump(limit):
            print(json.dumps(event))

    elif cmd == "log-event":
        if len(sys.argv) < 3:
            print("Usage: access_logger.py log-event <query> [results_json]")
            sys.exit(1)
        query = sys.argv[2]
        results = json.loads(sys.argv[3]) if len(sys.argv) > 3 else []
        log_event(query, results)
        print(f"Logged: {query} ({len(results)} results)")

    elif cmd == "log-session":
        if len(sys.argv) < 3:
            # Read from stdin
            transcript = sys.stdin.read()
        else:
            transcript = sys.argv[2]
        session_id = sys.argv[3] if len(sys.argv) > 3 else None
        count = extract_from_transcript(transcript, session_id)
        print(f"Extracted {count} access events")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
