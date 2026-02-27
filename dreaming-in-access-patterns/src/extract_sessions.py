#!/usr/bin/env python3
"""
Extract memory_search access events from OpenClaw session transcripts.

Runs as a cron job — no agent involvement. Scans session JSONL files,
extracts memory_search calls and their results, logs to access.db.

Tracks which sessions have been processed to avoid double-counting.

Usage:
  python extract_sessions.py              # Process new sessions
  python extract_sessions.py --all        # Reprocess all sessions
  python extract_sessions.py --backfill   # Process all historical sessions
"""

import json
import sqlite3
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Import the access logger
sys.path.insert(0, str(Path(__file__).parent))
from access_logger import log_event, get_db

SESSIONS_DIR = Path(os.path.expanduser("~/.openclaw/agents/main/sessions"))
STATE_DB = Path(__file__).parent / "access.db"


def get_processed_sessions() -> set:
    """Get set of session IDs already processed."""
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS processed_sessions (
            session_id TEXT PRIMARY KEY,
            processed_at REAL,
            events_extracted INTEGER
        )
    """)
    db.commit()
    rows = db.execute("SELECT session_id FROM processed_sessions").fetchall()
    db.close()
    return {r[0] for r in rows}


def mark_processed(session_id: str, events: int):
    """Mark a session as processed."""
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS processed_sessions (
            session_id TEXT PRIMARY KEY,
            processed_at REAL,
            events_extracted INTEGER
        )
    """)
    db.execute(
        "INSERT OR REPLACE INTO processed_sessions (session_id, processed_at, events_extracted) "
        "VALUES (?, ?, ?)",
        (session_id, time.time(), events)
    )
    db.commit()
    db.close()


def extract_session(session_path: Path) -> int:
    """
    Extract memory_search access events from a session JSONL file.
    
    Structure:
    - Assistant messages contain toolCall blocks with name=memory_search
    - Tool results are in messages with parentId matching the call message's id
    - Results are JSON: {results: [{path, startLine, endLine, score, snippet}]}
    
    Returns number of events extracted.
    """
    lines = []
    try:
        with open(session_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
    except (json.JSONDecodeError, IOError) as e:
        return 0

    session_id = session_path.stem
    count = 0

    # Build index: message id → message object
    id_to_obj = {}
    for obj in lines:
        oid = obj.get("id", "")
        if oid:
            id_to_obj[oid] = obj

    # Find memory_search calls
    for obj in lines:
        msg = obj.get("message", {})
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        call_msg_id = obj.get("id", "")
        timestamp_str = obj.get("timestamp", "")

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "toolCall" or block.get("name") != "memory_search":
                continue

            query = block.get("arguments", {}).get("query", "")
            if not query:
                continue

            # Parse timestamp
            ts = None
            if timestamp_str:
                try:
                    dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    ts = dt.timestamp()
                except (ValueError, TypeError):
                    pass
            if ts is None:
                ts = time.time()

            # Find the matching tool result (child message with parentId == this message's id)
            results = []
            for obj2 in lines:
                if obj2.get("parentId") != call_msg_id:
                    continue
                msg2 = obj2.get("message", {})
                if msg2.get("role") != "toolResult":
                    continue

                content2 = msg2.get("content", [])
                if isinstance(content2, list):
                    for b2 in content2:
                        if isinstance(b2, dict) and b2.get("type") == "text":
                            text = b2.get("text", "")
                            try:
                                parsed = json.loads(text)
                                if isinstance(parsed, dict) and "results" in parsed:
                                    for r in parsed["results"]:
                                        # Map to access_logger format
                                        # path could be a session transcript or a memory file
                                        file_path = r.get("path", "")
                                        # Only count memory file results, not session transcript hits
                                        if file_path.startswith("sessions/"):
                                            continue
                                        results.append({
                                            "file": file_path,
                                            "lines": str(r.get("startLine", "")),
                                            "score": r.get("score", 0),
                                        })
                            except (json.JSONDecodeError, TypeError):
                                pass
                break  # Found the result

            # Log the event
            log_event(query, results, session_id=session_id, timestamp=ts)
            count += 1

    return count


def main():
    reprocess = "--all" in sys.argv or "--backfill" in sys.argv

    if not SESSIONS_DIR.exists():
        print(f"Sessions directory not found: {SESSIONS_DIR}")
        sys.exit(1)

    session_files = sorted(SESSIONS_DIR.glob("*.jsonl"))
    processed = get_processed_sessions() if not reprocess else set()

    total_events = 0
    new_sessions = 0

    for sf in session_files:
        session_id = sf.stem
        if session_id in processed:
            continue

        events = extract_session(sf)
        mark_processed(session_id, events)

        if events > 0:
            new_sessions += 1
            total_events += events

    print(json.dumps({
        "sessions_scanned": len(session_files),
        "sessions_new": new_sessions,
        "sessions_skipped": len(session_files) - new_sessions,
        "events_extracted": total_events,
    }))


if __name__ == "__main__":
    main()
