#!/usr/bin/env python3
"""
Reconsolidation Pipeline — Access-driven memory reshaping.

Connects: access_logger (events) → vmem (embeddings) → DCT reconsolidation → updated vmem

The pipeline:
  1. Read access energy from access.db (which chunks get used, how often, how strongly)
  2. Load embeddings from vmem's memory.db
  3. Run DCT reconsolidation: amplify accessed embeddings before transform,
     truncate, inverse — accessed chunks survive compression better
  4. Write reconsolidated embeddings back to vmem
  5. Track metrics for delta monitoring

Run from heartbeat/cron. Idempotent — safe to run repeatedly.

Usage:
  python pipeline.py run                    # Full reconsolidation pass
  python pipeline.py run --dry-run          # Show what would change, don't write
  python pipeline.py metrics                # Show reconsolidation history
  python pipeline.py energy                 # Show current access energy map
"""

import json
import sqlite3
import struct
import sys
import time
import numpy as np
from pathlib import Path
from scipy.fft import dct, idct
from typing import Optional

# Paths
TOOLS_DIR = Path(__file__).parent.parent
ACCESS_DB = Path(__file__).parent / "access.db"
VMEM_DB = TOOLS_DIR / "vectordb" / "memory.db"
METRICS_DB = Path(__file__).parent / "metrics.db"
EMBEDDING_DIM = 384  # BGE-small-en-v1.5


def serialize_f32(vector: list[float]) -> bytes:
    """Serialize float32 vector for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def deserialize_f32(blob: bytes) -> np.ndarray:
    """Deserialize float32 vector from sqlite-vec."""
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def init_metrics_db():
    db = sqlite3.connect(str(METRICS_DB))
    db.execute("""
        CREATE TABLE IF NOT EXISTS reconsolidation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            n_chunks INTEGER,
            n_with_energy INTEGER,
            k_coefficients INTEGER,
            keep_ratio REAL,
            promotion_strength REAL,
            avg_sim_before REAL,
            avg_sim_after REAL,
            avg_delta REAL,
            max_promoted_delta REAL,
            max_demoted_delta REAL,
            total_access_events INTEGER,
            details TEXT  -- JSON with per-chunk deltas
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS retrieval_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            query TEXT,
            used_chunk_rank INTEGER,  -- rank of chunk actually referenced in response
            total_results INTEGER,
            session_id TEXT
        )
    """)
    db.commit()
    return db


def load_access_energy(half_life_hours: float = 168.0) -> dict:
    """
    Load access energy from access.db.
    Returns {chunk_key: energy} where energy is decayed by recency.
    
    half_life_hours=168 (1 week): access from a week ago = half weight of now.
    """
    if not ACCESS_DB.exists():
        return {}

    db = sqlite3.connect(str(ACCESS_DB))
    now = time.time()
    decay_rate = np.log(2) / (half_life_hours * 3600)

    rows = db.execute(
        "SELECT chunk_key, total_accesses, total_score, last_accessed, first_accessed "
        "FROM chunk_energy WHERE total_accesses > 0"
    ).fetchall()
    db.close()

    energy = {}
    for chunk_key, accesses, total_score, last_accessed, first_accessed in rows:
        age = now - (last_accessed or now)
        decay = np.exp(-decay_rate * age)
        # Energy combines frequency (accesses), strength (score), and recency (decay)
        e = (total_score / max(accesses, 1)) * accesses * decay
        energy[chunk_key] = float(e)

    # Normalize to [0, 1]
    if energy:
        max_e = max(energy.values())
        if max_e > 0:
            energy = {k: v / max_e for k, v in energy.items()}

    return energy


def load_vmem_chunks() -> tuple[list[dict], np.ndarray]:
    """Load chunks and their embeddings from vmem."""
    if not VMEM_DB.exists():
        return [], np.array([])

    db = sqlite3.connect(str(VMEM_DB))
    db.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    
    # Get chunks
    chunks = db.execute(
        "SELECT id, file_path, content, line_start, line_end "
        "FROM chunks ORDER BY id"
    ).fetchall()

    if not chunks:
        db.close()
        return [], np.array([])

    # Get embeddings (from chunk_embeddings virtual table)
    # sqlite-vec stores embeddings — we need to read them
    chunk_data = []
    embeddings = []

    for chunk_id, file_path, content, line_start, line_end in chunks:
        # Read embedding from the virtual table
        row = db.execute(
            "SELECT embedding FROM chunk_embeddings WHERE chunk_id = ?",
            (chunk_id,)
        ).fetchone()

        if row:
            emb = deserialize_f32(row[0])
            chunk_data.append({
                "id": chunk_id,
                "file_path": file_path,
                "content": content,
                "line_start": line_start,
                "line_end": line_end,
                "chunk_key": f"{file_path}:{line_start}",
            })
            embeddings.append(emb)

    db.close()
    return chunk_data, np.array(embeddings) if embeddings else np.array([])


def reconsolidate(
    keep_ratio: float = 0.15,
    promotion_strength: float = 1.5,
    dry_run: bool = False
) -> dict:
    """
    Run the reconsolidation pipeline.
    
    1. Load vmem embeddings + access energy
    2. Weight embeddings by access energy
    3. DCT → truncate → IDCT (accessed memories survive compression)
    4. Write back to vmem (unless dry_run)
    5. Log metrics
    """
    chunks, E = load_vmem_chunks()
    if len(chunks) == 0:
        return {"error": "no chunks in vmem", "action": "none"}

    N, D = E.shape
    energy_map = load_access_energy()

    if not energy_map:
        return {
            "error": "no access events yet",
            "action": "none",
            "n_chunks": N,
            "hint": "Run some sessions first — access events are logged at compaction"
        }

    # Build energy vector aligned to chunk order
    energy = np.zeros(N)
    n_with_energy = 0
    for i, chunk in enumerate(chunks):
        e = energy_map.get(chunk["chunk_key"], 0.0)
        energy[i] = e
        if e > 0:
            n_with_energy += 1

    if n_with_energy == 0:
        return {
            "action": "none",
            "n_chunks": N,
            "access_events": len(energy_map),
            "hint": "Access events exist but none match current vmem chunks (chunk keys may have shifted after reindex)"
        }

    # --- Standard DCT (baseline) ---
    k = max(1, int(N * keep_ratio))
    C_standard = dct(E, axis=0, norm='ortho')
    C_trunc = np.zeros_like(C_standard)
    C_trunc[:k] = C_standard[:k]
    R_standard = idct(C_trunc, axis=0, norm='ortho')

    # --- Promoted DCT ---
    weight = 1.0 + promotion_strength * energy
    E_weighted = E * weight[:, np.newaxis]

    C_promoted = dct(E_weighted, axis=0, norm='ortho')
    C_ptrunc = np.zeros_like(C_promoted)
    C_ptrunc[:k] = C_promoted[:k]
    R_promoted_raw = idct(C_ptrunc, axis=0, norm='ortho')
    R_promoted = R_promoted_raw / weight[:, np.newaxis]

    # Compute similarities (original vs reconstructed)
    sims_before = np.array([
        float(np.dot(E[i], R_standard[i]) / (np.linalg.norm(E[i]) * np.linalg.norm(R_standard[i]) + 1e-10))
        for i in range(N)
    ])
    sims_after = np.array([
        float(np.dot(E[i], R_promoted[i]) / (np.linalg.norm(E[i]) * np.linalg.norm(R_promoted[i]) + 1e-10))
        for i in range(N)
    ])
    delta = sims_after - sims_before

    # Top movers
    promoted_idx = np.argsort(delta)[-10:][::-1]
    demoted_idx = np.argsort(delta)[:5]

    result = {
        "action": "reconsolidated" if not dry_run else "dry_run",
        "n_chunks": N,
        "n_with_energy": n_with_energy,
        "k_coefficients": k,
        "keep_ratio": keep_ratio,
        "promotion_strength": promotion_strength,
        "avg_sim_before": float(np.mean(sims_before)),
        "avg_sim_after": float(np.mean(sims_after)),
        "avg_delta": float(np.mean(delta)),
        "promoted": [
            {
                "chunk_key": chunks[int(i)]["chunk_key"],
                "content": chunks[int(i)]["content"][:80],
                "energy": float(energy[i]),
                "delta": float(delta[i]),
            }
            for i in promoted_idx if delta[i] > 0.001
        ],
        "demoted": [
            {
                "chunk_key": chunks[int(i)]["chunk_key"],
                "content": chunks[int(i)]["content"][:80],
                "energy": float(energy[i]),
                "delta": float(delta[i]),
            }
            for i in demoted_idx if delta[i] < -0.001
        ],
    }

    if not dry_run:
        # Write reconsolidated embeddings back to vmem
        _write_embeddings(chunks, R_promoted)

        # Log metrics
        metrics_db = init_metrics_db()
        metrics_db.execute(
            "INSERT INTO reconsolidation_runs "
            "(timestamp, n_chunks, n_with_energy, k_coefficients, keep_ratio, "
            "promotion_strength, avg_sim_before, avg_sim_after, avg_delta, "
            "max_promoted_delta, max_demoted_delta, total_access_events, details) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(), N, n_with_energy, k, keep_ratio,
                promotion_strength,
                float(np.mean(sims_before)), float(np.mean(sims_after)),
                float(np.mean(delta)),
                float(np.max(delta)), float(np.min(delta)),
                sum(1 for v in energy_map.values() if v > 0),
                json.dumps(result["promoted"][:5] + result["demoted"][:3])
            )
        )
        metrics_db.commit()
        metrics_db.close()

    return result


def _write_embeddings(chunks: list[dict], embeddings: np.ndarray):
    """Write reconsolidated embeddings back to vmem's sqlite-vec table."""
    db = sqlite3.connect(str(VMEM_DB))
    db.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    for i, chunk in enumerate(chunks):
        emb = embeddings[i].astype(np.float32)
        # sqlite-vec UPDATE: delete and re-insert the embedding
        db.execute(
            "DELETE FROM chunk_embeddings WHERE chunk_id = ?",
            (chunk["id"],)
        )
        db.execute(
            "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
            (chunk["id"], serialize_f32(emb.tolist()))
        )

    db.commit()
    db.close()


def metrics(limit: int = 20) -> list:
    """Show reconsolidation history."""
    if not METRICS_DB.exists():
        return []
    db = init_metrics_db()
    rows = db.execute(
        "SELECT timestamp, n_chunks, n_with_energy, avg_sim_before, avg_sim_after, "
        "avg_delta, max_promoted_delta, total_access_events "
        "FROM reconsolidation_runs ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    ).fetchall()
    db.close()
    return [
        {
            "timestamp": r[0],
            "chunks": r[1],
            "with_energy": r[2],
            "sim_before": round(r[3], 4),
            "sim_after": round(r[4], 4),
            "delta": round(r[5], 5),
            "max_promoted": round(r[6], 5),
            "access_events": r[7],
        }
        for r in rows
    ]


def energy_map() -> list:
    """Show current access energy for all tracked chunks."""
    energy = load_access_energy()
    if not energy:
        return []
    sorted_energy = sorted(energy.items(), key=lambda x: x[1], reverse=True)
    return [{"chunk": k, "energy": round(v, 4)} for k, v in sorted_energy]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pipeline.py <run|metrics|energy> [--dry-run]")
        sys.exit(1)

    cmd = sys.argv[1]
    dry_run = "--dry-run" in sys.argv

    if cmd == "run":
        result = reconsolidate(dry_run=dry_run)
        print(json.dumps(result, indent=2))

    elif cmd == "metrics":
        for m in metrics():
            print(json.dumps(m))

    elif cmd == "energy":
        for e in energy_map():
            print(f"  {e['energy']:.4f}  {e['chunk']}")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
