"""
run_store.py
============
SQLite database layer. Stores every analysis run from upload to delivery.

WHY WE STORE RUNS IN A DATABASE:

  1. Caching — if the same 10-K is uploaded twice, we return the stored
     result immediately instead of re-spending $2 and 90 seconds on GPT-4o.
     The cache key is a SHA256 hash of the raw file bytes, so identical
     files always match regardless of their filename.

  2. Export on demand — the Excel and Word export buttons work by reading
     the stored extracted_json from a previous run. The analyst can run
     the analysis once and export to different formats without re-uploading.

  3. Audit trail — every run is recorded with its status, timestamp, and
     completeness score for debugging and quality monitoring.

STATUS LIFECYCLE:
  queued → running → completed
                  ↘ failed

WAL MODE (Write-Ahead Logging):
  SQLite is set to WAL mode. This allows one writer and many readers
  simultaneously without blocking. Without WAL, concurrent API requests
  would queue up and time out under load.

SCHEMA EVOLUTION:
  New columns are added with ALTER TABLE (not DROP + RECREATE), so existing
  rows load correctly after a schema change — new columns return None for
  old rows. The _ensure_column helper handles this safely.
"""

from __future__ import annotations

import sqlite3
import json
import time
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Path to the SQLite database file.
# Override via DB_PATH environment variable for Railway volume mounts.
DB_PATH = "credit_ai_runs.sqlite3"


def _now_ts() -> int:
    """Returns the current Unix timestamp as an integer. Used for created_at / updated_at."""
    return int(time.time())


def _sha256(b: bytes) -> str:
    """
    Compute a SHA256 fingerprint of the raw file bytes.

    This is the cache key. Two uploads of the identical filing produce the
    same hash and hit the cache. One changed byte produces a completely
    different hash, ensuring modified files are always re-analysed.
    """
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


@dataclass
class RunRecord:
    """
    A single analysis run as loaded from the database.

    This dataclass mirrors the `runs` table schema. Fields marked
    Optional[...] may be None for older rows that predate that column
    being added (backward-compatible schema evolution).

    Key fields:
        run_id:         Unique UUID for this run (primary key)
        status:         Current lifecycle state: queued/running/completed/failed
        file_sha256:    SHA256 of the uploaded file — used as the cache key
        extracted_json: The full GPT-4o extraction result as a JSON dict
        memo_markdown:  The generated underwriting memo text
        mda_summary:    Claude's MD&A segment driver analysis (stored so
                        Word export doesn't need to re-run the LLM)
        completeness:   Score 0.0–1.0 indicating extraction completeness
    """
    run_id: str
    status: str
    filename: str
    file_sha256: str
    created_at: int
    updated_at: int
    error: Optional[str]
    extracted_json: Optional[Dict[str, Any]]
    validation_flags: Optional[list]
    completeness: Optional[float]
    excerpt_preview: Optional[str]
    model_raw_preview: Optional[str]
    memo_markdown: Optional[str]
    mda_summary: Optional[str] = None   # Persisted so Word export doesn't need re-extraction
    model_name: Optional[str] = None
    prompt_version: Optional[str] = None
    statement_basis: Optional[str] = None
    period_count: Optional[int] = None


class RunStore:
    """
    SQLite-backed store for analysis runs.

    All database access goes through this class. It handles:
      - Table creation and schema evolution on startup
      - SHA256 cache lookups (find_completed_by_file_sha)
      - Run lifecycle management (create → running → completed/failed)
      - Backward-compatible column additions (_ensure_column)
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        """
        Open a SQLite connection with:
          - row_factory = sqlite3.Row so columns can be accessed by name
          - busy_timeout = 30 seconds so concurrent requests wait rather than crash
        """
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000;")
        return conn

    def _ensure_column(self, conn: sqlite3.Connection, table: str, col: str, col_def: str) -> None:
        """
        Add a column to a table if it doesn't already exist.
        This is how we evolve the schema without dropping existing data.
        Called during _init_db for every column added after the initial launch.
        """
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")

    def _init_db(self) -> None:
        """
        Create the runs table if it doesn't exist, then add any columns
        that were introduced after the initial schema.

        Safe to call on every startup — CREATE TABLE IF NOT EXISTS and
        ALTER TABLE (via _ensure_column) are idempotent.
        """
        with self._conn() as conn:
            # Enable WAL mode for concurrent read/write access
            conn.execute("PRAGMA journal_mode=WAL;")
            # NORMAL sync is safe with WAL and much faster than FULL
            conn.execute("PRAGMA synchronous=NORMAL;")

            # Core table — created on first ever startup
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_sha256 TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    error TEXT,
                    extracted_json TEXT,
                    validation_flags TEXT,
                    completeness REAL,
                    excerpt_preview TEXT,
                    model_raw_preview TEXT
                )
                """
            )
            # Index on file_sha256 makes cache lookups fast even with many rows
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_file_sha ON runs(file_sha256)")

            # Columns added after the initial schema — safe to run every time
            self._ensure_column(conn, "runs", "model_name", "TEXT")
            self._ensure_column(conn, "runs", "prompt_version", "TEXT")
            self._ensure_column(conn, "runs", "statement_basis", "TEXT")
            self._ensure_column(conn, "runs", "period_count", "INTEGER")
            self._ensure_column(conn, "runs", "memo_markdown", "TEXT")
            self._ensure_column(conn, "runs", "mda_summary",   "TEXT")

            conn.commit()

    def create_run(self, run_id: str, filename: str, file_bytes: bytes) -> str:
        """
        Register a new run in the database with status='queued'.

        Returns the SHA256 hash of the file, which is also the cache key.
        The hash is computed here so it can be used immediately to check
        the cache before starting any expensive processing.
        """
        ts = _now_ts()
        file_sha = _sha256(file_bytes)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, status, filename, file_sha256, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, "queued", filename, file_sha, ts, ts),
            )
            conn.commit()

        return file_sha

    def find_completed_by_file_sha(self, file_sha256: str) -> Optional[RunRecord]:
        """
        Look up a previously completed run by the SHA256 hash of the file.

        This is the cache check. If a completed run exists for this exact
        file, we return it immediately without re-running the extraction.
        We take the most recently updated completed run in case the same
        file was analysed multiple times with different borrower contexts.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE file_sha256 = ? AND status = 'completed' ORDER BY updated_at DESC LIMIT 1",
                (file_sha256,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def set_running(self, run_id: str) -> None:
        """Mark a run as actively being processed. Called after cache miss."""
        if not run_id:
            return
        self._update(run_id, status="running", updated_at=_now_ts())

    def set_failed(
        self,
        run_id: str,
        error: str,
        excerpt_preview: Optional[str] = None,
        model_raw_preview: Optional[str] = None,
    ) -> None:
        """Mark a run as failed with an error message. Shown in the frontend as an error banner."""
        if not run_id:
            return
        self._update(
            run_id,
            status="failed",
            updated_at=_now_ts(),
            error=error,
            excerpt_preview=excerpt_preview,
            model_raw_preview=model_raw_preview,
        )

    def set_completed(
        self,
        run_id: str,
        extracted_json: Dict[str, Any],
        validation_flags: list,
        completeness: float,
        excerpt_preview: Optional[str] = None,
        model_raw_preview: Optional[str] = None,
        memo_markdown: Optional[str] = None,
        mda_summary: Optional[str] = None,
        model_name: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> None:
        """
        Store the completed extraction results.

        Saves the full extracted_json blob, the memo, the MD&A summary,
        and some convenience fields (statement_basis, period_count) that
        are extracted from the JSON for quick access without re-parsing.

        mda_summary is stored separately from memo_markdown because the
        Word export needs the raw MD&A analysis to build segment sections,
        while the memo_markdown is the formatted output shown in the UI.
        """
        if not run_id:
            return

        # Extract convenience fields from the JSON blob
        statement_basis = extracted_json.get("statement_basis")
        periods = extracted_json.get("periods") or []
        period_count = len(periods) if isinstance(periods, list) else None

        self._update(
            run_id,
            status="completed",
            updated_at=_now_ts(),
            extracted_json=json.dumps(extracted_json),
            validation_flags=json.dumps(validation_flags),
            completeness=float(completeness) if completeness is not None else None,
            excerpt_preview=excerpt_preview,
            model_raw_preview=model_raw_preview,
            memo_markdown=memo_markdown,
            mda_summary=mda_summary,
            error=None,
            model_name=model_name,
            prompt_version=prompt_version,
            statement_basis=statement_basis,
            period_count=period_count,
        )

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Retrieve a run by its UUID. Used by the export endpoints."""
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def _update(self, run_id: str, **fields) -> None:
        """
        Generic update helper — builds a SET clause from keyword arguments.
        Avoids having to write separate UPDATE statements for every status transition.
        """
        keys = list(fields.keys())
        if not keys:
            return
        assignments = ", ".join([f"{k} = ?" for k in keys])
        values = [fields[k] for k in keys]

        with self._conn() as conn:
            conn.execute(f"UPDATE runs SET {assignments} WHERE run_id = ?", (*values, run_id))
            conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> RunRecord:
        """
        Convert a raw SQLite row into a typed RunRecord dataclass.

        Uses a safe _get helper so that columns added after the initial
        schema (which return KeyError on old rows) gracefully return None
        instead of crashing.
        """
        extracted = json.loads(row["extracted_json"]) if row["extracted_json"] else None
        flags = json.loads(row["validation_flags"]) if row["validation_flags"] else None

        def _get(col: str, default=None):
            try:
                return row[col]
            except Exception:
                return default

        return RunRecord(
            run_id=row["run_id"],
            status=row["status"],
            filename=row["filename"],
            file_sha256=row["file_sha256"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            error=row["error"],
            extracted_json=extracted,
            validation_flags=flags,
            completeness=row["completeness"],
            excerpt_preview=row["excerpt_preview"],
            model_raw_preview=row["model_raw_preview"],
            memo_markdown=_get("memo_markdown"),
            mda_summary=_get("mda_summary"),
            model_name=_get("model_name"),
            prompt_version=_get("prompt_version"),
            statement_basis=_get("statement_basis"),
            period_count=_get("period_count"),
        )


def compute_completeness(extracted_json: Dict[str, Any]) -> float:
    """
    Score extraction completeness on a 0.0 to 1.0 scale.

    WHAT THIS MEASURES:
    Not "how many fields were found" but "can we compute the three metrics
    a credit analyst needs most?" A filing that has revenue, EBITDA, and
    leverage scores 1.0 even if rent expense is missing. A filing with 20
    fields but no total_debt scores less than 0.65.

    COMPONENTS (must add to 1.0):
      0.35 — EBITDA is computable (operating income + D&A, or labeled EBITDA)
      0.35 — Leverage is computable (requires EBITDA > 0 AND total_debt)
      0.30 — FCC is computable (requires EBITDA, capex, cash taxes, CPLTD, cash interest)

    HOW IT'S USED:
    The frontend shows the score in the summary bar.
    If score < 0.55, a warning banner is prepended to the underwriting memo.
    The cache check uses it to decide if a cached result is "good enough"
    to return (cache_is_usable returns True only if score would likely be >= 0.55).

    Scores the BEST period (most complete year) rather than the average,
    because one complete year is sufficient for a credit decision.
    """
    try:
        from metrics import compute_ebitda, compute_fcc, compute_leverage

        periods = extracted_json.get("periods") or []
        if not isinstance(periods, list) or not periods:
            return 0.0

        def score_period(p: Dict[str, Any]) -> float:
            if not isinstance(p, dict):
                return 0.0
            is_ = p.get("income_statement") or {}
            bs = p.get("balance_sheet") or {}
            cf = p.get("cash_flow") or {}

            # Try to get EBITDA directly, or compute it from components
            ebitda = is_.get("ebitda")
            if not isinstance(ebitda, (int, float)):
                ebitda = compute_ebitda(
                    is_.get("operating_income"),
                    is_.get("depreciation_amortization"),
                    rent_expense=is_.get("rent_expense"),
                    include_rent=False,
                )
            ebitda_ok = isinstance(ebitda, (int, float))

            lev_ok = compute_leverage(bs.get("total_debt"), ebitda if ebitda_ok else None) is not None

            fcc_ok = compute_fcc(
                ebitda=ebitda if ebitda_ok else None,
                capex=cf.get("capex"),
                cash_taxes=cf.get("cash_paid_for_income_taxes"),
                cpltd=bs.get("current_portion_long_term_debt"),
                cash_interest=cf.get("cash_paid_for_interest"),
            ) is not None

            return (0.35 if ebitda_ok else 0.0) + (0.35 if lev_ok else 0.0) + (0.30 if fcc_ok else 0.0)

        # Score each period and return the best — one good year is enough
        best = max(score_period(p) for p in periods)
        return float(max(0.0, min(1.0, best)))
    except Exception:
        return 0.0
