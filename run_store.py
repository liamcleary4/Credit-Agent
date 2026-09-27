from __future__ import annotations

import sqlite3
import json
import time
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, Any

DB_PATH = "credit_ai_runs.sqlite3"


def _now_ts() -> int:
    return int(time.time())


def _sha256(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


@dataclass
class RunRecord:
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
    mda_summary: Optional[str] = None   # persisted so export-on-demand doesn't need re-extraction
    # Optional metadata (backward-compatible)
    model_name: Optional[str] = None
    prompt_version: Optional[str] = None
    statement_basis: Optional[str] = None
    period_count: Optional[int] = None


class RunStore:
    """
    SQLite-backed run store.
    Upgrades:
      - WAL + busy_timeout for concurrency
      - Backward-compatible schema evolution (ALTER TABLE if new cols missing)
      - Stores memo_markdown
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000;")
        return conn

    def _ensure_column(self, conn: sqlite3.Connection, table: str, col: str, col_def: str) -> None:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")

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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_file_sha ON runs(file_sha256)")

            # Backward-compatible schema evolution
            self._ensure_column(conn, "runs", "model_name", "TEXT")
            self._ensure_column(conn, "runs", "prompt_version", "TEXT")
            self._ensure_column(conn, "runs", "statement_basis", "TEXT")
            self._ensure_column(conn, "runs", "period_count", "INTEGER")
            self._ensure_column(conn, "runs", "memo_markdown", "TEXT")
            self._ensure_column(conn, "runs", "mda_summary",   "TEXT")

            conn.commit()

    def create_run(self, run_id: str, filename: str, file_bytes: bytes) -> str:
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
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE file_sha256 = ? AND status = 'completed' ORDER BY updated_at DESC LIMIT 1",
                (file_sha256,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def set_running(self, run_id: str) -> None:
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
        if not run_id:
            return

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
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def _update(self, run_id: str, **fields) -> None:
        keys = list(fields.keys())
        if not keys:
            return
        assignments = ", ".join([f"{k} = ?" for k in keys])
        values = [fields[k] for k in keys]

        with self._conn() as conn:
            conn.execute(f"UPDATE runs SET {assignments} WHERE run_id = ?", (*values, run_id))
            conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> RunRecord:
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


# Scores 0.0-1.0: can the three key metrics be computed?
def compute_completeness(extracted_json: Dict[str, Any]) -> float:
    """
    Metric-aligned quality gate computed over the best available period.

    Score components (max 1.0):
      - 0.35: EBITDA available or computable from (operating_income, depreciation_amortization)
      - 0.35: Leverage computable (total_debt / EBITDA), requires EBITDA > 0
      - 0.30: FCC computable per your definition
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

        best = max(score_period(p) for p in periods)
        return float(max(0.0, min(1.0, best)))
    except Exception:
        return 0.0
