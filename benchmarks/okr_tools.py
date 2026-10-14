"""Harness-owned bookkeeping tools, shared by the baseline arms.

AtomAgents binds four in-process tools alongside the 13 MCP ones. They are not
reusable outside it: they close over a live agent's session state (the memory
handle, the OKR board, the current job id), none of which exists in a ReAct
loop or a CrewAI crew. Reimplementing them here is therefore not a convenience
-- it is the only way the three arms can be scored by the same code.

What matters for the comparison is the *rows*, not the implementation: the
scorer reads `artefacts`, `validation_metrics` and `attempts`, and those tables
are created here with byte-identical schemas to the ones AtomAgents writes (see
`sqlite3 runs/jobs/session-*/memory.db .schema`). A baseline that registers a
deliverable therefore lands a row the scorer cannot distinguish by shape.

Deliberately NOT reproduced: the OKR board ritual (tasks/subtasks/key_results
progression). That is treatment, not environment -- see the plan's section 5
boundary, "what the tool does" vs "what you should do with it".

Every function returns a JSON string, matching the MCP tools' convention so the
agent sees one uniform result shape.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator, Optional

# Mirrors AtomAgents' own DDL. `path` is the primary key there too, so
# re-registering a file updates it rather than duplicating it.
_DDL = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    msg_json TEXT,
    msg_text TEXT,
    job_id TEXT,
    objective_id INTEGER,
    key_result_id INTEGER,
    embedding TEXT,
    agent_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_agent ON messages(agent_name);
CREATE TABLE IF NOT EXISTS artefacts (
    path TEXT PRIMARY KEY,
    kind TEXT,
    bytes INTEGER,
    hash TEXT,
    kr_id INTEGER,
    task_id INTEGER,
    created_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS validation_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric TEXT,
    value REAL,
    unit TEXT,
    job_id TEXT,
    kr_id INTEGER,
    created_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subtask_id INTEGER,
    description TEXT,
    command_hash TEXT,
    result TEXT,
    error_excerpt TEXT,
    created_at TIMESTAMP
);
"""


def _now() -> str:
    # UTC, matching sqlite's CURRENT_TIMESTAMP that stamps AtomAgents' rows —
    # otherwise identical tables carry a silent timezone skew between arms.
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def _ok(**kw: Any) -> str:
    return json.dumps({"status": "ok", **kw})


def _err(msg: str, **kw: Any) -> str:
    return json.dumps({"status": "error", "error": msg, **kw})


class OKRStore:
    """SQLite-backed bookkeeping for one benchmark run.

    One store per run, living in the run's own session directory so the scorer
    finds it exactly where it finds AtomAgents' memory.db.
    """

    def __init__(self, db_path: str | os.PathLike[str], project_root: str | os.PathLike[str]):
        self.db_path = Path(db_path)
        self.project_root = Path(project_root).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._con() as con:
            con.executescript(_DDL)

    @contextlib.contextmanager
    def _con(self) -> Iterator[sqlite3.Connection]:
        # Closed after every use, not just committed: a connection left open at
        # process exit strands the WAL sidecars (memory.db-shm/-wal) on disk,
        # and the runner's session scoring must never see database plumbing.
        con = sqlite3.connect(self.db_path, timeout=30)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            with con:
                yield con
        finally:
            con.close()

    # -- tools -------------------------------------------------------------

    def register_artefact(self, path: str, kind: str = "", note: str = "") -> str:
        """Record a deliverable file. Hashes it so the scorer can prove the
        bytes on disk are the bytes that were registered -- the same
        anti-fabrication check applied to the AtomAgents arm."""
        try:
            p = Path(path)
            if not p.is_absolute():
                p = self.project_root / p
            p = p.resolve()
            if not p.exists():
                return _err(f"no such file: {p}", path=str(p))
            if not p.is_file():
                return _err(f"not a regular file: {p}", path=str(p))
            data = p.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            # Recorded, never rejected: containment is a *measurement* in this
            # campaign (demo2-opus wrote outside its session dir twice), so a
            # tool that refused would destroy the signal it is meant to capture.
            # Path semantics, not string prefix — `session-x-evil` shares the
            # prefix of `session-x` and must not read as contained.
            contained = p.is_relative_to(self.project_root)
            # The schema is pinned to AtomAgents' (no note column), so the note
            # rides in `kind` rather than being silently dropped.
            kind_full = kind or p.suffix.lstrip(".") or "file"
            if note:
                kind_full = f"{kind_full}: {note[:200]}"
            with self._con() as con:
                con.execute(
                    "INSERT INTO artefacts (path, kind, bytes, hash, kr_id, task_id, created_at) "
                    "VALUES (?,?,?,?,NULL,NULL,?) "
                    "ON CONFLICT(path) DO UPDATE SET "
                    "kind=excluded.kind, bytes=excluded.bytes, hash=excluded.hash, "
                    "created_at=excluded.created_at",
                    (str(p), kind_full, len(data), digest, _now()),
                )
            return _ok(path=str(p), bytes=len(data), hash=digest[:16], contained=contained)
        except Exception as exc:  # noqa: BLE001 - a tool must never kill the loop
            return _err(f"{type(exc).__name__}: {exc}")

    def record_metric(self, metric: str, value: float, unit: str = "", job_id: str = "") -> str:
        """Record a validation number (E, Kt, peak stress, error %)."""
        try:
            val = float(value)
        except (TypeError, ValueError):
            return _err(f"value must be numeric, got {value!r}")
        try:
            with self._con() as con:
                con.execute(
                    "INSERT INTO validation_metrics (metric, value, unit, job_id, kr_id, created_at) "
                    "VALUES (?,?,?,?,NULL,?)",
                    (metric, val, unit, job_id, _now()),
                )
            return _ok(metric=metric, value=val, unit=unit)
        except Exception as exc:  # noqa: BLE001
            return _err(f"{type(exc).__name__}: {exc}")

    def record_attempt(self, description: str, result: str = "", error_excerpt: str = "") -> str:
        """Log an attempt -- especially a failed one -- so the transcript is not
        the only record that something was tried and did not work."""
        try:
            with self._con() as con:
                con.execute(
                    "INSERT INTO attempts (subtask_id, description, command_hash, result, "
                    "error_excerpt, created_at) VALUES (NULL,?,?,?,?,?)",
                    (
                        description,
                        hashlib.sha256(description.encode("utf-8")).hexdigest()[:16],
                        result,
                        (error_excerpt or "")[:2000],
                        _now(),
                    ),
                )
            return _ok(recorded=description[:80])
        except Exception as exc:  # noqa: BLE001
            return _err(f"{type(exc).__name__}: {exc}")

    def list_artefacts(self) -> str:
        """Read back what has been registered so far. The fourth tool: without
        it the agent can only append, never check its own bookkeeping."""
        try:
            with self._con() as con:
                rows = con.execute(
                    "SELECT path, kind, bytes, hash, created_at FROM artefacts ORDER BY created_at"
                ).fetchall()
            return _ok(
                count=len(rows),
                artefacts=[
                    {"path": r[0], "kind": r[1], "bytes": r[2], "hash": r[3][:16], "at": r[4]}
                    for r in rows
                ],
            )
        except Exception as exc:  # noqa: BLE001
            return _err(f"{type(exc).__name__}: {exc}")

    # -- transcript --------------------------------------------------------

    def log_message(self, agent_name: str, msg_json: str, msg_text: str, job_id: str = "") -> None:
        """Mirror one conversation message into the same `messages` table
        AtomAgents writes.

        This is what lets `runner.session_stats()` report `messages` and the
        per-agent breakdown for a baseline exactly as it does for ARM A, so the
        three manifests stay field-comparable instead of the baselines showing
        structural zeros that look like failures.

        `embedding` is deliberately left empty. The baselines have no semantic
        recall, so `vectors == 0` is the honest reading of an architectural
        difference -- not the silent-embedding bug that invalidated WS26.
        """
        try:
            with self._con() as con:
                con.execute(
                    "INSERT INTO messages (msg_json, msg_text, job_id, objective_id, "
                    "key_result_id, embedding, agent_name, created_at) "
                    "VALUES (?,?,?,NULL,NULL,'',?,?)",
                    (msg_json, msg_text, job_id, agent_name, _now()),
                )
        except Exception:  # noqa: BLE001 - telemetry must never break a run
            pass

    # -- scoring support ---------------------------------------------------

    def counts(self) -> dict[str, int]:
        with self._con() as con:
            return {
                t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                for t in ("artefacts", "validation_metrics", "attempts")
            }


#: Names and one-line descriptions, shared by both tool adapters so the two
#: baselines advertise byte-identical tool surfaces.
OKR_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "register_artefact",
        "method": "register_artefact",
        "description": (
            "Register a deliverable file you produced. Pass the absolute path. "
            "The file is hashed and recorded. Call this for every file the brief asks for."
        ),
        "args": {
            "path": (str, ..., "Absolute path to the file you produced."),
            "kind": (str, "", "Short kind label, e.g. 'script', 'data', 'figure', 'report'."),
            "note": (str, "", "Optional one-line note about what the file contains."),
        },
    },
    {
        "name": "record_metric",
        "method": "record_metric",
        "description": (
            "Record a validation number your run produced, e.g. Young's modulus, "
            "stress-concentration factor, peak stress, or a relative error."
        ),
        "args": {
            "metric": (str, ..., "Metric name, e.g. 'youngs_modulus' or 'Kt'."),
            "value": (float, ..., "The numeric value, from a real tool result."),
            "unit": (str, "", "Unit, e.g. 'GPa', 'percent', '' if dimensionless."),
            "job_id": (str, "", "Optional label tying the metric to a particular run."),
        },
    },
    {
        "name": "record_attempt",
        "method": "record_attempt",
        "description": (
            "Log an attempt and its outcome. Use it especially when something "
            "failed, so the failure is on the record rather than only in your reply."
        ),
        "args": {
            "description": (str, ..., "What you tried, in one line."),
            "result": (str, "", "Outcome: 'success', 'failed', or a short summary."),
            "error_excerpt": (str, "", "The error text, if any."),
        },
    },
    {
        "name": "list_artefacts",
        "method": "list_artefacts",
        "description": "List the deliverables you have registered so far, with sizes and hashes.",
        "args": {},
    },
]
