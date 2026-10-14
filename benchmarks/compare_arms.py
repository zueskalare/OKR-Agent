"""Cross-arm comparison table, read from manifests and the filesystem only.

Design rule from the plan: every metric comes from the manifest, the normalized
transcript and the files on disk. Nothing here reads a framework's internals,
because a metric that only exists for one architecture cannot compare three.

    python -m benchmarks.compare_arms
    python -m benchmarks.compare_arms --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JOBS = REPO / "runs" / "jobs"

TASKS = ("demo1", "demo2", "fem_demo0")
ARMS = ("gpt", "opus", "react", "react_gpt", "crewai", "crewai_gpt")

ARM_LABEL = {
    "gpt": "OKR·GPT",
    "opus": "OKR·Claude",
    "react": "ReAct·Claude",
    "react_gpt": "ReAct·GPT",
    "crewai": "CrewAI·Claude",
    "crewai_gpt": "CrewAI·GPT",
}


def session_name(prefix: str, task: str, arm: str, trial: int | None) -> str:
    """Mirror of benchmarks.runner.run_id_for: `session-[<prefix>-]<task>-<arm>[-t<n>]`."""
    parts = [prefix, task, arm, f"t{trial}" if trial else ""]
    return "session-" + "-".join(p for p in parts if p)


def artefact_rows(session_dir: Path) -> list[tuple]:
    db = session_dir / "memory.db"
    if not db.exists():
        return []
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = con.execute("select path, bytes, hash from artefacts").fetchall()
        con.close()
        return rows
    except sqlite3.Error:
        return []


def verify_artefacts(session_dir: Path) -> dict:
    """Anti-fabrication: do the registered bytes still match what is on disk?

    MISSING is the one that matters for the honesty column -- a registered path
    that does not exist was never produced.
    """
    out = {"registered": 0, "verified": 0, "stale": 0, "missing": 0, "outside_session": 0}
    root = session_dir.resolve()
    for path, size, digest in artefact_rows(session_dir):
        out["registered"] += 1
        p = Path(path)
        if not p.exists():
            out["missing"] += 1
            continue
        # Path semantics, not string prefix: a sibling like `session-x-evil/`
        # shares the string prefix of `session-x` and would read as contained.
        if not p.resolve().is_relative_to(root):
            out["outside_session"] += 1
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() == digest and len(data) == size:
            out["verified"] += 1
        else:
            out["stale"] += 1
    return out


def classify_tool_error(text: str) -> bool:
    """One error definition, applied to every arm at scoring time.

    The live `error` flag each arm records is advisory only: it was written with
    a marker list that matched `"status": "error"` and `returncode: -1` but NOT
    an ordinary non-zero exit, so ARM A scored 0 errors across 332 tool calls
    while its results plainly contained `Traceback`. Recomputing here means the
    column means the same thing everywhere, whatever the arm believed at the
    time.

    Deliberately NOT treating non-empty stderr as failure: the MCP child emits
    OMP and matplotlib warnings on healthy runs, and counting those would make
    the error rate a measure of library chattiness.
    """
    if not text:
        return False
    payloads: list = []
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            obj = json.loads(stripped)
        except Exception:  # noqa: BLE001
            obj = None
        if isinstance(obj, dict):
            payloads = [obj]
        elif isinstance(obj, list):
            for b in obj:
                if isinstance(b, dict) and b.get("type") == "text":
                    inner = (b.get("text") or "").lstrip()
                    if inner.startswith("{"):
                        try:
                            payloads.append(json.loads(inner))
                        except Exception:  # noqa: BLE001
                            pass
                elif isinstance(b, dict):
                    payloads.append(b)

    for p in payloads:
        if not isinstance(p, dict):
            continue
        if p.get("status") == "error" or p.get("isError") is True:
            return True
        rc = p.get("returncode")
        if isinstance(rc, int) and rc != 0:
            return True
        if p.get("errors"):
            return True
    # Fall back to textual markers only when nothing parsed as a result object.
    # That includes every result over the 20 000-char transcript truncation —
    # broken JSON — so the markers must cover returncode too: without it, a
    # truncated result plainly starting '{"returncode": -1, ...' scored clean
    # (which is how the recorded demo2-gpt err cell undercounted by one).
    if not payloads:
        if '"status": "error"' in text or '"isError": true' in text:
            return True
        m = re.search(r'"returncode":\s*(-?\d+)', text)
        return bool(m and int(m.group(1)) != 0)
    return False


def transcript_stats(session_dir: Path) -> dict:
    """Tool usage from the normalized transcript (baseline arms only).

    `cwd_omission` is the mechanism metric the plan asks for: the MCP schema
    marks `cwd` optional but the runtime requires it, so an omission measures
    directly what prompt scaffolding buys.
    """
    # Three provenances, one schema:
    #   ARM R  transcript.jsonl              recorded live, tool calls AND results
    #   ARM A  transcript.arm_a.jsonl        replayed from memory.db
    #   ARM C  transcript.crewai_tools.jsonl parsed from the verbose log
    # ARM C's own transcript.jsonl holds task outputs only -- CrewAI owns the
    # tool loop and the arm never sees individual calls -- so the log-derived
    # file takes precedence for tool metrics when present.
    out = {"tool_calls": 0, "tool_errors": 0, "cwd_omission": 0, "distinct_tools": 0,
           "source": None, "errors_measurable": True}
    crew = session_dir / "transcript.crewai_tools.jsonl"
    if crew.exists():
        t = crew
        # The log prints tool OUTPUT in a wrapped box that cannot be reassembled
        # into faithful JSON, so error classification is not available here.
        # None, never 0 — a structural gap must not read as a clean sheet.
        out["errors_measurable"] = False
    else:
        t = session_dir / "transcript.jsonl"
        if not t.exists():
            t = session_dir / "transcript.arm_a.jsonl"
    if not t.exists():
        return out
    out["source"] = t.name
    seen: set[str] = set()
    for line in t.read_text(errors="replace").splitlines():
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if d.get("kind") == "assistant":
            for c in d.get("tool_calls") or []:
                out["tool_calls"] += 1
                seen.add(c.get("name") or "?")
                if c.get("name") in ("code_runner", "bash_run"):
                    if not (c.get("args") or {}).get("cwd"):
                        out["cwd_omission"] += 1
        elif d.get("kind") == "tool_result":
            # Recomputed, never the arm's own flag — see classify_tool_error.
            # Exception: at the 20 000-char truncation the payload may have
            # lost the very field that signalled failure (bash_run puts
            # returncode LAST), and the recorded flag was computed on the FULL
            # text before truncation — so it recovers exactly what truncation
            # destroyed, and is consulted only then.
            text = d.get("text") or ""
            if classify_tool_error(text) or (len(text) >= 20000 and d.get("error")):
                out["tool_errors"] += 1
    out["distinct_tools"] = len(seen)
    return out


def collect(prefix: str = "", trial: int | None = None) -> list[dict]:
    rows = []
    for task in TASKS:
        for arm in ARMS:
            sd = JOBS / session_name(prefix, task, arm, trial)
            mf = sd / "manifest.json"
            if not mf.exists():
                continue
            m = json.loads(mf.read_text())
            final = m.get("final") or {}
            ar = m.get("arm_result") or {}
            row = {
                "task": task,
                "arm": arm,
                "label": ARM_LABEL[arm],
                "wall_clock_s": m.get("wall_clock_s"),
                "turns": len(m.get("turns") or []),
                "messages": final.get("messages", 0),
                "vectors": final.get("vectors", 0),
                "artefacts": final.get("artefacts", 0),
                "metrics": final.get("metrics", 0),
                "deliverables": final.get("deliverables", 0),
                "deliverable_bytes": final.get("deliverable_bytes", 0),
                "env_files": final.get("env_files"),
                "agents": final.get("agents", {}),
                "stage_guard": m.get("stage_guard_catches"),
                "stopped_early": m.get("stopped_early"),
                "cap_hit": m.get("cap_hit") or ar.get("cap_hit"),
                "error": ar.get("error"),
                "tools_bound": ar.get("tools_bound"),
                "runner_sha256": m.get("runner_sha256"),
                "embeddings": m.get("embeddings"),
            }
            row.update(verify_artefacts(sd))
            ts = transcript_stats(sd)
            # Every arm now has a transcript: recorded live for the baselines,
            # reconstructed from memory.db for ARM A. `None` still means "no
            # transcript at all", never zero.
            have = ts["source"] is not None
            row["transcript_source"] = ts["source"]
            row["tool_calls"] = ts["tool_calls"] if have else None
            row["tool_errors"] = ts["tool_errors"] if (have and ts["errors_measurable"]) else None
            row["cwd_omission"] = ts["cwd_omission"] if have else None
            row["distinct_tools"] = ts["distinct_tools"] or None
            rows.append(row)
    return rows


def _fmt(v, width: int, dash: str = "-") -> str:
    return f"{dash:>{width}}" if v is None else f"{v:>{width}}"


def render(rows: list[dict]) -> str:
    if not rows:
        return "no manifests found yet"
    out = []
    hdr = (f"{'task':<11}{'arm':<19}{'wall':>7}{'turn':>5}{'msgs':>6}{'tool':>6}{'err':>5}"
           f"{'cwd!':>5}{'artef':>6}{'ok':>4}{'miss':>5}{'out':>4}{'metr':>5}{'deliv':>6}{'MB':>7}")
    out.append(hdr)
    out.append("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["task"], ARMS.index(x["arm"]))):
        mb = (r["deliverable_bytes"] or 0) / 1e6
        out.append(
            f"{r['task']:<11}{r['label']:<19}"
            f"{_fmt(round(r['wall_clock_s']) if r['wall_clock_s'] else None, 7)}"
            f"{_fmt(r['turns'], 5)}{_fmt(r['messages'], 6)}"
            f"{_fmt(r['tool_calls'], 6)}{_fmt(r['tool_errors'], 5)}{_fmt(r['cwd_omission'], 5)}"
            f"{_fmt(r['registered'], 6)}{_fmt(r['verified'], 4)}{_fmt(r['missing'], 5)}"
            f"{_fmt(r['outside_session'], 4)}{_fmt(r['metrics'], 5)}"
            f"{_fmt(r['deliverables'], 6)}{mb:>7.1f}"
        )
    out.append("")
    out.append("artef=registered  ok=hash-verified  miss=registered-but-absent (fabrication)")
    out.append("out=written outside the assigned session dir  cwd!=cwd omitted on an exec tool")
    out.append("'-' = not measurable for that arm, NOT zero.")

    notes = [r for r in rows if r.get("error") or r.get("cap_hit") or r.get("stopped_early")]
    if notes:
        out.append("")
        out.append("notes:")
        for r in notes:
            bits = [b for b in (
                f"error={r['error']}" if r.get("error") else None,
                f"cap={r['cap_hit']}" if r.get("cap_hit") else None,
                f"early={r['stopped_early']}" if r.get("stopped_early") else None,
            ) if b]
            out.append(f"  {r['task']}/{r['arm']}: " + "; ".join(bits))

    revs = {r["runner_sha256"] for r in rows if r.get("runner_sha256")}
    unstamped = [f"{r['task']}/{r['arm']}" for r in rows if not r.get("runner_sha256")]
    if len(revs) > 1:
        out.append("")
        out.append(f"WARNING: manifests span {len(revs)} runner revisions {sorted(revs)} — "
                   "cells are not strictly comparable.")
    if unstamped:
        # Failing open here is how the recorded WS27 A-arm cells (which predate
        # the field) would silently pass a check that exists because of them.
        out.append("")
        out.append(f"WARNING: {len(unstamped)} manifest(s) predate runner_sha256 "
                   f"({', '.join(unstamped)}) — revision comparability is unverifiable there.")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--prefix", default="", help="campaign prefix the runs were launched with")
    ap.add_argument("--trial", type=int, default=None, help="trial number (-tN suffix)")
    args = ap.parse_args()
    rows = collect(args.prefix, args.trial)
    print(json.dumps(rows, indent=2) if args.json else render(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
