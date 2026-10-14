"""Reconstruct a normalized `transcript.jsonl` for ARM A from its `memory.db`.

The baselines write a transcript as they run; AtomAgents does not. Without one,
three of the plan's mechanism metrics -- `tool_calls`, `tool_error_rate` and
`cwd_omission_rate` -- are computable for the controls but not for the system
under test, and a comparison table has to print `-` in the column that measures
most directly what prompt scaffolding buys.

It turns out nothing needs re-running: AtomAgents stores each LangChain message
verbatim in `messages.msg_json`, including populated `tool_calls` (with args)
and the tool-result messages. This exporter replays those rows into the exact
schema `benchmarks/arms/*.py` emit, so one reader covers all four arms.

    python -m benchmarks.export_transcript --all
    python -m benchmarks.export_transcript --session runs/jobs/session-demo1-gpt-t1

Writes `transcript.arm_a.jsonl` beside `memory.db` -- deliberately NOT
`transcript.jsonl`, so a reconstructed transcript can never be mistaken for one
recorded live.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JOBS = REPO / "runs" / "jobs"
OUT_NAME = "transcript.arm_a.jsonl"

# Mirrors react_arm's classifier so the `tool_errors` column means the same
# thing for every arm.
_ERROR_MARKERS = ('"status": "error"', 'returncode": -1', '"isError": true')


def _display_path(p: Path) -> str:
    """Repo-relative when possible; a session outside the repo (tests, ad-hoc
    exports) must not crash the exporter over a cosmetic string."""
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif b.get("type") == "thinking":
                    parts.append("[thinking]")
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def export_crewai(session_dir: Path, arm_log: Path) -> dict:
    """Reconstruct ARM C's tool calls from CrewAI's verbose log.

    ARM C's live transcript records task OUTPUTS, not individual tool calls --
    CrewAI's executor owns the tool loop and the arm never sees it. Its verbose
    log does print every call as `Using Tool: <name>` followed by a boxed
    `Tool Input`, so the mechanism metrics are recoverable.

    Parsed defensively: the box is rich-wrapped with padding, so reassembling
    the JSON exactly is unreliable. Presence of the `cwd` key inside the box is
    robust to that wrapping, and presence is all `cwd_omission` needs.

    Appends to the arm's existing transcript rather than replacing it, so the
    recorded task outputs survive.
    """
    session_dir = Path(session_dir).resolve()
    arm_log = Path(arm_log).resolve()
    if not arm_log.exists():
        return {"session": session_dir.name, "skipped": "no arm log"}

    # Only names on the pinned tool surface count: a tool OUTPUT that echoes
    # the phrase "Using Tool:" (an agent cat-ing this very log, a result
    # quoting the executor) must not mint a phantom call.
    try:
        known = set(json.loads((REPO / "benchmarks" / "tools.lock.json").read_text())["tools"])
    except Exception:  # noqa: BLE001
        known = set()

    text = arm_log.read_text(errors="replace")
    lines = text.splitlines()
    calls: list[dict] = []
    i = 0
    while i < len(lines):
        m = re.search(r"Using Tool:\s*([A-Za-z_][A-Za-z0-9_]*)", lines[i])
        if not m or (known and m.group(1) not in known):
            i += 1
            continue
        name = m.group(1)
        # Scan forward for this call's Tool Input box; stop at the next call so
        # a tool without a box cannot borrow the following one's arguments.
        box, j, seen_header = [], i + 1, False
        while j < len(lines) and j < i + 400:
            if "Using Tool:" in lines[j]:
                break
            if "Tool Input" in lines[j]:
                seen_header = True
                j += 1
                continue
            if seen_header:
                if lines[j].startswith("╰"):
                    break
                box.append(lines[j].strip("│ ").strip())
            j += 1
        blob = " ".join(b for b in box if b)
        calls.append({"name": name, "raw_input": blob[:4000], "has_cwd": '\\"cwd\\"' in blob or '"cwd"' in blob})
        i += 1

    out_path = session_dir / "transcript.crewai_tools.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "t": 0.0, "kind": "provenance", "source": arm_log.name,
            "note": "tool calls reconstructed from CrewAI verbose log after the run",
        }) + "\n")
        for c in calls:
            fh.write(json.dumps({
                "t": None, "kind": "assistant", "agent": "Crew",
                "text": "", "tool_calls": [{
                    "name": c["name"],
                    # Only the cwd key is asserted; the wrapped box cannot be
                    # reassembled into faithful JSON, so no other arg is claimed.
                    "args": {"cwd": "<present>"} if c["has_cwd"] else {},
                }],
            }) + "\n")

    omitted = sum(1 for c in calls if c["name"] in ("bash_run", "code_runner") and not c["has_cwd"])
    return {
        "session": session_dir.name,
        "tool_calls": len(calls),
        "cwd_omission": omitted,
        "written": _display_path(out_path),
        "by_tool": dict(sorted(((n, sum(1 for c in calls if c["name"] == n))
                                for n in {c["name"] for c in calls}), key=lambda x: -x[1])),
    }


def export(session_dir: Path) -> dict:
    db = session_dir / "memory.db"
    if not db.exists():
        return {"session": session_dir.name, "skipped": "no memory.db"}
    if (session_dir / "transcript.jsonl").exists():
        # A baseline arm: it recorded a real transcript as it ran, and its
        # memory.db mirror deliberately carries no tool_calls. Reconstructing
        # here would write a transcript that reads as 0 tool calls beside a
        # live one showing 104 — an artefact indistinguishable from a finding.
        return {"session": session_dir.name, "skipped": "live transcript.jsonl exists (baseline arm)"}

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "select msg_json, agent_name, created_at from messages order by id"
        ).fetchall()
    except sqlite3.Error as exc:
        con.close()
        return {"session": session_dir.name, "skipped": f"db error: {exc}"}
    con.close()

    out_path = session_dir / OUT_NAME
    stats = {
        "session": session_dir.name,
        "messages": len(rows),
        "assistant": 0,
        "tool_result": 0,
        "tool_calls": 0,
        "tool_errors": 0,
        "cwd_omission": 0,
        "agents": {},
    }

    with out_path.open("w", encoding="utf-8") as fh:

        def emit(rec: dict) -> None:
            fh.write(json.dumps(rec) + "\n")

        emit({"t": 0.0, "kind": "provenance", "source": "memory.db",
              "note": "reconstructed after the run; timings are DB timestamps, not wall offsets"})

        for msg_json, agent_name, created_at in rows:
            try:
                d = json.loads(msg_json)
            except Exception:  # noqa: BLE001
                continue
            mtype = d.get("type")
            data = d.get("data") or {}
            text = _text_of(data.get("content"))
            who = agent_name or data.get("name") or "?"
            stats["agents"][who] = stats["agents"].get(who, 0) + 1

            if mtype == "ai":
                calls = data.get("tool_calls") or []
                stats["assistant"] += 1
                stats["tool_calls"] += len(calls)
                for c in calls:
                    if c.get("name") in ("code_runner", "bash_run"):
                        if not (c.get("args") or {}).get("cwd"):
                            stats["cwd_omission"] += 1
                emit({
                    "t": created_at, "kind": "assistant", "agent": who, "text": text,
                    "tool_calls": [{"name": c.get("name"), "args": c.get("args")} for c in calls],
                    "usage": data.get("usage_metadata"),
                })
            elif mtype == "tool":
                is_err = any(m in text for m in _ERROR_MARKERS)
                stats["tool_result"] += 1
                stats["tool_errors"] += int(is_err)
                emit({
                    "t": created_at, "kind": "tool_result", "agent": who,
                    "name": data.get("name", "?"), "text": text[:20000], "error": is_err,
                })
            elif mtype == "human":
                emit({"t": created_at, "kind": "user", "agent": who, "text": text})

    stats["written"] = _display_path(out_path)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", help="one session dir")
    ap.add_argument("--all", action="store_true",
                    help="every session-* with a memory.db (narrow with --prefix)")
    ap.add_argument("--prefix", default="", help="campaign prefix the runs were launched with")
    args = ap.parse_args()

    if args.session:
        targets = [Path(args.session).resolve()]
    elif args.all:
        pattern = f"session-{args.prefix}-*" if args.prefix else "session-*"
        targets = sorted(d for d in JOBS.glob(pattern) if (d / "memory.db").exists())
    else:
        ap.error("give --session or --all")

    for d in targets:
        # ARM C first: its tool calls live only in the verbose log.
        if re.search(r"-crewai(_gpt)?(-t\d+)?$", d.name):
            log = REPO / "runs" / "logs" / f"{d.name.replace('session-', '')}.arm.log"
            c = export_crewai(d, log)
            if c.get("skipped"):
                print(f"skip {c['session']}: {c['skipped']}")
            else:
                print(f"{c['session']:<34} tool_calls={c['tool_calls']:<4} "
                      f"cwd_omitted={c['cwd_omission']:<3} -> {Path(c['written']).name}")
            continue
        s = export(d)
        if s.get("skipped"):
            print(f"skip {s['session']}: {s['skipped']}")
            continue
        print(f"{s['session']:<34} msgs={s['messages']:<5} tool_calls={s['tool_calls']:<4} "
              f"errors={s['tool_errors']:<3} cwd_omitted={s['cwd_omission']:<3} -> {Path(s['written']).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
