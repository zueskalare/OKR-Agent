"""Extract the three benchmark briefs from README.md into hashed files.

The WS26 campaign was reproducible only by hand-copying curl blocks out of the
README, and the exact text of what each arm received survives nowhere except
inside each session's memory.db. That is not a benchmark, it is an anecdote.

This script makes README.md the single source of truth and pins the result by
SHA-256, so a rerun can prove it sent the same bytes as the run before it.

    python -m benchmarks.extract_briefs           # write briefs/ + lockfile
    python -m benchmarks.extract_briefs --check   # verify, exit 1 on drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
BRIEF_DIR = Path(__file__).resolve().parent / "briefs"
LOCKFILE = BRIEF_DIR / "briefs.lock.json"

# Section heading -> task key. The heading is the anchor because it is stable
# prose; line numbers are not.
TASKS = [
    ("### MD demo 1 — twisted CNT fiber bundle under tension", "demo1"),
    ("### MD demo 2 — polycrystalline Fe under compression", "demo2"),
    ("### FEM demo — plate with a circular hole (FEniCSx)", "fem_demo0"),
]

# Which prompt family each task runs under. fenicsagent swaps ALL THREE agents
# to AtomAgents/templates/fenicsx/ (backend/openai_backend.py, `if pipeline ==
# "fenicsagent"`), which is a different and much larger prompt set than
# templates/prompts/. Recording it here keeps that from being an invisible
# variable.
PROMPT_FLAVOR = {
    "atomagent": "AtomAgents/templates/prompts",
    "fenicsagent": "AtomAgents/templates/fenicsx",
}

_PAYLOAD_RE = re.compile(r"curl -sS -N http://localhost:8002/v1/responses.*?-d '(\{.*?\})'\n", re.DOTALL)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract() -> dict:
    readme = README.read_text(encoding="utf-8")
    out: dict[str, dict] = {}

    for heading, key in TASKS:
        try:
            start = readme.index(heading)
        except ValueError as exc:
            raise SystemExit(f"README heading not found for {key}: {heading!r}") from exc
        # Stop at the next section so a missing block cannot silently borrow the
        # following task's payload.
        nxt = readme.find("\n### ", start + len(heading))
        segment = readme[start : nxt if nxt != -1 else len(readme)]

        m = _PAYLOAD_RE.search(segment)
        if not m:
            raise SystemExit(f"no /v1/responses payload found in the {key} section")
        body = json.loads(m.group(1))

        brief = body["input"]
        model = body["model"]
        if model not in PROMPT_FLAVOR:
            raise SystemExit(f"{key}: unknown model id {model!r}")

        out[key] = {
            "task": key,
            "model": model,
            "prompt_flavor": PROMPT_FLAVOR[model],
            "session_dir": body["session_dir"],
            "brief_file": f"{key}.txt",
            "chars": len(brief),
            "sha256": sha256(brief),
            "_text": brief,
        }
    return out


def write(tasks: dict) -> None:
    BRIEF_DIR.mkdir(parents=True, exist_ok=True)
    lock = {
        "source": "README.md",
        "note": (
            "Regenerate with `python -m benchmarks.extract_briefs`; verify with "
            "`--check`. The runner refuses to launch on a hash mismatch."
        ),
        "tasks": {},
    }
    for key, spec in sorted(tasks.items()):
        (BRIEF_DIR / spec["brief_file"]).write_text(spec["_text"], encoding="utf-8")
        lock["tasks"][key] = {k: v for k, v in spec.items() if not k.startswith("_")}
    LOCKFILE.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


def check(tasks: dict) -> int:
    if not LOCKFILE.exists():
        print("lockfile missing — run without --check first", file=sys.stderr)
        return 1
    lock = json.loads(LOCKFILE.read_text())
    bad = 0
    for key, spec in sorted(tasks.items()):
        pinned = lock["tasks"].get(key)
        if pinned is None:
            print(f"FAIL {key}: absent from lockfile")
            bad += 1
            continue
        on_disk = (BRIEF_DIR / spec["brief_file"]).read_text(encoding="utf-8")
        problems = []
        if spec["sha256"] != pinned["sha256"]:
            problems.append(f"README drifted from lockfile ({pinned['sha256'][:12]} -> {spec['sha256'][:12]})")
        if sha256(on_disk) != pinned["sha256"]:
            problems.append("brief file on disk does not match the lockfile")
        if spec["model"] != pinned["model"]:
            problems.append(f"model {pinned['model']} -> {spec['model']}")
        if problems:
            bad += 1
            print(f"FAIL {key}: " + "; ".join(problems))
        else:
            print(f"ok   {key}: {spec['chars']:>5} chars  {spec['sha256'][:16]}  {spec['model']}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify, do not write")
    args = ap.parse_args()

    tasks = extract()
    if args.check:
        return check(tasks)
    write(tasks)
    for key, spec in sorted(tasks.items()):
        print(f"wrote {key}: {spec['chars']:>5} chars  {spec['sha256'][:16]}  "
              f"{spec['model']} -> {spec['prompt_flavor']}")
    print(f"lockfile: {LOCKFILE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
