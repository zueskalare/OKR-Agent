"""Drive one benchmark run end to end, and record enough to prove what it did.

Design notes, each one paid for by a real failure:

* POLL, NEVER TRUST THE STREAM. The graph runs server-side. A client whose
  connection dies does not stop it — a WS27 pilot's SSE truncated at 16 KB
  while the run went on to completion. Terminal state is read from the session
  database and the filesystem, not from the HTTP response.

* RESTART THE BACKEND BETWEEN RUNS. `_APP_GRAPHS` caches compiled graphs by
  model_id+session_dir, the MCP client is a module-level singleton built at
  import, and ATOMAGENTS_MAX_SESSIONS=32 starts LRU-evicting checkpointer
  connections part-way through a campaign.

* WRITE THE MANIFEST BEFORE THE FIRST TURN. WS26's per-arm configuration had
  to be reverse-engineered from log signatures, and a broken embedding backend
  went unnoticed for an entire campaign. If it is not in the manifest, it did
  not happen.

Usage:
    python -m benchmarks.runner --task fem_demo0 --arm opus
    python -m benchmarks.runner --all --arms gpt --trial 1   # one OKR-Agent trial, 3 tasks
    python -m benchmarks.runner --all                        # 3 tasks x 6 published arms, serial

Prerequisites (docs/BENCHMARK.md): `.env` filled in from `.env.example`, and the
MCP tool service running (`make mcp`; `make mcp-check` must report the pinned
17-tool digest). The runner starts and restarts the backend itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BRIEF_DIR = Path(__file__).resolve().parent / "briefs"
LOCKFILE = BRIEF_DIR / "briefs.lock.json"
JOBS = REPO / "runs" / "jobs"
LOGS = REPO / "runs" / "logs"
BACKEND_URL = "http://localhost:8002"


def resolve_interpreters(environ: dict | None = None) -> tuple[str, str]:
    """(framework interpreter, CrewAI-environment interpreter).

    The framework environment is `./venv` by convention (a conda env, see
    docs/INSTALL.md); OKR_PYTHON overrides it, and a checkout without ./venv
    falls back to the interpreter running the runner. The CrewAI baseline has
    its own environment under benchmarks/crewai-env (OKR_CREWAI_PYTHON overrides).
    """
    e = os.environ if environ is None else environ
    venv_py = REPO / "venv" / "bin" / "python"
    py = e.get("OKR_PYTHON") or (str(venv_py) if venv_py.exists() else sys.executable)
    crewai_py = e.get("OKR_CREWAI_PYTHON") or str(
        REPO / "benchmarks" / "crewai-env" / ".venv" / "bin" / "python"
    )
    return py, crewai_py


PY, CREWAI_PY = resolve_interpreters()

# Model identifiers as recorded in the published run manifests. Defaults only:
# ATOMAGENTS_OPENAI_MODEL / ATOMAGENTS_ANTHROPIC_MODEL (OKR-Agent arms) and
# BENCH_MODEL (baseline arms) from .env take precedence, and the effective
# values are written to manifest["models"] so a run is never ambiguous about
# which model served it.
PAPER_MODELS = {"openai": "gpt-5.6-sol", "anthropic": "claude-opus-5"}
# MCP tools the service must publish (benchmarks/toolbind.py pins the full
# 17-tool surface: these 13 plus 4 bookkeeping tools).
EXPECTED_MCP_TOOLS = 13
# Stamped into every manifest. The study-era runner is identified by its
# `runner_sha256` instead (docs/BENCHMARK.md, "Provenance").
RUNNER_RELEASE = "v1.0.0"

# One kickoff plus at most three byte-identical nudges. WS26 used ad-hoc human
# steering whose text survives nowhere; a fixed, recorded continuation string is
# reproducible and is itself reported.
CONTINUE_TEXT = "Continue."
MAX_TURNS = 4
TURN_TIMEOUT_S = 5400          # 90 min; the longest WS26 turn was ~75 min
POLL_INTERVAL_S = 20

# Per-arm configuration. The asymmetry is evidenced, not arbitrary: the WS27
# pilot showed the shipped coordinator-only config produces zero artifacts on
# claude-opus-5 in 105 s (it escalates to the human for permissions), while
# full-tools executed the FEM brief in 6.5 min. gpt works in shipped config.
ARMS = {
    "gpt": {
        "provider": "openai",
        "model_suffix": "",
        "kind": "http",
        "env": {
            "ATOMAGENTS_LLM_PROVIDER": "openai",
            "ATOMAGENTS_SUPERVISOR_EXEC_TOOLS": "0",   # shipped coordinator-only
            "ATOMAGENTS_ENABLE_TRIAGE": "1",
            "ATOMAGENTS_REASONING_EFFORT": "high",
            "ATOMAGENTS_STAGE_GUARD": "1",
            "ATOMAGENTS_ENABLE_HITL": "0",
            # remote since WS29 (was `local` 384-dim MiniLM through WS27, which
            # made those gpt cells report-ineligible under the
            # exclude-local-embeddings rule).
            "ATOMAGENTS_EMBED_PROVIDER": "remote",
        },
    },
    "opus": {
        "provider": "anthropic",
        "model_suffix": "@anthropic",
        "kind": "http",
        "env": {
            "ATOMAGENTS_LLM_PROVIDER": "anthropic",
            "ATOMAGENTS_SUPERVISOR_EXEC_TOOLS": "1",   # see WS27 pilot
            "ATOMAGENTS_ENABLE_TRIAGE": "1",
            "ATOMAGENTS_REASONING_EFFORT": "high",
            "ATOMAGENTS_STAGE_GUARD": "1",
            "ATOMAGENTS_ENABLE_HITL": "0",
            # remote = text-embedding-3-large (3072-dim) via ATOMAGENTS_EMBED_*.
            # `remote` never silently falls back, so a dead endpoint aborts the
            # run instead of quietly downgrading it — which is the failure that
            # invalidated the whole WS26 campaign.
            # Since WS29 the gpt arm is also `remote`, so the AtomAgents arms
            # differ only in exec_tools; manifest["embeddings"] records the
            # RESOLVED backend each run actually used.
            "ATOMAGENTS_EMBED_PROVIDER": "remote",
        },
    },
    # Same model/route as `opus` but the SHIPPED coordinator-only config —
    # the configuration WS26 recorded freezing into a belief loop. Running it
    # at n=3 on the repaired briefs measures whether that failure mode is a
    # property of the config, the brief, or the day.
    "opus_std": {
        "provider": "anthropic",
        "model_suffix": "@anthropic",
        "kind": "http",
        # Not one of the six published configurations: `--all` skips it unless
        # it is named explicitly (--arms opus_std).
        "withheld": "ablation arm (coordinator-only Supervisor on the Anthropic route); "
                    "not part of the published 30-run dataset",
        "env": {
            "ATOMAGENTS_LLM_PROVIDER": "anthropic",
            "ATOMAGENTS_SUPERVISOR_EXEC_TOOLS": "0",   # shipped coordinator-only
            "ATOMAGENTS_ENABLE_TRIAGE": "1",
            "ATOMAGENTS_REASONING_EFFORT": "high",
            "ATOMAGENTS_STAGE_GUARD": "1",
            "ATOMAGENTS_ENABLE_HITL": "0",
            "ATOMAGENTS_EMBED_PROVIDER": "remote",
        },
    },
    # --- baselines -------------------------------------------------------
    # Both run claude-opus-5 at effort=high with the SAME 17 tools (see
    # benchmarks/toolbind.py + tools.lock.json) and the simple prompts in
    # benchmarks/prompts/. What differs from ARM A is the architecture and the
    # prompt scaffolding — which is the whole claim under test.
    #
    # These arms have no ATOMAGENTS_* configuration because they are not
    # AtomAgents: no supervisor, no triage, no stage-guard, no semantic memory.
    # `vectors == 0` in their manifests is therefore correct and expected, not
    # the WS26 silent-embedding bug.
    "react": {
        "provider": "anthropic",
        "model_suffix": "",
        "kind": "inproc",
        "entry": "benchmarks.arms.react_arm",
        "python": None,          # repo venv
        "env": {},
        # Backstop above the arm's own WALL_CLOCK_SEC=10800: the arm checks its
        # clock between stream chunks, so one hung MCP call rides past it.
        "timeout_s": 14400,
    },
    "crewai": {
        "provider": "anthropic",
        "model_suffix": "",
        "kind": "subproc",
        "entry": "benchmarks.arms.crewai_arm",
        # crewai 0.201.1 lives in its own venv (Python 3.12); the repo venv
        # cannot import it and it cannot import langchain_core.
        "python": CREWAI_PY,
        "env": {"CREWAI_TRACING_ENABLED": "false"},  # else it blocks on a TTY prompt
        # The crew's legitimate worst case is 3 sequential agents, each under
        # its own max_execution_time=10800 — a single shared cap would kill a
        # healthy crew mid-stage.
        "timeout_s": 3 * 10800 + 600,
    },
    # gpt-flavoured baselines: identical machinery and tool surface; the only
    # change is BENCH_PROVIDER routing the arm through the same gateway's
    # OpenAI surface with gpt-5.6-sol.
    "react_gpt": {
        "provider": "openai",
        "model_suffix": "",
        "kind": "inproc",
        "entry": "benchmarks.arms.react_arm",
        "python": None,
        "env": {"BENCH_PROVIDER": "openai", "BENCH_MODEL": "gpt-5.6-sol"},
        "timeout_s": 14400,
    },
    "crewai_gpt": {
        "provider": "openai",
        "model_suffix": "",
        "kind": "subproc",
        "entry": "benchmarks.arms.crewai_arm",
        "python": CREWAI_PY,
        "env": {
            "CREWAI_TRACING_ENABLED": "false",
            "BENCH_PROVIDER": "openai",
            "BENCH_MODEL": "gpt-5.6-sol",
        },
        "timeout_s": 3 * 10800 + 600,
    },
}


def default_arms() -> list[str]:
    """The arms `--all` runs when none are named: the six published configurations."""
    return sorted(a for a, c in ARMS.items() if not c.get("withheld"))


def run_id_for(prefix: str, task: str, arm: str, trial: int | None = None) -> str:
    """`[<prefix>-]<task>-<arm>[-t<n>]` — no leading dash when there is no prefix."""
    parts = [prefix, task, arm, f"t{trial}" if trial else ""]
    return "-".join(p for p in parts if p)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[runner {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------


def load_dotenv() -> dict:
    """Parse .env without letting it clobber PATH or anything else exported."""
    env = dict(os.environ)
    path = REPO / ".env"
    if not path.exists():
        return env
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key.isupper():
            env[key] = val
    return env


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:  # noqa: BLE001 - a missing git is not a run failure
            return ""

    return {
        "head": run("rev-parse", "HEAD"),
        "describe": run("describe", "--tags", "--always"),
        "dirty": bool(run("status", "--porcelain")),
    }


def prompt_hashes(flavor_dir: str) -> dict:
    """SHA-256 of every prompt file the run will actually load."""
    import hashlib

    out = {}
    d = REPO / flavor_dir
    for p in sorted(d.glob("*.yaml")):
        out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def _lammps_banner(text: str) -> str:
    """The version line of `lmp -h`, not merely the first line containing LAMMPS.

    The study-era probe kept the first line containing "LAMMPS", which in the
    usage text is the options header, so every published manifest records that
    header instead of the build date. The title line ("… Massively Parallel
    Simulator - 22 Jul 2025 - Update 3") is the one that carries the version;
    a log-style "LAMMPS (<date>)" banner is accepted as well.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if "Massively Parallel Simulator" in ln:
            return ln
    for ln in lines:
        if re.match(r"LAMMPS \(", ln):
            return ln
    return next((ln for ln in lines if "LAMMPS" in ln), "")


def toolchain_versions(env: dict) -> dict:
    """Capture the scientific stack. FEM acceptance is quoted to 3 s.f. and
    depends on the dolfinx build, so a git SHA alone is not reproducible."""
    probe = (
        "import json,importlib;out={}\n"
        "for m in ('dolfinx','gmsh','ufl','petsc4py','ovito','numpy'):\n"
        "    try:\n"
        "        out[m]=getattr(importlib.import_module(m),'__version__','?')\n"
        "    except Exception as e:\n"
        "        out[m]=f'unavailable: {type(e).__name__}'\n"
        "print(json.dumps(out))"
    )
    try:
        r = subprocess.run([PY, "-c", probe], capture_output=True, text=True, timeout=120, env=env, check=False)
        versions = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        versions = {"error": str(exc)}
    # Mirror the MCP service's own resolution order (tools.py): mpirun+lmp_mpi
    # first, then lmp, then lmp_serial. Probing only `lmp` reported "unavailable"
    # on a machine where LAMMPS works fine via lmp_mpi.
    versions["lammps"] = "unavailable"
    for binary in ("lmp_mpi", "lmp", "lmp_serial"):
        try:
            r = subprocess.run(
                [binary, "-h"], capture_output=True, text=True, timeout=60, env=env, check=False
            )
            banner = _lammps_banner(r.stdout or r.stderr)
            if banner:
                versions["lammps"] = f"{binary}: {banner}"
                break
        except (OSError, subprocess.SubprocessError):
            continue
    # The launcher matters as much as the build: a conda MPICH `mpirun` driving
    # an Open MPI `lmp_mpi` silently runs N independent serial copies.
    try:
        r = subprocess.run(
            ["mpirun", "--version"], capture_output=True, text=True, timeout=30, env=env, check=False
        )
        first = next((ln.strip() for ln in (r.stdout or r.stderr).splitlines() if ln.strip()), "")
        versions["mpirun"] = first or "unavailable"
    except (OSError, subprocess.SubprocessError):
        versions["mpirun"] = "unavailable"
    return versions


# --------------------------------------------------------------------------
# backend lifecycle
# --------------------------------------------------------------------------


def backend_pids() -> list[int]:
    r = subprocess.run(
        ["lsof", "-nP", "-iTCP:8002", "-sTCP:LISTEN", "-t"],
        capture_output=True, text=True, check=False,
    )
    return [int(x) for x in r.stdout.split()]


def stop_backend() -> None:
    for pid in backend_pids():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    for _ in range(30):
        if not backend_pids():
            return
        time.sleep(1)
    for pid in backend_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(2)


def start_backend(env: dict, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a")  # noqa: SIM115 - lives as long as the child
    child_env = dict(env)
    # The startup banner is a bare print(). Writing to a FILE makes stdout
    # block-buffered, so it never reached the log and the runner reported "no
    # startup banner found" on a run whose configuration was in fact correct.
    # Logging-based lines were unaffected, which is what made it confusing.
    child_env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [PY, "-m", "backend.openai_backend", "--port", "8002"],
        cwd=REPO, env=child_env, stdout=fh, stderr=subprocess.STDOUT,
    )
    for _ in range(90):
        if backend_pids():
            time.sleep(2)
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"backend exited early (rc={proc.returncode}); see {log_path}")
        time.sleep(1)
    raise RuntimeError(f"backend did not bind :8002 in 90s; see {log_path}")


def read_banner(log_path: Path, wait_s: int = 20) -> str:
    """The print() banner if present, else the llm_factory log line.

    The fallback matters: the banner is a print and the factory line goes
    through logging, so they have different flushing behaviour. Verifying the
    provider against whichever arrived is more robust than trusting one.
    """
    deadline = time.time() + wait_s
    fallback = ""
    while time.time() < deadline:
        for line in reversed(log_path.read_text(errors="replace").splitlines()):
            if "[OpenAI Backend] LLM config:" in line:
                return line.strip()
            if not fallback and "llm_factory: LLM: provider=" in line:
                fallback = line.split("llm_factory: ", 1)[-1].strip()
        if fallback:
            return fallback
        time.sleep(2)
    return fallback


# --------------------------------------------------------------------------
# turns
# --------------------------------------------------------------------------


def post_turn(payload: dict, timeout: int, capture: Path | None = None) -> tuple[bool, str]:
    """Fire one turn. Returns (completed_cleanly, note).

    A transport failure is NOT a run failure — the graph keeps going — so the
    caller falls back to polling the session for terminal state. When `capture`
    is given the raw SSE bytes are teed to that file while draining: the stream
    carries the usage frames, which are the only per-turn token record there is.
    """
    req = urllib.request.Request(
        f"{BACKEND_URL}/v1/responses",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if capture is not None:
                capture.parent.mkdir(parents=True, exist_ok=True)
                with open(capture, "wb") as out:
                    for chunk in resp:
                        out.write(chunk)
            else:
                for _ in resp:  # drain the stream; we only care that it ended
                    pass
        return True, "stream closed cleanly"
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        return False, f"transport ended early ({type(exc).__name__}: {exc})"


def session_stats(session_dir: Path) -> dict:
    db = session_dir / "memory.db"
    out = {
        "messages": 0, "vectors": 0, "artefacts": 0, "metrics": 0,
        "agents": {}, "files": 0, "deliverables": 0,
    }
    if db.exists():
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            out["messages"] = c.execute("select count(*) from messages").fetchone()[0]
            out["vectors"] = c.execute(
                "select count(*) from messages where length(embedding)>2"
            ).fetchone()[0]
            out["agents"] = dict(
                c.execute("select agent_name,count(*) from messages group by agent_name")
            )
            for tbl, key in (("artefacts", "artefacts"), ("validation_metrics", "metrics")):
                try:
                    out[key] = c.execute(f"select count(*) from {tbl}").fetchone()[0]
                except sqlite3.Error:
                    pass
            c.close()
        except sqlite3.Error as exc:
            out["db_error"] = str(exc)
    infra = {"memory.db", "graph_state.db", "README.md", "transcript.jsonl",
             "manifest.json", "arm_result.json",
             # Post-hoc reconstructions written by export_transcript, not work
             # product either.
             "transcript.arm_a.jsonl", "transcript.crewai_tools.jsonl"}
    # Directories that are installed environment, never work product. `.venv`
    # matters as much as `.home`: a baseline agent that runs `uv venv` inside
    # its session drops ~10k files and 144 MB there, which would score as the
    # most productive run of the campaign. No AtomAgents session created one,
    # so without this the arms' deliverable counts are not comparable at all.
    env_dirs = {
        ".home", "__pycache__", ".venv", "venv", "env", "site-packages",
        "node_modules", ".git", ".cache", ".rustup", ".cargo",
        ".pytest_cache", ".ipynb_checkpoints", ".mypy_cache",
        # ws29-fem_demo0-gpt-t1 bootstrapped a conda-style env inside its
        # session (.runtime/pkgs + .deps vendored wheels): ~67k installed files
        # scored as deliverables and defeated the early-stop delta. The t2
        # rerun invented yet another spelling (.docker_deps, ~4k files).
        ".runtime", ".deps", ".docker_deps", "pkgs", ".conda", ".micromamba", ".pixi",
    }
    out["deliverable_bytes"] = 0
    out["env_files"] = 0
    for p in session_dir.rglob("*"):
        if not p.is_file():
            continue
        out["files"] += 1
        # Deliverables = real work product: excludes the framework's own files,
        # sqlite sidecars, FEniCS JIT cache, __pycache__ and installed envs.
        rel = p.relative_to(session_dir)
        # Directory components only: a FILE named `env` is plausibly work
        # product (an env dump); a directory named `env` never is.
        if env_dirs & set(rel.parts[:-1]):
            out["env_files"] += 1
            continue
        # startswith, not equality: sqlite leaves -wal/-shm sidecars beside a
        # WAL-mode database. Matching only the exact name scored every
        # baseline's memory.db-shm + memory.db-wal as 2 deliverables / 32 KiB —
        # the phantom pair recorded in all six WS28 baseline manifests.
        if (
            p.name not in infra
            and not p.name.startswith(("memory.db", "graph_state.db"))
        ):
            out["deliverables"] += 1
            # Bytes as well as count, so a turn that only REWRITES existing
            # files still registers as progress and does not trip early-stop.
            try:
                out["deliverable_bytes"] += p.stat().st_size
            except OSError:
                pass
    return out


def wait_for_quiet(session_dir: Path, timeout: int) -> dict:
    """Poll until the session stops changing, or timeout. Returns final stats."""
    last, stable_for, waited = None, 0, 0
    while waited < timeout:
        time.sleep(POLL_INTERVAL_S)
        waited += POLL_INTERVAL_S
        stats = session_stats(session_dir)
        fingerprint = (stats["messages"], stats["files"], stats["artefacts"])
        if fingerprint == last:
            stable_for += POLL_INTERVAL_S
            if stable_for >= 60:
                return stats
        else:
            stable_for = 0
            last = fingerprint
    return session_stats(session_dir)


def archive_superseded(session_dir: Path, archive_root: Path | None = None) -> Path | None:
    """Move a cell's previous contents aside before relaunching into it.

    WS28 defect #7: `mkdir(exist_ok=True)` plus OKRStore appending to whatever
    `memory.db` is already there meant a relaunched cell RESUMED on top of its
    previous attempt — caught live when `session-ws27-demo2-react` inherited 12
    stale messages from a killed run. `--skip-existing` keys on manifest.json,
    so a KILLED cell (no manifest) was neither skipped nor cleaned.

    Never deletes: the superseded attempt moves to runs/archive so it stays
    inspectable evidence. A cell being deliberately resumed is not a case this
    runner has — every launch is a fresh run of the whole cell.
    """
    if not session_dir.exists() or not any(session_dir.iterdir()):
        return None
    root = archive_root or (REPO / "runs" / "archive" / "superseded-cells")
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    dest = root / f"{session_dir.name}-{stamp}"
    n = 0
    while dest.exists():
        n += 1
        dest = root / f"{session_dir.name}-{stamp}-{n}"
    shutil.move(str(session_dir), str(dest))
    log(f"archived previous contents of {session_dir.name} -> {dest}")
    return dest


# --------------------------------------------------------------------------
# baseline arms (ARM R in-process, ARM C in its own venv)
# --------------------------------------------------------------------------


def run_baseline(
    task: str, arm: str, cfg: dict, spec: dict, manifest: dict, session_dir: Path,
    env: dict, t_start: float,
) -> dict:
    """Execute a baseline arm and fill the SAME manifest schema as ARM A.

    The baselines are not HTTP services, so none of the backend restart /
    banner / post_turn machinery applies. Everything downstream of the run --
    session_stats, the manifest keys, --skip-existing -- is shared, which is
    what keeps the three arms' manifests comparable field by field.
    """
    run_id = manifest["run_id"]
    arm_log = LOGS / f"{run_id}.arm.log"
    arm_log.parent.mkdir(parents=True, exist_ok=True)
    result_path = session_dir / "arm_result.json"
    transcript = session_dir / "transcript.jsonl"

    manifest["toolchain"] = toolchain_versions(env)
    # Structural, not a failure: the baselines have no semantic memory at all.
    manifest["embeddings"] = "n/a (baseline arm has no semantic memory)"
    manifest["prompt_flavor"] = "benchmarks/prompts"
    manifest["prompt_hashes"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        for p in sorted((REPO / "benchmarks" / "prompts").glob("*.md"))
    }
    manifest["tools_lock"] = json.loads(
        (REPO / "benchmarks" / "tools.lock.json").read_text()
    )["digest"]

    child_env = dict(os.environ)
    child_env.update({k: v for k, v in env.items() if isinstance(v, str)})
    child_env.update(cfg.get("env") or {})
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONPATH"] = str(REPO)

    py = cfg.get("python") or PY
    argv = [
        py, "-m", cfg["entry"],
        "--task", task,
        "--session-dir", str(session_dir),
        "--transcript", str(transcript),
        "--result", str(result_path),
    ]
    log(f"launching {arm}: {' '.join(argv[:3])} … (log: {arm_log.name})")
    t0 = time.time()
    timed_out = False
    with open(arm_log, "w") as fh:
        try:
            proc = subprocess.run(
                argv, cwd=str(REPO), env=child_env, stdout=fh,
                stderr=subprocess.STDOUT,
                # Per-arm: a single shared cap would kill a healthy 3-stage
                # crew whose stages are each inside their own budget.
                timeout=cfg.get("timeout_s") or TURN_TIMEOUT_S * MAX_TURNS,
            )
        except subprocess.TimeoutExpired:
            # The manifest must survive the kill: hours of real work sit in the
            # session dir, and a cell with no manifest is invisible to
            # compare_arms and re-run from scratch by --skip-existing.
            timed_out = True
            proc = subprocess.CompletedProcess(argv, returncode=-9)
    elapsed = round(time.time() - t0, 1)

    arm_result: dict = {}
    if result_path.exists():
        try:
            arm_result = json.loads(result_path.read_text())
        except Exception:  # noqa: BLE001
            pass
    if not arm_result:
        # react_arm prints its result dict; recover it from the log.
        text = arm_log.read_text(errors="replace")
        for marker in ("RESULT_JSON ", ):
            if marker in text:
                try:
                    arm_result = json.loads(text.split(marker, 1)[1].splitlines()[0])
                except Exception:  # noqa: BLE001
                    pass
        if not arm_result and text.lstrip().startswith("{"):
            try:
                arm_result = json.loads(text[text.index("{"): text.rindex("}") + 1])
            except Exception:  # noqa: BLE001
                pass

    stats = session_stats(session_dir)
    manifest["arm_result"] = arm_result
    manifest["exit_code"] = proc.returncode
    # The requested-model alias ("atomagent"/"fenicsagent") is a backend
    # concept; a baseline manifest must name the model that actually served it.
    if arm_result.get("model"):
        manifest["model_id"] = arm_result["model"]
    # The runner's own turn budget applies to the HTTP arms only; recording it
    # here claimed a budget nobody passed to the arm.
    manifest["max_turns"] = arm_result.get("max_turns", "n/a (arm-managed)")
    manifest["turns"] = arm_result.get("turns") or [
        {"turn": 1, "sent": "brief", "seconds": elapsed, "transport_ok": proc.returncode == 0,
         "note": "single kickoff", "stats": stats}
    ]
    if arm_result.get("stopped_early"):
        manifest["stopped_early"] = arm_result["stopped_early"]
    if arm_result.get("cap_hit"):
        manifest["cap_hit"] = arm_result["cap_hit"]
    if timed_out:
        manifest["cap_hit"] = "runner_subprocess_timeout"
    manifest["banner"] = (
        f"arm={arm} model={arm_result.get('model','?')} "
        f"tools={arm_result.get('tools_bound','?')} "
        f"prompt={arm_result.get('prompt_file','?')} "
        f"effort=high"
    )
    manifest["resolved_model_lines"] = [
        f"provider={cfg['provider']} api={cfg['provider']} model={arm_result.get('model', '?')}"
    ]
    # Structural zero, recorded so it is never mistaken for a clean sheet:
    # only AtomAgents has a runtime fabrication guard.
    manifest["stage_guard_catches"] = None
    manifest["finished_at"] = _now()
    manifest["wall_clock_s"] = round(time.time() - t_start, 1)
    manifest["final"] = stats

    (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"  {elapsed}s exit={proc.returncode} tools={arm_result.get('tools_bound')} "
        f"tool_calls={arm_result.get('tool_calls')} "
        f"artefacts={stats['artefacts']} deliverables={stats['deliverables']}")
    if arm_result.get("error"):
        log(f"  ARM ERROR: {arm_result['error']}")
    log(f"done in {manifest['wall_clock_s']}s -> {session_dir / 'manifest.json'}")
    return manifest


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------

_TOOL_COUNT_RE = re.compile(r"Loaded (\d+) MCP tools")


def backend_tool_count(log_text: str) -> int | None:
    """How many MCP tools the backend bound for this session, from its log.

    None when the backend never said — which is itself a finding.
    """
    hits = _TOOL_COUNT_RE.findall(log_text)
    return int(hits[-1]) if hits else None


def preflight_mcp(url: str | None = None) -> dict:
    """Refuse to start a billed run unless the MCP service publishes exactly the
    pinned 17-tool surface.

    The HTTP arms never checked this: the backend logs "Failed to get MCP tools"
    and continues with the four bookkeeping tools only, so a run started with
    the tool service down completed, consumed a full LLM budget, and read as a
    (very lazy) agent until a human noticed the 0-deliverable manifest. The
    baseline arms already assert the lock themselves (react_arm, crewai_arm);
    this makes every arm do so before spending anything.
    """
    import tempfile

    from benchmarks import toolbind as tb
    from benchmarks.mcp_client import MCPClient
    from benchmarks.okr_tools import OKRStore

    url = (url or os.environ.get("ATOMAGENTS_MCP_URL") or "http://localhost:9999/mcp/").rstrip("/")
    with tempfile.TemporaryDirectory() as tmp:
        store = OKRStore(Path(tmp) / "probe.db", tmp)
        with MCPClient(url) as client:
            specs = tb.tool_specs(client, store)
            tb.assert_lock(specs)
    return {
        "url": url,
        "tools": len(specs),
        "mcp": sum(1 for s in specs if s.source == "mcp"),
        "digest": tb.lock_payload(specs)["digest"][:16],
    }


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------


def run_one(
    task: str, arm: str, *, prefix: str = "", trial: int | None = None,
    dry_run: bool = False,
) -> dict:
    lock = json.loads(LOCKFILE.read_text())
    if task not in lock["tasks"]:
        raise SystemExit(f"unknown task {task!r}; have {sorted(lock['tasks'])}")
    spec = lock["tasks"][task]
    cfg = ARMS[arm]

    brief = (BRIEF_DIR / spec["brief_file"]).read_text(encoding="utf-8")
    import hashlib

    if hashlib.sha256(brief.encode()).hexdigest() != spec["sha256"]:
        raise SystemExit(f"{task}: brief file does not match the lockfile — re-extract")

    run_id = run_id_for(prefix, task, arm, trial)
    session_name = f"session-{run_id}"
    session_dir = JOBS / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    backend_log = LOGS / f"{run_id}.backend.log"

    env = load_dotenv()
    env.update(cfg["env"])
    # Pin the model identifiers explicitly. The campaign took them from .env;
    # left unset they would fall back to the framework's generic defaults, and
    # the manifest must say which model actually served the run.
    env.setdefault("ATOMAGENTS_OPENAI_MODEL", PAPER_MODELS["openai"])
    env.setdefault("ATOMAGENTS_ANTHROPIC_MODEL", PAPER_MODELS["anthropic"])
    env.setdefault("BENCH_MODEL", PAPER_MODELS[cfg["provider"]])
    model_id = spec["model"] + cfg["model_suffix"]

    manifest = {
        "run_id": run_id,
        "task": task,
        "arm": arm,
        "campaign": prefix,
        "trial": trial,
        "started_at": _now(),
        "model_id": model_id,
        "provider": cfg["provider"],
        "env": cfg["env"],
        "brief": {k: spec[k] for k in ("sha256", "chars", "brief_file", "session_dir")},
        "prompt_flavor": spec["prompt_flavor"],
        "prompt_hashes": prompt_hashes(spec["prompt_flavor"]),
        "git": git_state(),
        # WS27 ran fem_demo0-opus on one runner revision and the other five on
        # another, which silently changed the early-stop predicate mid-campaign.
        # Hashing the runner itself makes that detectable instead of archaeological.
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
        "runner_release": RUNNER_RELEASE,
        # The identifiers that will serve this run (defaults: PAPER_MODELS).
        "models": {
            "openai": env.get("ATOMAGENTS_OPENAI_MODEL"),
            "anthropic": env.get("ATOMAGENTS_ANTHROPIC_MODEL"),
            "bench_model": env.get("BENCH_MODEL"),
        },
        "continuation_text": CONTINUE_TEXT,
        "max_turns": MAX_TURNS,
        "turns": [],
    }
    if manifest["git"]["dirty"]:
        log("WARNING: working tree is dirty — the run is not reproducible from the SHA alone")
    if (
        cfg.get("kind") == "http"
        and env.get("ATOMAGENTS_EMBED_PROVIDER") == "remote"
        and not env.get("ATOMAGENTS_EMBED_BASE_URL")
    ):
        log(
            "WARNING: ATOMAGENTS_EMBED_BASE_URL is unset — remote embeddings will inherit "
            "OPENAI_BASE_URL; a chat-only gateway has no /v1/embeddings and the run will "
            "abort after turn 1 with 0 vectors (see .env.example)"
        )

    if dry_run:
        print(json.dumps(manifest, indent=2))
        return manifest

    log(f"=== {run_id} :: {model_id} :: {spec['prompt_flavor']} ===")

    # Every launch starts from an empty cell; whatever a previous attempt left
    # (finished OR killed) is archived, never inherited. See archive_superseded.
    archive_superseded(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    # Tool-surface preflight for EVERY arm, before anything is billed.
    if os.environ.get("OKR_SKIP_MCP_PREFLIGHT") == "1":
        manifest["mcp_preflight"] = "skipped (OKR_SKIP_MCP_PREFLIGHT=1)"
    else:
        try:
            manifest["mcp_preflight"] = preflight_mcp(env.get("ATOMAGENTS_MCP_URL"))
        except Exception as exc:  # noqa: BLE001 - any failure here means "do not run"
            manifest["mcp_preflight"] = {"error": f"{type(exc).__name__}: {exc}"}
            manifest["finished_at"] = _now()
            (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            raise RuntimeError(
                f"{run_id}: MCP preflight failed — is the tool service running "
                f"(`make mcp`) and does it match tools.lock.json (`make mcp-check`)? {exc}"
            ) from exc
        log(
            f"mcp preflight: {manifest['mcp_preflight']['tools']} tools "
            f"(digest {manifest['mcp_preflight']['digest']})"
        )

    if cfg.get("kind") != "http":
        return run_baseline(task, arm, cfg, spec, manifest, session_dir, env, t_start=time.time())

    log("restarting backend")
    stop_backend()
    # Everything read back out of the backend log must come from THIS run only.
    # The log is opened in append mode and is named per (task, arm), so a
    # re-run concatenates onto the previous one; scanning from byte 0 then
    # reported the OLDEST run's values. That is not hypothetical — it made all
    # three opus manifests claim `local:…MiniLM` for runs that demonstrably
    # used 3072-dim remote embeddings.
    log_offset = backend_log.stat().st_size if backend_log.exists() else 0
    start_backend(env, backend_log)
    manifest["toolchain"] = toolchain_versions(env)

    banner = read_banner(backend_log)
    manifest["banner"] = banner
    # The RESOLVED embedding backend, not just the requested provider. A run
    # must be auditable after the fact for whether semantic recall was really
    # on and at what dimensionality.
    def this_run_log() -> str:
        with open(backend_log, "r", errors="replace") as fh:
            fh.seek(log_offset)
            return fh.read()

    # The Embeddings line is printed a beat after the banner (ws29-demo1-gpt-t1
    # missed it by one second and recorded "(not reported)" for a healthy run),
    # so poll briefly instead of reading once.
    manifest["embeddings"] = "(not reported)"
    _emb_deadline = time.time() + 30
    while time.time() < _emb_deadline:
        _emb = next(
            (
                ln.split("Embeddings:", 1)[-1].strip()
                for ln in this_run_log().splitlines()
                if "Embeddings:" in ln
            ),
            None,
        )
        if _emb:
            manifest["embeddings"] = _emb
            break
        time.sleep(2)
    log(f"embeddings: {manifest['embeddings']}")
    log(banner or "WARNING: no startup banner found")
    if cfg["provider"] not in banner:
        log(f"WARNING: banner does not name provider={cfg['provider']} — check .env")

    t0 = time.time()
    prev = session_stats(session_dir)
    for turn in range(1, MAX_TURNS + 1):
        text = brief if turn == 1 else CONTINUE_TEXT
        log(f"turn {turn}/{MAX_TURNS} ({'kickoff' if turn == 1 else 'continue'})")
        ts = time.time()
        ok, note = post_turn(
            {"model": model_id, "session_dir": session_name, "input": text, "stream": True},
            TURN_TIMEOUT_S,
            capture=LOGS / f"{run_id}.turn{turn}.sse",
        )
        if not ok:
            log(f"{note}; falling back to polling (the graph keeps running server-side)")
            wait_for_quiet(session_dir, TURN_TIMEOUT_S)
        stats = session_stats(session_dir)
        manifest["turns"].append({
            "turn": turn,
            "sent": "brief" if turn == 1 else CONTINUE_TEXT,
            "seconds": round(time.time() - ts, 1),
            "transport_ok": ok,
            "note": note,
            "stats": stats,
        })
        log(f"  {round(time.time()-ts)}s  msgs={stats['messages']} vec={stats['vectors']} "
            f"artefacts={stats['artefacts']} deliverables={stats['deliverables']} "
            f"agents={stats['agents']}")
        if turn == 1:
            # The backend binds its tools lazily on the first request and says
            # how many it got. Anything but the pinned count means the agents
            # are running without the tool surface the benchmark is about.
            n_tools = backend_tool_count(this_run_log())
            manifest["mcp_tools_loaded"] = n_tools
            if n_tools != EXPECTED_MCP_TOOLS:
                manifest["mcp_failure"] = (
                    f"backend bound {n_tools} MCP tools, expected {EXPECTED_MCP_TOOLS}"
                )
                manifest["finished_at"] = _now()
                manifest["wall_clock_s"] = round(time.time() - t0, 1)
                manifest["final"] = stats
                (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                raise RuntimeError(f"{run_id}: {manifest['mcp_failure']} — aborting the run")
        if stats["vectors"] == 0 and stats["messages"] > 0:
            # WS27 defect #8: nothing gated on the probe, so a recall-off run
            # completed and read as success. Abort instead — after writing the
            # manifest, so the failed cell stays visible to compare_arms and is
            # not silently re-run from scratch by --skip-existing.
            manifest["embedding_failure"] = "0 vectors after a turn with messages"
            manifest["finished_at"] = _now()
            manifest["wall_clock_s"] = round(time.time() - t0, 1)
            manifest["final"] = stats
            (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            raise RuntimeError(
                f"{run_id}: semantic recall is OFF (0 embedding vectors) — aborting the run"
            )

        # Early stop. A turn that adds no messages AND no deliverables means the
        # agent has nothing left to do (or is wedged); further identical nudges
        # only burn tokens. Without this the loop always spends MAX_TURNS, which
        # on opus is hours per run.
        gained_work = stats["deliverables"] - prev["deliverables"]
        gained_bytes = stats.get("deliverable_bytes", 0) - prev.get("deliverable_bytes", 0)
        if turn > 1 and gained_work == 0 and gained_bytes == 0:
            # A continuation exists to make more progress. If it produced no new
            # and no changed work product, further identical nudges will not
            # either — the first version keyed on message count instead, and
            # three dead turns still "gained" a reply each and ran anyway.
            manifest["stopped_early"] = (
                f"turn {turn} produced no new or modified deliverables"
            )
            log(f"  early stop: {manifest['stopped_early']}")
            break
        prev = stats

    # The provider/model lines are emitted lazily when the graph is first built,
    # i.e. during turn 1 — after the startup read. Refresh so the manifest always
    # carries proof of which model actually served the run.
    if not manifest.get("banner"):
        manifest["banner"] = read_banner(backend_log, wait_s=0)
    # Same root cause as the embeddings field: read only this run's slice.
    manifest["resolved_model_lines"] = sorted({
        ln.split("llm_factory: ", 1)[-1].strip()
        for ln in this_run_log().splitlines()
        if "llm_factory: LLM: provider=" in ln
    })
    manifest["finished_at"] = _now()
    manifest["wall_clock_s"] = round(time.time() - t0, 1)
    manifest["final"] = session_stats(session_dir)
    manifest["stage_guard_catches"] = this_run_log().count("[stage-guard]")
    # Provider throttling killed two WS27 cells; count it so the campaign
    # driver can halt for diagnosis instead of burning the remaining quota.
    # Match real HTTP-429 signatures only: the first version matched any line
    # containing "429" and "error", which flagged 53 lines of agent PROSE in a
    # fully clean run and halted the ws29 t1 campaign for nothing.
    _throttle_signs = (
        "Error code: 429", "status_code=429", "HTTP/1.1 429",
        "Too Many Requests", "RateLimitError",
    )
    manifest["provider_error_429_lines"] = sum(
        1
        for ln in this_run_log().splitlines()
        if any(sig in ln for sig in _throttle_signs)
    )
    # The credential pool can also die outright (503 auth_unavailable killed
    # fem_demo0-t2's first attempt in 39 s). Counted separately: a couple of
    # absorbed blips are survivable, a storm means the campaign must stop.
    manifest["provider_auth_503_lines"] = sum(
        1 for ln in this_run_log().splitlines() if "auth_unavailable" in ln
    )

    (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"done in {manifest['wall_clock_s']}s -> {session_dir / 'manifest.json'}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["demo1", "demo2", "fem_demo0"])
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--all", action="store_true", help="all 3 tasks x all arms, serial")
    ap.add_argument(
        "--arms",
        default="",
        help="comma-separated arm subset for --all, e.g. 'react,crewai' (default: the six "
             "published configurations; withheld arms such as opus_std must be named)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip cells that already have a manifest.json (resume a campaign)",
    )
    ap.add_argument(
        "--prefix", default="",
        help="optional campaign prefix for run ids / session dirs "
             "(session-<prefix>-<task>-<arm>[-t<n>]; default: none, matching the dataset layout)",
    )
    ap.add_argument(
        "--trial", type=int, default=None,
        help="trial number for repeated-trial campaigns; suffixes the run id with -tN",
    )
    ap.add_argument(
        "--halt-on-failure", action="store_true",
        help="stop the campaign on the first failed run (or provider 429s) instead of continuing",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the manifest, launch nothing")
    args = ap.parse_args()

    def rid_for(task: str, arm: str) -> str:
        return run_id_for(args.prefix, task, arm, args.trial)

    if args.all:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()] or default_arms()
        unknown = [a for a in arms if a not in ARMS]
        if unknown:
            ap.error(f"unknown arm(s) {unknown}; have {sorted(ARMS)}")
        jobs = [(t, a) for a in arms for t in ("demo1", "demo2", "fem_demo0")]
    elif args.task and args.arm:
        jobs = [(args.task, args.arm)]
    else:
        ap.error("give --task and --arm, or --all")

    if args.skip_existing:
        keep = []
        for task, arm in jobs:
            done = JOBS / f"session-{rid_for(task, arm)}" / "manifest.json"
            if done.exists():
                log(f"skip {task}/{arm}: already has {done.name}")
            else:
                keep.append((task, arm))
        jobs = keep

    log(f"campaign: {len(jobs)} run(s) -> " + ", ".join(f"{t}/{a}" for t, a in jobs))

    results = []
    for task, arm in jobs:
        try:
            result = run_one(
                task, arm, prefix=args.prefix, trial=args.trial, dry_run=args.dry_run
            )
        except Exception as exc:  # noqa: BLE001 - one bad run must not sink the campaign
            log(f"RUN FAILED {task}/{arm}: {type(exc).__name__}: {exc}")
            result = {"run_id": rid_for(task, arm), "error": str(exc)}
        results.append(result)
        if args.halt_on_failure and (
            "error" in result
            or result.get("provider_error_429_lines")
            # >=10 distinguishes an auth STORM (the 39 s void cell logged 12+)
            # from a couple of blips a healthy run absorbed via retries.
            or (result.get("provider_auth_503_lines") or 0) >= 10
        ):
            log(
                "HALT: --halt-on-failure and the last run failed or hit provider "
                "throttling — stopping the campaign for diagnosis"
            )
            break

    if not args.dry_run:
        print("\n=== summary ===")
        for r in results:
            if "error" in r:
                print(f"{r['run_id']:<28} ERROR {r['error']}")
                continue
            f = r["final"]
            print(f"{r['run_id']:<28} {r['wall_clock_s']:>7.0f}s  msgs={f['messages']:<4} "
                  f"vec={f['vectors']:<4} artefacts={f['artefacts']:<3} "
                  f"deliverables={f['deliverables']:<5} agents={f['agents']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
