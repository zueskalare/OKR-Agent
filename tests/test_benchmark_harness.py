"""Contracts for the WS27 rerun harness.

These pin the things that, when they silently drifted, cost whole campaigns:
the brief text, the per-arm configuration, the prompt family a task runs under,
and the telemetry that proves semantic recall was on.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from benchmarks import extract_briefs as eb
from benchmarks import runner as rn

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# briefs
# --------------------------------------------------------------------------


def test_briefs_on_disk_match_the_readme():
    """README.md is the single source of truth. If someone edits a benchmark
    brief there, this fails rather than the campaign silently comparing runs
    that received different bytes."""
    assert eb.check(eb.extract()) == 0


def test_every_task_is_locked_with_a_hash():
    lock = json.loads(eb.LOCKFILE.read_text())
    assert set(lock["tasks"]) == {"demo1", "demo2", "fem_demo0"}
    for key, spec in lock["tasks"].items():
        assert len(spec["sha256"]) == 64, key
        assert spec["chars"] > 500, f"{key} brief suspiciously short"
        body = (eb.BRIEF_DIR / spec["brief_file"]).read_text()
        assert eb.sha256(body) == spec["sha256"], key


def test_fem_task_uses_the_fenicsx_prompt_family():
    """The trap: fenicsagent swaps ALL THREE agents to templates/fenicsx (a
    different, much larger prompt set). A harness that assumed templates/prompts
    would compare two different systems and call it one."""
    lock = json.loads(eb.LOCKFILE.read_text())
    assert lock["tasks"]["fem_demo0"]["model"] == "fenicsagent"
    assert lock["tasks"]["fem_demo0"]["prompt_flavor"].endswith("templates/fenicsx")
    for md in ("demo1", "demo2"):
        assert lock["tasks"][md]["model"] == "atomagent"
        assert lock["tasks"][md]["prompt_flavor"].endswith("templates/prompts")


def test_prompt_flavor_directories_exist_and_are_hashable():
    lock = json.loads(eb.LOCKFILE.read_text())
    for spec in lock["tasks"].values():
        hashes = rn.prompt_hashes(spec["prompt_flavor"])
        assert "supervisor.yaml" in hashes, spec["prompt_flavor"]
        assert all(len(h) == 16 for h in hashes.values())


# --------------------------------------------------------------------------
# arm configuration
# --------------------------------------------------------------------------


def test_arm_asymmetry_is_the_evidenced_one():
    """gpt runs the shipped coordinator-only Supervisor; opus runs full-tools.
    This is not a preference — the WS27 pilot showed coordinator-only produces
    zero artifacts on opus. If someone 'tidies' this into symmetry, the opus
    arm silently stops working."""
    assert rn.ARMS["gpt"]["env"]["ATOMAGENTS_SUPERVISOR_EXEC_TOOLS"] == "0"
    assert rn.ARMS["opus"]["env"]["ATOMAGENTS_SUPERVISOR_EXEC_TOOLS"] == "1"


ATOMAGENT_ARMS = sorted(a for a, c in rn.ARMS.items() if c.get("kind") == "http")
BASELINE_ARMS = sorted(a for a, c in rn.ARMS.items() if c.get("kind") != "http")


@pytest.mark.parametrize("arm", ATOMAGENT_ARMS)
def test_every_arm_pins_provider_effort_and_embeddings(arm):
    env = rn.ARMS[arm]["env"]
    # ATOMAGENTS_LLM_PROVIDER is commented out in .env, so an unset provider
    # silently falls back to gpt-5.6-sol — an opus run would quietly be a gpt run.
    assert env["ATOMAGENTS_LLM_PROVIDER"] == rn.ARMS[arm]["provider"]
    assert env["ATOMAGENTS_REASONING_EFFORT"] == "high"
    assert env["ATOMAGENTS_STAGE_GUARD"] == "1"
    assert env["ATOMAGENTS_ENABLE_TRIAGE"] == "1"
    # the whole reason for the rerun: recall must be explicitly configured,
    # never left to inherit OPENAI_BASE_URL. `remote` additionally refuses to
    # fall back silently, so a dead endpoint aborts instead of downgrading.
    assert env["ATOMAGENTS_EMBED_PROVIDER"] in ("local", "remote")


def test_model_id_selects_the_provider_route():
    lock = json.loads(eb.LOCKFILE.read_text())
    fem = lock["tasks"]["fem_demo0"]["model"]
    md = lock["tasks"]["demo1"]["model"]
    assert fem + rn.ARMS["opus"]["model_suffix"] == "fenicsagent@anthropic"
    assert md + rn.ARMS["opus"]["model_suffix"] == "atomagent@anthropic"
    assert md + rn.ARMS["gpt"]["model_suffix"] == "atomagent"


def test_continuation_is_fixed_and_recorded():
    """WS26 used ad-hoc human steering whose text survives nowhere. A rerun
    must send a constant, and the manifest must carry it."""
    assert rn.CONTINUE_TEXT.strip()
    assert rn.MAX_TURNS >= 1


# --------------------------------------------------------------------------
# environment handling
# --------------------------------------------------------------------------


def test_load_dotenv_does_not_clobber_path(monkeypatch, tmp_path):
    """Sourcing .env in a shell wiped PATH mid-session (curl vanished). The
    runner parses it instead of sourcing it, and must not import junk."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        "PATH=/nonsense\n"
        "OPENAI_API_KEY=sk-x\n"
        'QUOTED="value"\n'
        "lowercase=ignored\n"
        "malformed-line-without-equals\n"
    )
    monkeypatch.setattr(rn, "REPO", tmp_path)
    monkeypatch.setenv("PATH", "/real/path")
    env = rn.load_dotenv()
    assert env["OPENAI_API_KEY"] == "sk-x"
    assert env["QUOTED"] == "value"
    assert "lowercase" not in env
    # PATH is uppercase so it *is* imported — assert it at least parsed, and
    # that the runner never sources the file into its own process.
    import os

    assert os.environ["PATH"] == "/real/path"


# --------------------------------------------------------------------------
# telemetry
# --------------------------------------------------------------------------


def _make_session(tmp_path, *, vectors: int, messages: int):
    d = tmp_path / "session-x"
    d.mkdir()
    con = sqlite3.connect(d / "memory.db")
    con.execute("create table messages (id integer primary key, agent_name text, embedding text)")
    con.execute("create table artefacts (id integer primary key)")
    con.execute("create table validation_metrics (id integer primary key)")
    for i in range(messages):
        con.execute(
            "insert into messages (agent_name, embedding) values (?,?)",
            ("Supervisor", json.dumps([0.1, 0.2]) if i < vectors else ""),
        )
    con.commit()
    con.close()
    return d


def test_session_stats_counts_vectors(tmp_path):
    """vec==0 is the fingerprint of the bug that invalidated WS26; the runner
    must surface it per turn."""
    d = _make_session(tmp_path, vectors=3, messages=5)
    stats = rn.session_stats(d)
    assert stats["messages"] == 5
    assert stats["vectors"] == 3
    assert stats["agents"] == {"Supervisor": 5}


def test_deliverables_exclude_framework_and_cache_files(tmp_path):
    """A deliverable count that includes graph_state.db, FEniCS JIT cache and
    __pycache__ would score an empty run as productive — the same class of
    mistake as the .deps/ false-COMPLETE in the WS26 driver."""
    d = _make_session(tmp_path, vectors=1, messages=1)
    (d / "README.md").write_text("board")
    (d / "graph_state.db").write_text("x")
    (d / "graph_state.db-wal").write_text("x")
    (d / ".home" / "cache").mkdir(parents=True)
    (d / ".home" / "cache" / "libffcx.so").write_text("x")
    (d / "src").mkdir()
    (d / "src" / "__pycache__").mkdir()
    (d / "src" / "__pycache__" / "m.pyc").write_text("x")
    (d / "src" / "solve.py").write_text("real work")
    (d / "stress.dat").write_text("real work")

    stats = rn.session_stats(d)
    assert stats["deliverables"] == 2, "only solve.py and stress.dat are work product"
    assert stats["files"] > stats["deliverables"]


def test_installed_environments_are_not_deliverables(tmp_path):
    """A baseline agent that runs `uv venv` inside its session drops thousands
    of files there. Counting them would make it the most 'productive' run of
    the campaign, and no AtomAgents session creates one — so leaving .venv in
    would compare arms on an artefact of their install habits."""
    d = _make_session(tmp_path, vectors=1, messages=1)
    (d / "build.py").write_text("real work")
    for junk in (
        d / ".venv" / "bin" / "python",
        d / "sub" / ".venv" / "lib" / "site-packages" / "numpy" / "__init__.py",
        d / "node_modules" / "pkg" / "index.js",
        d / ".cache" / "wheels" / "x.whl",
        d / "src" / "__pycache__" / "m.pyc",
    ):
        junk.parent.mkdir(parents=True, exist_ok=True)
        junk.write_text("x")

    # A FILE named `env` is plausibly work product (an env dump); only
    # DIRECTORY components mark installed environment.
    (d / "env").write_text("PATH=...")

    stats = rn.session_stats(d)
    assert stats["deliverables"] == 2, "build.py and the env dump are work product"
    assert stats["env_files"] == 5
    assert stats["files"] > stats["deliverables"]


def test_harness_own_files_are_not_deliverables(tmp_path):
    """transcript.jsonl / manifest.json / arm_result.json are written BY the
    harness (and the reconstructed transcripts by export_transcript). Counting
    them would give every baseline free deliverables and put a zero-work run
    on the board as productive."""
    d = _make_session(tmp_path, vectors=1, messages=1)
    for name in ("transcript.jsonl", "manifest.json", "arm_result.json",
                 "transcript.arm_a.jsonl", "transcript.crewai_tools.jsonl"):
        (d / name).write_text("{}")
    assert rn.session_stats(d)["deliverables"] == 0


def test_sqlite_sidecars_are_never_deliverables(tmp_path):
    """WAL mode leaves memory.db-shm/-wal beside the database. The exact-name
    infra filter missed them, so every recorded WS28 baseline manifest carried
    a phantom '2 deliverables / 32 KiB' — database plumbing scored as work
    product."""
    d = _make_session(tmp_path, vectors=0, messages=1)
    (d / "memory.db-shm").write_bytes(b"\0" * 32768)
    (d / "memory.db-wal").write_bytes(b"")
    (d / "graph_state.db-wal").write_bytes(b"")
    stats = rn.session_stats(d)
    assert stats["deliverables"] == 0
    assert stats["deliverable_bytes"] == 0


def test_okr_store_closes_its_connections(tmp_path):
    """The sidecars in the previous test existed because OKRStore never closed
    a connection: sqlite removes -shm/-wal on the last clean close, so a store
    that closes leaves a bare memory.db behind."""
    from benchmarks.okr_tools import OKRStore

    store = OKRStore(tmp_path / "memory.db", tmp_path)
    f = tmp_path / "d.dat"
    f.write_text("x")
    store.register_artefact(str(f))
    store.record_metric("E", 1.0)
    assert not (tmp_path / "memory.db-wal").exists()
    assert not (tmp_path / "memory.db-shm").exists()


def test_session_stats_survives_a_missing_database(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    stats = rn.session_stats(d)
    assert stats["messages"] == 0 and stats["vectors"] == 0


def test_deliverable_bytes_tracked_so_rewrites_count_as_progress(tmp_path):
    """Early-stop keys on deliverables AND bytes. A turn that only rewrites an
    existing file must still read as progress, or the runner would abandon a
    run that is actively iterating on its scripts."""
    d = _make_session(tmp_path, vectors=1, messages=1)
    f = d / "solve.py"
    f.write_text("v1")
    before = rn.session_stats(d)
    f.write_text("v2 is longer")
    after = rn.session_stats(d)
    assert before["deliverables"] == after["deliverables"] == 1
    assert after["deliverable_bytes"] > before["deliverable_bytes"]


def test_lammps_probe_matches_the_mcp_resolution_order():
    """The MCP service resolves mpirun+lmp_mpi before lmp; a probe that only
    tried `lmp` reported LAMMPS unavailable on a machine where it works."""
    import inspect

    src = inspect.getsource(rn.toolchain_versions)
    order = [b for b in ("lmp_mpi", "lmp_serial") if b in src]
    assert order == ["lmp_mpi", "lmp_serial"], "probe must try the mpi build too"


# --------------------------------------------------------------------------
# baseline arms
# --------------------------------------------------------------------------


def test_both_baselines_exist_and_are_not_atomagents():
    """The comparison needs exactly two controls, and neither may carry
    ATOMAGENTS_* configuration — if one did, it would be AtomAgents wearing a
    different name and the contrast would be meaningless."""
    assert BASELINE_ARMS == ["crewai", "crewai_gpt", "react", "react_gpt"]
    for arm in BASELINE_ARMS:
        cfg = rn.ARMS[arm]
        # Two model flavours per architecture; the *_gpt arms route through the
        # gateway's OpenAI surface via BENCH_PROVIDER, everything else equal.
        expected_provider = "openai" if arm.endswith("_gpt") else "anthropic"
        assert cfg["provider"] == expected_provider, arm
        if arm.endswith("_gpt"):
            assert cfg["env"].get("BENCH_PROVIDER") == "openai", arm
        assert not any(k.startswith("ATOMAGENTS_") for k in cfg["env"]), arm
        assert cfg["entry"].startswith("benchmarks.arms."), arm


def test_crewai_runs_in_its_own_interpreter():
    """crewai 0.201.1 cannot coexist with the repo venv (it has no
    langchain_core; the repo venv has no crewai). A baseline silently launched
    with the wrong interpreter would fail at import and score zero."""
    assert rn.ARMS["react"]["python"] is None, "ARM R runs in the repo venv"
    py = Path(rn.ARMS["crewai"]["python"])
    if not py.exists():
        # The CrewAI venv is machine-local by design (gitignored, absolute
        # path). Its absence means ARM C cannot RUN here — it does not make
        # the committed configuration wrong, so skip rather than fail.
        pytest.skip(f"CrewAI venv not provisioned on this machine: {py}")


def test_crewai_tracing_is_disabled():
    """CrewAI prompts 'view your execution traces? [y/N]' on a 20 s TTY timeout.
    In an unattended campaign that is 20 wasted seconds per run at best, and a
    blocked run at worst."""
    assert rn.ARMS["crewai"]["env"]["CREWAI_TRACING_ENABLED"] == "false"


def test_baseline_prompts_are_present_and_fully_substituted():
    """{PROJECT_ROOT} is the only placeholder, and it must be the ONLY one —
    a survivor would ship the literal token to the model as its working dir."""
    import re

    for name in ("baseline_md.md", "baseline_fem.md"):
        text = (REPO / "benchmarks" / "prompts" / name).read_text()
        assert len(text.splitlines()) > 60, name
        placeholders = set(re.findall(r"\{[A-Za-z_]+\}", text))
        assert placeholders == {"{PROJECT_ROOT}"}, (name, placeholders)
        # the honesty instruction is the one thing both arms share with ARM A
        assert "must come from a real tool result" in text, name


def test_tools_lock_pins_seventeen_tools():
    """13 MCP + 4 bookkeeping. If the service gains or loses a tool between two
    arms' runs, they are no longer comparable and the runner must refuse."""
    lock = json.loads((REPO / "benchmarks" / "tools.lock.json").read_text())
    assert lock["count"] == 17
    sources = [t["source"] for t in lock["tools"].values()]
    assert sources.count("mcp") == 13 and sources.count("okr") == 4
    for name in ("lammps_run", "code_runner", "write_file", "register_artefact", "record_metric"):
        assert name in lock["tools"], name


def test_okr_store_matches_the_atomagents_schema(tmp_path):
    """session_stats() reads memory.db. If a baseline's tables differ, every
    baseline manifest reports structural zeros that look like failure."""
    from benchmarks.okr_tools import OKRStore

    store = OKRStore(tmp_path / "memory.db", tmp_path)
    f = tmp_path / "d.dat"
    f.write_text("x" * 10)
    store.register_artefact(str(f), "data")
    store.record_metric("Kt", 3.02, "")
    store.log_message("ReActAgent", "{}", "hello")

    stats = rn.session_stats(tmp_path)
    assert stats["artefacts"] == 1
    assert stats["metrics"] == 1
    assert stats["messages"] == 1
    assert stats["agents"] == {"ReActAgent": 1}
    # No semantic memory in a baseline: honest zero, not the WS26 silent bug.
    assert stats["vectors"] == 0


def test_register_artefact_flags_out_of_session_writes(tmp_path):
    """Containment is measured, not enforced — the MCP jail is runs/jobs for
    every arm, so an agent can write outside its own session (AtomAgents did,
    twice). The tool must record that rather than hide it."""
    from benchmarks.okr_tools import OKRStore

    session = tmp_path / "session"
    session.mkdir()
    outside = tmp_path / "elsewhere.dat"
    outside.write_text("stray")
    store = OKRStore(session / "memory.db", session)
    assert json.loads(store.register_artefact(str(outside)))["contained"] is False
    inside = session / "ok.dat"
    inside.write_text("good")
    assert json.loads(store.register_artefact(str(inside)))["contained"] is True
    # Path semantics, not string prefix: a sibling dir sharing the session's
    # name prefix must read as OUTSIDE.
    sibling = tmp_path / "session-evil"
    sibling.mkdir()
    stray = sibling / "s.dat"
    stray.write_text("stray")
    assert json.loads(store.register_artefact(str(stray)))["contained"] is False


def test_register_artefact_keeps_the_note(tmp_path):
    """The tool advertises a `note` argument; the pinned AtomAgents schema has
    no note column, so it rides in `kind` rather than being silently dropped —
    an advertised argument that vanishes teaches the model its bookkeeping
    is ignored."""
    from benchmarks.okr_tools import OKRStore
    import sqlite3

    store = OKRStore(tmp_path / "memory.db", tmp_path)
    f = tmp_path / "d.dat"
    f.write_text("x")
    store.register_artefact(str(f), kind="data", note="stress-strain gage output")
    con = sqlite3.connect(tmp_path / "memory.db")
    kind = con.execute("select kind from artefacts").fetchone()[0]
    con.close()
    assert "stress-strain gage output" in kind


def test_tool_error_classifier_catches_ordinary_nonzero_exits():
    """The arms' live `error` flags matched only `"status": "error"` and
    `returncode: -1`, so ARM A scored 0 errors across 332 tool calls while its
    results plainly contained Traceback. The scorer recomputes with ONE
    definition so the column means the same thing for every arm."""
    from benchmarks.compare_arms import classify_tool_error as ce

    assert ce('{"stdout": "", "stderr": "Traceback...", "returncode": 1}')
    assert ce('{"returncode": -1, "stderr": "cwd is required"}')
    assert ce('{"status": "error", "error": "no such file"}')
    assert ce('[{"type": "text", "text": "{\\"returncode\\": 2, \\"stderr\\": \\"boom\\"}"}]')
    assert ce('{"errors": ["ERROR: Unknown pair style"]}')

    assert not ce('{"stdout": "ok", "stderr": "", "returncode": 0}')
    assert not ce("")
    # Non-empty stderr on a SUCCESSFUL run must not count: the MCP child emits
    # OMP/matplotlib warnings routinely, and counting them would turn the error
    # rate into a measure of library chattiness.
    assert not ce('{"stdout": "4.15.2", "stderr": "OMP: Warning #179", "returncode": 0}')
    assert not ce('[{"type": "text", "text": "{\\"returncode\\": 0, \\"stderr\\": \\"warn\\"}"}]')

    # Truncated payloads are the case the recorded campaign actually hit: every
    # result over the 20 000-char transcript cap is broken JSON, and the
    # marker fallback missed a demo2-gpt result plainly starting
    # '{"returncode": -1, ...' — the published err cell undercounted by one.
    assert ce(json.dumps({"returncode": 1, "stdout": "x" * 30000})[:20000])
    assert ce(('{"returncode": -1, "errors": [], "output": "' + "y" * 30000)[:20000])
    assert not ce(json.dumps({"returncode": 0, "stdout": "x" * 30000})[:20000])


def test_arm_a_transcript_is_reconstructable_and_marked_as_such():
    """Three mechanism metrics were unmeasurable for ARM A because it writes no
    transcript. It stores populated tool_calls in memory.db, so nothing needed
    re-running — but the reconstruction must never masquerade as a live
    recording, hence the distinct filename."""
    from benchmarks import export_transcript as ex

    assert ex.OUT_NAME == "transcript.arm_a.jsonl"
    assert ex.OUT_NAME != "transcript.jsonl"


def test_tool_output_cap_matches_atomagents_exactly():
    """ARM A caps every tool result at TOOL_OUTPUT_MAX_BYTES via
    MCPToolNode._cap_output; its stored messages top out at 65_581 chars. The
    baselines had no cap, so one md_visualize base64 PNG added ~250_000 tokens
    in a single call and killed two runs. Same constant, same env var, same
    head/tail shape — this is environment parity, not a concession."""
    from AtomAgents import config as ac

    from benchmarks import toolbind as tb

    assert tb.TOOL_OUTPUT_MAX_BYTES == ac.TOOL_OUTPUT_MAX_BYTES == 64 * 1024

    out = tb.cap_output("x" * 3_000_000)
    assert len(out) < tb.TOOL_OUTPUT_MAX_BYTES + 200
    assert "truncated client-side" in out
    # The TAIL must survive: LAMMPS writes its verdict at the end of a log, so a
    # head-only cap would hide the line saying whether the run succeeded.
    assert out.endswith("x")
    assert tb.cap_output("short") == "short"


def test_capping_does_not_change_the_tool_surface():
    """The cap wraps the CALL, never the schema. If it altered names,
    descriptions or schemas the lockfile would drift and the arms would no
    longer be comparable."""
    lock = json.loads((REPO / "benchmarks" / "tools.lock.json").read_text())
    assert lock["digest"].startswith("357bd60c7792587e"), (
        "tool surface changed — regenerate the lock deliberately, and re-run "
        "every arm, or the campaign spans two tool surfaces"
    )


def test_every_arm_has_context_management():
    """Measured the hard way: with no context editing, ARM R's demo1 cell died
    at 74 tool calls with 'prompt is too long: 1007846 tokens > 1000000'
    (17 lammps_run log tails). AtomAgents declares MAX_INPUT_TOKENS=120_000 and
    summarizes long histories, and CrewAI defaults respect_context_window=True,
    so an unmanaged ReAct arm would have measured context management rather
    than architecture. ARM R's trigger must stay comfortably ABOVE AtomAgents'
    window, so the baseline is never the starved side of the comparison."""
    from AtomAgents import config as ac

    from benchmarks import toolbind as tb
    from benchmarks.arms import react_arm as ra

    assert ra.CONTEXT_TRIGGER_TOKENS > ac.MAX_INPUT_TOKENS * 2
    assert ra.KEEP_RECENT_TOOL_RESULTS >= 3
    # ...and with enough headroom below the 1M hard ceiling that `keep` recent
    # results (each now capped at 64 KiB) cannot push it back over.
    assert ra.CONTEXT_TRIGGER_TOKENS < 1_000_000
    headroom = 1_000_000 - ra.CONTEXT_TRIGGER_TOKENS
    worst_case_kept = ra.KEEP_RECENT_TOOL_RESULTS * tb.TOOL_OUTPUT_MAX_BYTES  # chars
    assert headroom > worst_case_kept, "kept results could re-overflow the window"

    # Constants alone do not prove wiring: pin that the middleware is actually
    # instantiated with exact counting (the approximate default undercounts
    # base64 by 2.73x — the reason attempt 2 died) and handed to create_agent,
    # and that ARM C does not opt out of CrewAI's own context management.
    ra_src = Path(ra.__file__).read_text()
    assert 'token_count_method="model"' in ra_src
    assert "middleware=[context_mw]" in ra_src
    from benchmarks.arms import crewai_arm as ca

    ca_src = Path(ca.__file__).read_text()
    assert "respect_context_window" not in ca_src, (
        "CrewAI defaults respect_context_window=True; overriding it would "
        "strip ARM C's only context management"
    )


def test_shared_budget_caps_bind_before_framework_limits():
    """'Raise framework-native caps so ONE shared cap binds first, and record
    which fired.' A run terminated by a framework's own limit is not a
    measurement of the agent.

    ARM R: enforced — the recursion limit sits above the tool-call budget, and
    the pin is on the CONSTANT the call site actually passes, not a formula
    re-derived here (the original assertion was a tautology that stayed green
    under any hardcoded limit).

    ARM C: detected-and-recorded — CrewAI owns its loop, and its per-agent
    max_iter CAN bind before the shared budget (3 x 40 < 240). It did, silently,
    in two of the three recorded WS28 cells; the arm now tallies steps per stage
    and records a cap_hit, so a framework truncation can never again read as a
    voluntary stop."""
    from benchmarks.arms import crewai_arm as ca
    from benchmarks.arms import react_arm as ra

    assert ra.MAX_TOOL_CALLS > 0 and ra.MAX_LLM_CALLS > 0
    assert ra.RECURSION_LIMIT > ra.MAX_TOOL_CALLS + ra.MAX_LLM_CALLS
    ra_src = Path(ra.__file__).read_text()
    assert '"recursion_limit": RECURSION_LIMIT' in ra_src, (
        "the stream call no longer passes the pinned constant"
    )

    assert ca.MAX_ITER == 40
    ca_src = Path(ca.__file__).read_text()
    assert "max_iter=MAX_ITER" in ca_src
    assert "step_callback" in ca_src and "task_callback" in ca_src
    assert 'result["cap_hit"] = f"max_iter=' in ca_src, (
        "a max_iter truncation must be recorded, not silent"
    )


def test_both_atomagent_arms_use_remote_embeddings():
    """WS27 ran the gpt arm on local MiniLM (384-dim), which made those cells
    report-ineligible under the exclude-local-embeddings rule. Since WS29 both
    AtomAgents arms embed with text-embedding-3-large (3072-dim) so the arms
    differ only in exec_tools; the runner still writes the RESOLVED backend
    into manifest['embeddings'] because `requested` is not `used`."""
    assert rn.ARMS["opus"]["env"]["ATOMAGENTS_EMBED_PROVIDER"] == "remote"
    assert rn.ARMS["gpt"]["env"]["ATOMAGENTS_EMBED_PROVIDER"] == "remote"


def test_optional_int_params_bind_as_int_not_str():
    """FastMCP publishes `nz: int | None = None` as an anyOf with no top-level
    type. Falling back to str made pydantic REJECT the integer the tool wants
    (v2 does not coerce int -> str), so the natural call nz=8 failed validation
    in both baseline arms while '8' passed."""
    from benchmarks import toolbind as tb

    schema = {
        "type": "object",
        "properties": {
            "nz": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
            "path": {"type": "string"},
        },
        "required": ["path"],
    }
    M = tb._pydantic_model("probe", schema)
    inst = M(path="/x", nz=8)
    assert inst.nz == 8


def test_sse_parser_skips_notification_events():
    """A streamable-HTTP server may emit notifications (logging, progress) on
    the response stream BEFORE the JSON-RPC result. Returning the first data
    event handed the caller None in place of the tool result, and the follow-on
    AttributeError would kill ARM R's loop — scoring an infra event as an
    architecture failure."""
    import httpx

    from benchmarks.mcp_client import _parse

    body = (
        'data: {"jsonrpc": "2.0", "method": "notifications/message", "params": {}}\n'
        "\n"
        'data: {"jsonrpc": "2.0", "id": 1, "result": {"ok": true}}\n'
    )
    resp = httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=body.encode()
    )
    payload = _parse(resp)
    assert payload is not None and payload["result"] == {"ok": True}


def test_shim_neutralizes_react_markers_in_the_thought():
    """CrewAI's parser splits the shim's output at the FIRST Action:/Action
    Input: and treats any Final Answer: as terminal. A thought quoting an
    earlier step therefore mis-parsed ~a quarter of the recorded ARM C tool
    calls (16/64, 6/63, 22/96). Only the appended real Action pair may remain
    parseable."""
    from benchmarks.arms.crewai_arm import _neutralize_react_markers as nz

    echoed = 'as before: Action Input: {"path": "x"} Observation: ok Final Answer: none yet'
    out = nz(echoed)
    for marker in ("Action Input:", "Action:", "Observation:", "Final Answer:"):
        assert marker not in out, marker
    assert "Action Input -" in out  # defused, not deleted: the thought survives
    assert nz("plain reasoning text") == "plain reasoning text"


def test_crewai_arm_never_wires_the_off_gateway_repair_path():
    """function_calling_llm routes CrewAI's tool-argument repair through
    instructor -> litellm.completion with neither api_base nor api_key: every
    trigger bypassed the shim, left the local gateway, and died with an auth
    error — 64/24/88 times in the recorded cells. With it unset the agent gets
    the stock, actionable 'tool arguments error' observation instead."""
    from benchmarks.arms import crewai_arm as ca

    src = Path(ca.__file__).read_text()
    assert "function_calling_llm=" not in src.replace(
        "NO function_calling_llm", ""
    ), "the repair path is wired to a client that cannot reach the gateway"


def test_relaunched_cell_archives_stale_session(tmp_path):
    """WS28 defect #7: mkdir(exist_ok=True) + OKRStore appending to an existing
    memory.db meant a relaunched cell RESUMED on top of its previous attempt
    (session-ws27-demo2-react inherited 12 stale messages from a killed run).
    Every launch must start from an empty cell, with the old contents archived
    — never deleted, never inherited."""
    d = tmp_path / "session-x"
    d.mkdir()
    (d / "memory.db").write_text("stale rows")
    dest = rn.archive_superseded(d, archive_root=tmp_path / "arch")
    assert dest is not None
    assert not d.exists(), "the stale cell must be moved, not merged into"
    assert (dest / "memory.db").read_text() == "stale rows", "evidence preserved"

    # An empty or absent cell is a no-op.
    d.mkdir()
    assert rn.archive_superseded(d, archive_root=tmp_path / "arch") is None
    assert rn.archive_superseded(tmp_path / "nonexistent", archive_root=tmp_path / "arch") is None


def test_baseline_timeouts_cover_the_arms_worst_case_and_keep_the_manifest():
    """The old single cap (6 h) sat below a healthy crew's legitimate worst
    case (3 sequential agents x 3 h), and an uncaught TimeoutExpired threw away
    the manifest for a cell holding hours of real work — leaving it invisible
    to compare_arms and silently re-run by --skip-existing."""
    from benchmarks.arms import crewai_arm as ca
    from benchmarks.arms import react_arm as ra

    assert rn.ARMS["react"]["timeout_s"] > ra.WALL_CLOCK_SEC
    assert rn.ARMS["crewai"]["timeout_s"] > 3 * ca.WALL_CLOCK_SEC
    src = Path(rn.__file__).read_text()
    assert "except subprocess.TimeoutExpired" in src, (
        "a timeout must still write the manifest"
    )
    assert "runner_subprocess_timeout" in src


def test_assert_lock_refuses_a_drifted_surface(tmp_path, monkeypatch):
    """The lockfile tests pin static bytes; this pins the RUNTIME refusal the
    docstrings promise — an arm facing a drifted tool surface must raise, not
    run an incomparable cell."""
    from benchmarks import toolbind as tb

    def spec(name, source, desc="d"):
        return tb.ToolSpec(name, desc, {"type": "object", "properties": {}},
                           lambda **kw: "", source)

    specs = [spec(f"t{i}", "mcp") for i in range(13)] + [spec(f"o{i}", "okr") for i in range(4)]
    monkeypatch.setattr(tb, "LOCKFILE", tmp_path / "lock.json")
    tb.write_lock(specs)
    tb.assert_lock(specs)  # identical surface passes

    drifted = [spec(f"t{i}", "mcp") for i in range(13)] + [spec(f"o{i}", "okr") for i in range(4)]
    drifted[0].description = "changed description"
    with pytest.raises(tb.ToolBindError, match="drifted"):
        tb.assert_lock(drifted)


def test_export_reconstructs_arm_a_tool_calls_from_memory_db(tmp_path):
    """The ARM A transcript is REBUILT from memory.db msg_json; before this
    test, export() had zero coverage and only its filename constant was pinned.
    Feed it the stored LangChain shapes and check the three mechanism metrics
    it exists to recover."""
    import sqlite3

    from benchmarks import export_transcript as ex
    from benchmarks.okr_tools import OKRStore

    OKRStore(tmp_path / "memory.db", tmp_path)  # creates the schema
    rows = [
        ("Supervisor", {"type": "human", "data": {"content": "do the thing"}}),
        ("Simulator", {"type": "ai", "data": {
            "content": "running",
            "tool_calls": [
                {"name": "bash_run", "args": {"command": "ls"}},          # cwd omitted
                {"name": "code_runner", "args": {"code": "1", "cwd": "/x"}},
            ]}}),
        ("Simulator", {"type": "tool", "data": {
            "name": "bash_run", "content": '{"status": "error", "error": "cwd is required"}'}}),
    ]
    con = sqlite3.connect(tmp_path / "memory.db")
    for agent, msg in rows:
        con.execute(
            "insert into messages (msg_json, msg_text, job_id, embedding, agent_name) "
            "values (?,?,?,?,?)",
            (json.dumps(msg), "", "", "", agent),
        )
    con.commit()
    con.close()

    stats = ex.export(tmp_path)
    assert stats["tool_calls"] == 2
    assert stats["cwd_omission"] == 1
    assert stats["tool_errors"] == 1
    out = tmp_path / ex.OUT_NAME
    first = json.loads(out.read_text().splitlines()[0])
    assert first["kind"] == "provenance", "a reconstruction must declare itself"


# --------------------------------------------------------------------------
# release hardening (v1.0.0): nothing below changes what an arm receives
# --------------------------------------------------------------------------


def test_interpreters_are_env_overridable():
    """Machine paths were hardcoded (./venv and a CrewAI venv under a sibling
    scaffold). A public checkout must be able to point both at its own envs."""
    py, crewai_py = rn.resolve_interpreters({"OKR_PYTHON": "/x/py", "OKR_CREWAI_PYTHON": "/y/py"})
    assert (py, crewai_py) == ("/x/py", "/y/py")
    py, crewai_py = rn.resolve_interpreters({})
    assert py.endswith("python") or py.endswith("python3")
    assert crewai_py.endswith("benchmarks/crewai-env/.venv/bin/python")
    assert rn.ARMS["crewai"]["python"] == rn.ARMS["crewai_gpt"]["python"] == rn.CREWAI_PY


def test_run_id_without_prefix_has_no_double_dash():
    """`--prefix ""` used to produce `session--demo1-gpt`; the dataset layout is
    prefix-less, so the default must be too."""
    assert rn.run_id_for("", "demo1", "gpt", 2) == "demo1-gpt-t2"
    assert rn.run_id_for("", "demo1", "gpt", None) == "demo1-gpt"
    assert rn.run_id_for("ws29", "fem_demo0", "opus", 3) == "ws29-fem_demo0-opus-t3"


def test_all_excludes_the_withheld_ablation_by_default():
    """`--all` without --arms must run exactly the six published configurations."""
    assert set(rn.default_arms()) == {"gpt", "opus", "react", "react_gpt", "crewai", "crewai_gpt"}
    assert "opus_std" in rn.ARMS and rn.ARMS["opus_std"].get("withheld")


def test_paper_model_ids_are_the_defaults_and_recorded(tmp_path, monkeypatch):
    """The campaign took its model ids from .env; with none set, the runner
    must pin the published identifiers and write them into the manifest."""
    for key in ("ATOMAGENTS_OPENAI_MODEL", "ATOMAGENTS_ANTHROPIC_MODEL", "BENCH_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(rn, "REPO", tmp_path)        # no .env, no git, no templates
    monkeypatch.setattr(rn, "JOBS", tmp_path / "jobs")
    monkeypatch.setattr(rn, "LOGS", tmp_path / "logs")
    m = rn.run_one("demo1", "gpt", dry_run=True)
    assert m["run_id"] == "demo1-gpt"
    assert m["models"] == {
        "openai": rn.PAPER_MODELS["openai"],
        "anthropic": rn.PAPER_MODELS["anthropic"],
        "bench_model": rn.PAPER_MODELS["openai"],
    }
    assert m["runner_release"] == rn.RUNNER_RELEASE
    assert "mcp_preflight" not in m, "a dry run must not touch the tool service"
    # and an explicit .env value wins over the published default
    monkeypatch.setenv("ATOMAGENTS_OPENAI_MODEL", "my-model")
    assert rn.run_one("demo1", "gpt", dry_run=True)["models"]["openai"] == "my-model"


def test_backend_tool_count_is_parsed():
    log = "[OpenAI Backend] startup\n[OpenAI Backend] Loaded 13 MCP tools (on demand)\nmore\n"
    assert rn.backend_tool_count(log) == 13
    assert rn.backend_tool_count("[OpenAI Backend] Failed to get MCP tools: boom") is None
    from benchmarks import toolbind as tb

    assert rn.EXPECTED_MCP_TOOLS == tb.EXPECTED_MCP_TOOLS


def test_mcp_preflight_refuses_a_drifted_surface(tmp_path, monkeypatch):
    """The HTTP arms used to start a billed run with the tool service down
    (the backend continues with 0 MCP tools). The preflight must raise on a
    missing or drifted surface and report the pinned one when it matches."""
    from benchmarks import mcp_client as mc
    from benchmarks import toolbind as tb

    class FakeClient:
        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def spec(name, source, desc="d"):
        return tb.ToolSpec(name, desc, {"type": "object", "properties": {}}, lambda **kw: "", source)

    good = [spec(f"t{i}", "mcp") for i in range(13)] + [spec(f"o{i}", "okr") for i in range(4)]
    monkeypatch.setattr(mc, "MCPClient", FakeClient)
    monkeypatch.setattr(tb, "LOCKFILE", tmp_path / "lock.json")
    tb.write_lock(good)

    monkeypatch.setattr(tb, "tool_specs", lambda client, store: good)
    info = rn.preflight_mcp("http://fake/mcp/")
    assert (info["tools"], info["mcp"]) == (17, 13)
    assert info["url"] == "http://fake/mcp"

    drifted = [spec(f"t{i}", "mcp", "changed" if i == 0 else "d") for i in range(13)] + good[13:]
    monkeypatch.setattr(tb, "tool_specs", lambda client, store: drifted)
    with pytest.raises(tb.ToolBindError, match="drifted"):
        rn.preflight_mcp("http://fake/mcp/")


def test_lammps_banner_prefers_the_version_line():
    """The study-era probe recorded the options header of `lmp -h` instead of
    the build date (it kept the first line containing "LAMMPS")."""
    help_text = (
        "\nLarge-scale Atomic/Molecular Massively Parallel Simulator - 22 Jul 2025 - Update 3\n\n"
        "Usage example: /usr/local/bin/lmp_mpi -var t 300 -echo screen -in in.alloy\n\n"
        "List of command-line options supported by this LAMMPS executable:\n"
    )
    assert rn._lammps_banner(help_text).endswith("22 Jul 2025 - Update 3")
    assert rn._lammps_banner("LAMMPS (22 Jul 2025 - Update 3)\nwarning") == "LAMMPS (22 Jul 2025 - Update 3)"
    assert rn._lammps_banner("no banner here") == ""
