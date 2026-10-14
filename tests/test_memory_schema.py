"""SQLiteRAGMemory: schema, hygiene, retrieval, rebind, OKR/artefact stores."""

import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from AtomAgents.agents.messages import ReportMessage
from AtomAgents.memory.sqlite_rag_memory import SQLiteRAGMemory


class StubEmb:
    def embed_query(self, text):
        h = abs(hash(text))
        return [((h >> i) % 100) / 100 for i in range(8)]


@pytest.fixture()
def mem(tmp_path):
    return SQLiteRAGMemory(
        sqlite_path=str(tmp_path / "memory.db"),
        project_root=str(tmp_path),
        embedding_model=StubEmb(),
    )


def _tables(db_path):
    with sqlite3.connect(db_path) as conn:
        return {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }


def test_full_schema_created(mem):
    tables = _tables(mem.sqlite_path)
    assert {
        "messages",
        "tasks",
        "subtasks",
        "key_results",
        "artefacts",
        "validation_metrics",
        "attempts",
    } <= tables
    # FTS virtual table
    with sqlite3.connect(mem.sqlite_path) as conn:
        names = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master")
        }
    assert "messages_fts" in names


def test_duplicate_and_blob_hygiene(mem):
    long_text = "pair_style must precede read_data in the input script. " * 6
    mem.save_context(inputs=[], outputs=[AIMessage(content=long_text)], agent_name="Simulator")
    mem.save_context(inputs=[], outputs=[AIMessage(content=long_text)], agent_name="Simulator")
    tc = AIMessage(content="", tool_calls=[{"name": "lammps_run", "args": {"work_dir": "/x"}, "id": "t1"}])
    mem.save_context(inputs=[], outputs=[tc], agent_name="Supervisor")
    img = HumanMessage(
        content=[
            {"type": "text", "text": "see image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 100000}},
        ]
    )
    mem.save_context(inputs=[img], outputs=[], agent_name="Supervisor")

    with sqlite3.connect(mem.sqlite_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3  # dup skipped
        tool_text = conn.execute(
            "SELECT msg_text FROM messages WHERE msg_text LIKE '%tool_call%'"
        ).fetchone()
        assert tool_text and "lammps_run" in tool_text[0]
        biggest = conn.execute("SELECT MAX(length(msg_json)) FROM messages").fetchone()[0]
        assert biggest < 10000  # base64 payload replaced by placeholder


def test_lexical_search_survives_raw_prose_queries(mem):
    # FTS5 MATCH treats input as query syntax; raw prose with dots/parens/
    # quotes used to raise "fts5: syntax error" and silently disable the
    # lexical path (caught live in the smoke test).
    mem.save_context(
        inputs=[],
        outputs=[AIMessage(content="pair_style ordering diagnosis details " * 5)],
        agent_name="Simulator",
    )
    nasty = 'SMOKE TEST — do not run. Use okr_update ("Smoke test") with pair_style (v1.2).'
    hits = mem._lexical_search(nasty, limit=5)
    assert hits and "pair_style" in hits[0]["msg_text"]
    assert mem._lexical_search("...(((", limit=5) == []


def test_hybrid_search_and_own_agent_exclusion(mem):
    mem.save_context(
        inputs=[],
        outputs=[AIMessage(content="pair_style ordering error diagnosis " * 5)],
        agent_name="Simulator",
    )
    hits = mem._lexical_search("pair_style", limit=5)
    assert hits and "pair_style" in hits[0]["msg_text"]
    res = mem.search("pair_style", k=3, exclude_agent="Simulator")
    assert all(r["agent_name"] != "Simulator" for r in res)


def test_rebind_backfills_fts_on_legacy_db(mem, tmp_path):
    legacy = tmp_path / "legacy.db"
    with sqlite3.connect(legacy) as conn:
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, msg_json TEXT, "
            "msg_text TEXT, job_id TEXT, objective_id INTEGER, key_result_id INTEGER)"
        )
        conn.execute("ALTER TABLE messages ADD COLUMN embedding TEXT")
        conn.execute("ALTER TABLE messages ADD COLUMN agent_name TEXT")
        conn.execute(
            "INSERT INTO messages (msg_json, msg_text, embedding, agent_name) "
            "VALUES ('{}', 'legacy atom_style angle error', '[]', 'Simulator')"
        )
    mem.rebind(str(legacy), project_root=str(tmp_path))
    hits = mem._lexical_search("atom_style", limit=5)
    assert hits and "legacy" in hits[0]["msg_text"]


def test_update_task_status_matches_primary_key(mem):
    # Regression (ultrareview bug_001): update_task_status used to match
    # WHERE job_id=? (TEXT UUID) while callers pass the integer task PK —
    # every status update silently matched zero rows.
    tid = mem.create_task("a1b2c3d4-uuid-job", "Build model", status="submitted")
    sid = mem.create_subtask(tid, "Geometry", status="pending")
    mem.update_task_status(tid, "done")
    task = mem.get_current_task()
    assert task["id"] == tid and task["status"] == "done"
    # cascade applies to this task's subtasks
    assert all(s["status"] == "done" for s in task["subtasks"]), task["subtasks"]
    with pytest.raises(ValueError):
        mem.update_task_status(999999, "done")


def test_okr_crud_and_artefacts(mem, tmp_path):
    tid = mem.create_task("J1", "Build model")
    assert mem.task_exists(tid)
    sid = mem.create_subtask(tid, "Geometry")
    mem.update_subtask_status(sid, "done")

    f = tmp_path / "structure.data"
    f.write_text("atoms...\n")
    info = mem.register_artefact(str(f), kind="data", task_id=tid)
    assert info["bytes"] > 0 and len(info["hash"]) == 64
    with pytest.raises(FileNotFoundError):
        mem.register_artefact(str(tmp_path / "missing.data"))
    empty = tmp_path / "empty.dat"
    empty.touch()
    with pytest.raises(ValueError):
        mem.register_artefact(str(empty))

    mem.record_metric("final_stress", 14.3, unit="GPa")
    mem.record_attempt("atom_style bond", "failed", error_excerpt="No angles allowed")
    assert mem.get_all_metrics()[0]["metric"] == "final_stress"
    assert mem.get_recent_attempts()[0]["result"] == "failed"


def test_report_message_survives_roundtrip(mem):
    mem.save_context(
        inputs=[], outputs=[ReportMessage(content="handoff", name="Modeler")], agent_name="Modeler"
    )
    msgs = mem.get_all_raw_messages()
    assert any(type(m).__name__ == "ReportMessage" for m in msgs)


def test_okr_add_task_dedupes(mem, tmp_path):
    # Live-smoke regression: a looping run re-added the same objective 44x.
    from AtomAgents.models.okr_models import TaskOperation
    from AtomAgents.utils.okr_tools import apply_task_operation

    apply_task_operation(mem, TaskOperation(op="add_task", job_id="J1", description="Smoke test"))
    with pytest.raises(ValueError, match="already exists as id 1"):
        apply_task_operation(mem, TaskOperation(op="add_task", job_id="J2", description="  smoke TEST "))
    assert len(mem.get_all_tasks()) == 1
    apply_task_operation(mem, TaskOperation(op="add_subtask", task_id=1, description="Verify board"))
    with pytest.raises(ValueError, match="Subtask already exists"):
        apply_task_operation(mem, TaskOperation(op="add_subtask", task_id=1, description="verify board"))
