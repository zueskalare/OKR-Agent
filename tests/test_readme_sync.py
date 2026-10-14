"""README board: marker sections, single-writer dedupe, projection rendering."""

from AtomAgents.utils.readme_sync import (
    ensure_readme,
    get_section,
    render_artefacts_markdown,
    render_tasks_markdown,
    update_section,
)


def test_marker_roundtrip(tmp_path):
    ensure_readme(tmp_path)
    update_section(tmp_path, "Tasks", "task content v1")
    assert get_section(tmp_path, "Tasks") == "task content v1"
    update_section(tmp_path, "Tasks", "task content v2")
    text = (tmp_path / "README.md").read_text()
    assert "task content v1" not in text
    assert get_section(tmp_path, "Tasks") == "task content v2"


def test_stray_duplicate_heading_removed(tmp_path):
    ensure_readme(tmp_path)
    update_section(tmp_path, "Simulator", "authoritative")
    with open(tmp_path / "README.md", "a") as f:
        f.write("\n## Simulator\nrogue shell-written content\n")
    update_section(tmp_path, "Simulator", "authoritative v2")
    text = (tmp_path / "README.md").read_text()
    assert text.count("## Simulator") == 1
    assert "rogue" not in text
    assert get_section(tmp_path, "Simulator") == "authoritative v2"


def test_tasks_rendering_with_rollups():
    tasks = [{"id": 1, "job_id": "J1", "description": "Build", "status": "in_progress"}]
    subtasks = [
        {"id": 1, "task_id": 1, "description": "geometry", "status": "done"},
        {"id": 2, "task_id": 1, "description": "potential", "status": "pending"},
    ]
    krs = [{"id": 1, "task_id": 1, "description": "structure.data exists", "achieved": 1}]
    md = render_tasks_markdown(tasks, subtasks, krs)
    assert "(1/2 subtasks done)" in md
    assert "KR #1 [x]" in md


def test_artefacts_rendering():
    md = render_artefacts_markdown(
        [{"path": "/x/stress.dat", "kind": "data", "bytes": 123, "kr_id": 1}],
        [{"metric": "final_stress", "value": 14.3, "unit": "GPa", "kr_id": 1}],
    )
    assert "stress.dat" in md and "final_stress" in md
