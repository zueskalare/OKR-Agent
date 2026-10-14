"""Tool-layer behaviors: log capture, launcher resolution, path guards,
file reading windows, write_file."""

import importlib.util
import json
from pathlib import Path

import pytest


TOOLS_PATH = Path(__file__).parent.parent / "AtomAgents" / "mcp_service" / "tools.py"
spec = importlib.util.spec_from_file_location("mcp_tools_under_test", TOOLS_PATH)
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)


@pytest.fixture()
def jail_root(tmp_path, monkeypatch):
    """Write/exec guards fail closed since WS23 — give them a jail."""
    monkeypatch.setenv("ATOMAGENTS_TOOL_ROOT", str(tmp_path))
    monkeypatch.delenv("ATOMAGENTS_ALLOW_UNJAILED", raising=False)
    return tmp_path


def test_in_script_log_redirect_is_honored(tmp_path):
    # Historical bug: the runner read only its -log target, but agent scripts
    # contain their own `log <path>` command redirecting LAMMPS elsewhere.
    real = tmp_path / "real.log"
    real.write_text("LAMMPS (2 Aug 2023)\nERROR: Unknown thermo_modify keyword: file\nlast\n")
    script = "units metal\nlog real.log\nrun 0\n"
    text, used = tools._read_lammps_log(script, tmp_path, tmp_path / "default.txt")
    assert "ERROR: Unknown" in text
    assert used.endswith("real.log")


def test_launcher_env_override(monkeypatch):
    monkeypatch.setenv("ATOMAGENTS_LAMMPS_CMD", "mpirun -np 4 lmp_custom")
    cmd = tools._lammps_command(Path("/x/in.file"), Path("/x/out.log"))
    assert cmd[:4] == ["mpirun", "-np", "4", "lmp_custom"]


def test_workspace_path_guard(jail_root, monkeypatch):
    assert tools._guard_workspace_path("") is not None
    assert tools._guard_workspace_path("/input") is not None
    assert tools._guard_workspace_path("relative/path") is not None
    assert tools._guard_workspace_path("/usr/lib/x") is not None
    assert tools._guard_workspace_path(str(jail_root / "proj" / "x")) is None
    # S2: without a jail the guard fails closed instead of allowing anywhere
    monkeypatch.delenv("ATOMAGENTS_TOOL_ROOT", raising=False)
    err = tools._guard_workspace_path(str(jail_root / "proj" / "x"))
    assert err and "without a workspace jail" in err


def test_file_reader_sniffs_content_not_extension(tmp_path):
    data_file = tmp_path / "model.data"  # previously blocked by whitelist
    data_file.write_text("LAMMPS data file\n10 atoms\n")
    ok, res = tools.mcp_file_reader(str(data_file))
    assert ok and "10 atoms" in json.loads(res)[0]["text"]

    binary = tmp_path / "traj.bin"
    binary.write_bytes(b"\x00\x01\x02")
    ok, res = tools.mcp_file_reader(str(binary))
    assert not ok and "Binary file" in res


def test_file_reader_windows_large_files(tmp_path):
    big = tmp_path / "big.log"
    big.write_text("line\n" * 100000)
    ok, res = tools.mcp_file_reader(str(big))
    text = json.loads(res)[0]["text"]
    assert ok and "head+tail window" in text
    assert len(text) < 70000


def test_write_file_guards_and_writes(tmp_path, jail_root):
    target = tmp_path / "input" / "run.in"
    r = json.loads(tools.mcp_write_file(str(target), "units metal\n"))
    assert r["bytes_written"] == len("units metal\n")
    assert target.read_text() == "units metal\n"
    for bad in ("", "/input", "rel/path"):
        assert "error" in json.loads(tools.mcp_write_file(bad, "x"))


def test_lammps_runner_structured_error_without_binary(tmp_path, jail_root, monkeypatch):
    # Force launcher resolution to fail
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("ATOMAGENTS_LAMMPS_CMD", raising=False)
    r = json.loads(tools.mcp_lammps_runner("run 0\n", str(tmp_path / "wd"), "run.in"))
    assert "No LAMMPS executable" in (r.get("error") or "")
