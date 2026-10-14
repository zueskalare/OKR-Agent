"""MCP tool sandbox (WS20.1) — the incident fix, pinned.

The recorded incident: a cwd-less bash_run inherited the SERVICE cwd (this
repo) and the full env (git identity via HOME, API keys), letting a "hi"
session commit to git. These are the first tests mcp_service guards have
ever had.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from AtomAgents.mcp_service import tools as T


@pytest.fixture()
def jail(tmp_path, monkeypatch):
    root = tmp_path / "jobs"
    root.mkdir()
    session = root / "session-x"
    session.mkdir()
    monkeypatch.setenv("ATOMAGENTS_TOOL_ROOT", str(root))
    monkeypatch.delenv("ATOMAGENTS_ALLOW_GIT", raising=False)
    return root, session


# ---------------------------------------------------------------------------
# The jail (guard extension)
# ---------------------------------------------------------------------------


def test_guard_allows_inside_jail(jail):
    root, session = jail
    assert T._guard_workspace_path(str(session / "out.txt")) is None


def test_guard_denies_outside_jail(jail, tmp_path):
    outside = tmp_path / "elsewhere" / "f.txt"
    err = T._guard_workspace_path(str(outside))
    assert err and "outside the session root" in err


def test_guard_denies_traversal_escape(jail):
    root, session = jail
    sneaky = str(session / ".." / ".." / "escape.txt")
    err = T._guard_workspace_path(sneaky)
    assert err and "outside the session root" in err


def test_guard_read_mode_skips_jail(jail, tmp_path):
    outside = tmp_path / "potentials" / "Fe.eam.fs"
    # reads are deliberately NOT jailed (potential files live outside)
    assert T._guard_workspace_path(str(outside), jail=False) is None
    # but system dirs stay denied even for reads
    assert T._guard_workspace_path("/etc/passwd", jail=False) is not None


def test_guard_fails_closed_without_tool_root(monkeypatch, tmp_path):
    # S2: no jail root means NO writes/execs — not "anywhere goes" (the old,
    # fail-open behaviour this test used to pin).
    monkeypatch.delenv("ATOMAGENTS_TOOL_ROOT", raising=False)
    monkeypatch.delenv("ATOMAGENTS_ALLOW_UNJAILED", raising=False)
    err = T._guard_workspace_path(str(tmp_path / "anywhere.txt"))
    assert err and "without a workspace jail" in err


def test_guard_fails_closed_when_tool_root_missing_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ATOMAGENTS_TOOL_ROOT", str(tmp_path / "does-not-exist"))
    monkeypatch.delenv("ATOMAGENTS_ALLOW_UNJAILED", raising=False)
    err = T._guard_workspace_path(str(tmp_path / "anywhere.txt"))
    assert err and "without a workspace jail" in err


def test_guard_unjailed_opt_in_restores_legacy(monkeypatch, tmp_path):
    monkeypatch.delenv("ATOMAGENTS_TOOL_ROOT", raising=False)
    monkeypatch.setenv("ATOMAGENTS_ALLOW_UNJAILED", "1")
    assert T._guard_workspace_path(str(tmp_path / "anywhere.txt")) is None
    assert T._guard_workspace_path("/usr/local/x") is not None


# ---------------------------------------------------------------------------
# bash_run
# ---------------------------------------------------------------------------


def test_bash_requires_cwd(jail):
    out = json.loads(T.mcp_bash_runner("echo hi", None, 30))
    assert out["returncode"] == -1
    assert "cwd is required" in out["stderr"]


def test_bash_denies_git(jail):
    root, session = jail
    out = json.loads(T.mcp_bash_runner("git status", str(session), 30))
    assert out["returncode"] == -1
    assert "git commands are disabled" in out["stderr"]
    # ...including chained forms
    out2 = json.loads(T.mcp_bash_runner("echo x && git commit -m y", str(session), 30))
    assert "git commands are disabled" in out2["stderr"]


def test_bash_git_allow_flag(jail, monkeypatch):
    root, session = jail
    monkeypatch.setenv("ATOMAGENTS_ALLOW_GIT", "1")
    out = json.loads(T.mcp_bash_runner("git --version", str(session), 30))
    assert out["returncode"] == 0


def test_bash_denies_dangerous_patterns(jail):
    root, session = jail
    for cmd in ("sudo ls", "rm -rf /", "shutdown -h now", "ssh host"):
        out = json.loads(T.mcp_bash_runner(cmd, str(session), 30))
        assert out["returncode"] == -1, cmd
        assert "denylist" in out["stderr"] or "disabled" in out["stderr"], cmd


def test_bash_scrubbed_env(jail, monkeypatch):
    root, session = jail
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    out = json.loads(
        T.mcp_bash_runner('echo "key=[$OPENAI_API_KEY] home=[$HOME]"', str(session), 30)
    )
    assert out["returncode"] == 0
    assert "sk-secret" not in out["stdout"]
    assert str(session) in out["stdout"]  # HOME redirected into the workspace
    # git identity can't resolve: HOME has no .gitconfig
    assert not (session / ".home" / ".gitconfig").exists()


def test_bash_denies_cwd_outside_jail(jail, tmp_path):
    out = json.loads(T.mcp_bash_runner("echo hi", str(tmp_path / "outside"), 30))
    assert out["returncode"] == -1
    assert "outside the session root" in out["stderr"]


def test_bash_timeout_clamped(jail, monkeypatch):
    monkeypatch.setenv("ATOMAGENTS_BASH_TIMEOUT_MAX", "2")
    assert T._clamp_timeout(9999, "ATOMAGENTS_BASH_TIMEOUT_MAX", 600) == 2
    assert T._clamp_timeout(1, "ATOMAGENTS_BASH_TIMEOUT_MAX", 600) == 1
    assert T._clamp_timeout(None, "ATOMAGENTS_BASH_TIMEOUT_MAX", 600) == 2


def test_bash_runs_inside_jail(jail):
    root, session = jail
    out = json.loads(
        T.mcp_bash_runner(f"echo data > {session}/f.txt && cat {session}/f.txt",
                          str(session), 30)
    )
    assert out["returncode"] == 0
    assert "data" in out["stdout"]


# ---------------------------------------------------------------------------
# code_runner
# ---------------------------------------------------------------------------


def test_code_runner_requires_cwd(jail):
    out = json.loads(T.mcp_code_runner("print('x')", None, 30))
    assert out["returncode"] == -1
    assert "cwd is required" in out["stderr"]


def test_code_runner_scrubbed_env_and_script_location(jail, monkeypatch):
    root, session = jail
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    code = (
        "import os\n"
        "print('key=[' + os.environ.get('OPENAI_API_KEY', '') + ']')\n"
        "print('file=' + __file__)\n"
    )
    out = json.loads(T.mcp_code_runner(code, str(session), 30))
    assert out["returncode"] == 0
    assert "key=[]" in out["stdout"]  # API key scrubbed
    assert str(session) in out["stdout"]  # script lives inside the jail


# ---------------------------------------------------------------------------
# write_file / file_reader / lammps log containment
# ---------------------------------------------------------------------------


def test_write_file_jailed(jail, tmp_path):
    root, session = jail
    ok = json.loads(T.mcp_write_file(str(session / "a.txt"), "hello"))
    assert ok.get("path")
    denied = json.loads(T.mcp_write_file(str(tmp_path / "outside.txt"), "x"))
    assert "outside the session root" in denied.get("error", "")


def test_file_reader_image_size_cap(jail, monkeypatch):
    root, session = jail
    big = session / "big.png"
    big.write_bytes(b"\x89PNG" + b"0" * 1024)
    monkeypatch.setenv("ATOMAGENTS_TOOL_BINARY_MAX", "100")
    ok, res = T.mcp_file_reader(str(big))
    assert ok is False
    assert "too large" in json.loads(res)["error"]


def test_lammps_log_follow_contained(jail, tmp_path):
    root, session = jail
    secret = tmp_path / "secret.log"
    secret.write_text("SECRET CONTENT THAT MUST NOT LEAK padded to be long " * 20)
    default_log = session / "log.lammps"
    default_log.write_text("normal log")
    script = f"log {secret}\nrun 100\n"
    text, used = T._read_lammps_log(script, session, default_log)
    assert "SECRET" not in text
    assert used == str(default_log)


# ---------------------------------------------------------------------------
# S3 — denylist bypass corpus (WS21 probe table; 18/20 used to be ALLOWED)
# ---------------------------------------------------------------------------

BYPASS_CORPUS = [
    "echo hi\ngit status",
    "/usr/bin/git status",
    "env git status",
    "  git status",
    "bash -c 'git status'",
    "`git status`",
    'g"i"t status',
    "echo hi\nsudo ls",
    "/usr/bin/sudo ls",
    "rm -rf /Users",
    "rm -rf --no-preserve-root /",
    "Git status",
    "GIT status",
    'echo"git status"',
    "g\\it status",
    "$(which git) status",
    "x=1; git log",
    "sudo ls",
    "rm -rf /",
    "shutdown -h now",
    "ssh host",
    "mkfs.ext4 /dev/sda",
]

SAFE_CORPUS = [
    "legit_tool --sudo-mode",
    "rm -rf ./build",
    "rm out.txt",
    "rm -rf /Users/jt/runs/jobs/s1/tmp",
    'grep -r "digit" .',
    "cat .gitignore",
    "mkdir -p data && cd data",
    "lmp_mpi -in in.lammps",
    'echo "commit the results"',
    'python -c "print(1)"',
    "tail -n 50 log.lammps",
    "cp a b",
    "mv a b",
    "ls /usr/bin",
    'find . -name "*.dat"',
    "echo $HOME",
    "sshd_config_check --dry",
    "python analyze.py",
    'awk "{print $1}" stress.dat',
    "head -5 model1.data",
]


@pytest.mark.parametrize("cmd", BYPASS_CORPUS)
def test_deny_command_bypass_corpus(cmd, monkeypatch):
    monkeypatch.delenv("ATOMAGENTS_ALLOW_GIT", raising=False)
    assert T._deny_command(cmd) is not None, cmd


@pytest.mark.parametrize("cmd", SAFE_CORPUS)
def test_deny_command_allows_safe_commands(cmd, monkeypatch):
    monkeypatch.delenv("ATOMAGENTS_ALLOW_GIT", raising=False)
    assert T._deny_command(cmd) is None, cmd


def test_deny_command_git_allow_flag_still_denies_dangerous(monkeypatch):
    monkeypatch.setenv("ATOMAGENTS_ALLOW_GIT", "1")
    assert T._deny_command("git status") is None
    assert T._deny_command("sudo ls") is not None


def test_bash_run_rejects_bypass_corpus_end_to_end(jail):
    root, session = jail
    for cmd in ("echo hi\ngit status", "/usr/bin/git status", "env git status",
                'g"i"t status', "rm -rf /Users"):
        out = json.loads(T.mcp_bash_runner(cmd, str(session), 30))
        assert out["returncode"] == -1, cmd
        assert "disabled" in out["stderr"] or "denylist" in out["stderr"], cmd


# ---------------------------------------------------------------------------
# S5 — the denylist also covers code_runner
# ---------------------------------------------------------------------------


def test_code_runner_denies_shelling_out_to_git(jail):
    root, session = jail
    code = 'import subprocess\nsubprocess.run(["git", "status"])\n'
    out = json.loads(T.mcp_code_runner(code, str(session), 30))
    assert out["returncode"] == -1
    assert "git commands are disabled" in out["stderr"]


def test_code_runner_allows_plain_science_code(jail):
    root, session = jail
    code = (
        "# mentions digits and .gitignore, which must not false-positive\n"
        "print(sum(range(10)))\n"
    )
    out = json.loads(T.mcp_code_runner(code, str(session), 30))
    assert out["returncode"] == 0
    assert "45" in out["stdout"]


# ---------------------------------------------------------------------------
# S16 — the code_runner scan reads Python as Python, not as a shell line
# ---------------------------------------------------------------------------
#
# Wiring the SHELL denylist into code_runner (the S5 fix) made it token-scan
# Python source, so domain code was refused: `Su = 250e6`, `halt = False`, a
# "shutdown" in a comment. Only shell-escape call sites are scanned now.
# PYTHON_SAFE_CORPUS is deliberately NOT merged into SAFE_CORPUS above — that
# corpus pins BASH semantics, where `Su = 250e6` must keep being denied.

PYTHON_SAFE_CORPUS = [
    "Su = 250e6  # ultimate strength",
    "halt = False",
    "print('reboot the loop')",
    "'''deploy via ssh later'''",
    "shutdown_after = 5",
    'df = pd.read_csv("sudo_results.csv")',
    # realistic FEM post-processing
    (
        "import numpy as np\n"
        "from dolfinx import fem\n"
        "Su = 250e6  # ultimate tensile strength [Pa]\n"
        "halt = False\n"
        "u_max = float(np.max(np.abs(uh.x.array)))\n"
        "if u_max > 1e-2:\n"
        "    halt = True\n"
        "print(f'max_displacement={u_max:.6e} m, Su={Su:.3e}')\n"
    ),
    # realistic LAMMPS log post-processing
    (
        "import numpy as np\n"
        "reboot_count = 0\n"
        "data = np.loadtxt('stress.dat', skiprows=1)\n"
        "strain, stress = data[:, 0], data[:, 1]\n"
        "su = stress.max()\n"
        "print('ultimate stress', su, 'restarts', reboot_count)\n"
    ),
    # denylist words as argv DATA (not the executable) must pass: only
    # argv[0] runs (the F5 false-positive class).
    'import subprocess\nsubprocess.run(["grep", "halt", "log.lammps"])',
    'import subprocess\nsubprocess.run(["echo", "reboot count: 3"])',
    'import subprocess\nsubprocess.check_output(["awk", "/halt/{print}", "log"])',
]

PYTHON_DENY_CORPUS = [
    'import subprocess\nsubprocess.run(["git", "status"])',
    'import os\nos.system("shutdown -h now")',
    'import subprocess\nsubprocess.check_output("sudo ls", shell=True)',
    'import subprocess\nsubprocess.Popen(["ssh", "host"])',
    'import os\nm = "x"\nos.system(f"git commit -m {m}")',
    'from os import system\nsystem("sudo rm -rf /")',
    # F4: aliased / from-imported exec APIs must not bypass the scan
    'import subprocess as sp\nsp.run("ssh user@host", shell=True)',
    'from subprocess import run\nrun(["git", "commit", "-m", "x"])',
    'from subprocess import check_output as co\nco("sudo ls", shell=True)',
    # F5: shell/wrapper argv[0] — the REST is command text and is scanned
    'import subprocess\nsubprocess.run(["bash", "-c", "git push"])',
    'import subprocess\nsubprocess.run(["env", "git", "status"])',
]


@pytest.mark.parametrize("src", PYTHON_SAFE_CORPUS)
def test_python_scan_allows_realistic_science_code(src, monkeypatch):
    monkeypatch.delenv("ATOMAGENTS_ALLOW_GIT", raising=False)
    assert T._deny_python_code(src) is None, src


@pytest.mark.parametrize("src", PYTHON_DENY_CORPUS)
def test_python_scan_denies_shell_escapes(src, monkeypatch):
    monkeypatch.delenv("ATOMAGENTS_ALLOW_GIT", raising=False)
    assert T._deny_python_code(src) is not None, src


def test_code_runner_allows_domain_identifiers_end_to_end(jail):
    """The headline S16 regression: `Su`/`halt` are physics, not commands."""
    root, session = jail
    code = "Su = 250e6\nhalt = False\nprint(Su, halt)\n"
    out = json.loads(T.mcp_code_runner(code, str(session), 30))
    assert out["returncode"] == 0, out["stderr"]
    assert "250000000.0" in out["stdout"]


def test_bash_scan_unchanged_for_python_lookalikes(monkeypatch):
    """Bash semantics must not be weakened: as a shell line, `Su = 250e6`
    still runs the `Su` binary and stays denied."""
    monkeypatch.delenv("ATOMAGENTS_ALLOW_GIT", raising=False)
    assert T._deny_command("Su = 250e6") is not None
    assert T._deny_command("halt") is not None


def test_pathological_source_does_not_raise(jail):
    """F12: CPython's compiler raises RecursionError (not SyntaxError) on
    deeply nested expressions; the guard must return, not explode, and
    code_runner must keep its JSON return contract."""
    bomb = "-1" * 100000
    assert T._python_shell_commands(bomb) == []
    assert T._deny_python_code(bomb) is None
    root, session = jail
    out = json.loads(T.mcp_code_runner(bomb, str(session), 30))
    # The CHILD interpreter rejects it (its own RecursionError/SyntaxError);
    # the tool still returns the structured payload.
    assert "returncode" in out and "stderr" in out
    assert out["returncode"] != 0


def test_repo_python_files_are_not_refused(monkeypatch):
    """No repo source may be refused as a side effect of the guard.

    The ledger measured 4 of 91 .py files rejected by the whole-source scan
    (incl. backend/openai_backend.py, on the bare word "shutdown" in a
    comment). This file is excluded because it genuinely shells out to git in
    the seatbelt repro — a true positive, not the false positive under test.
    """
    monkeypatch.delenv("ATOMAGENTS_ALLOW_GIT", raising=False)
    repo = Path(__file__).resolve().parents[1]
    skip_parts = {"venv", "site-packages", "node_modules", ".git", "external",
                  "Comparison"}
    refused = []
    scanned = 0
    for sub in ("AtomAgents", "backend", "tests"):
        for path in (repo / sub).rglob("*.py"):
            # This file and the release-hygiene test both shell out to git on
            # purpose (true positives of the scan, not false ones).
            if skip_parts & set(path.parts) or path.name in (
                Path(__file__).name, "test_release_hygiene.py"
            ):
                continue
            scanned += 1
            if T._deny_python_code(path.read_text(errors="replace")):
                refused.append(str(path.relative_to(repo)))
    assert scanned > 50, f"corpus too small ({scanned} files) — did the tree move?"
    assert not refused, f"repo sources refused by the code_runner scan: {refused}"


# ---------------------------------------------------------------------------
# S2 — fail closed end-to-end
# ---------------------------------------------------------------------------


def test_write_file_fails_closed_without_root(monkeypatch, tmp_path):
    monkeypatch.delenv("ATOMAGENTS_TOOL_ROOT", raising=False)
    monkeypatch.delenv("ATOMAGENTS_ALLOW_UNJAILED", raising=False)
    denied = json.loads(T.mcp_write_file(str(tmp_path / "x.txt"), "x"))
    assert "without a workspace jail" in denied.get("error", "")


# ---------------------------------------------------------------------------
# S4 + S14 — lammps_run containment
# ---------------------------------------------------------------------------


def _stub_launcher(session, extra_lines=()):
    """Fake LAMMPS launcher: dumps env markers + writes the -log file ($4)."""
    stub = session / "fake_lmp.sh"
    body = ["#!/bin/sh"]
    body += list(extra_lines)
    body += [
        (
            'printf "HOME=%s TMPDIR=%s KEY=[%s]\\n" "$HOME" "$TMPDIR" '
            '"$FAKE_SECRET_KEY" > "$4"'
        )
    ]
    stub.write_text("\n".join(body) + "\n")
    stub.chmod(0o755)
    return stub


def test_lammps_scrubbed_env_no_key_leak(jail, monkeypatch):
    root, session = jail
    stub = _stub_launcher(session)
    monkeypatch.setenv("ATOMAGENTS_LAMMPS_CMD", str(stub))
    monkeypatch.setenv("FAKE_SECRET_KEY", "canary-9182736450")
    r = json.loads(T.mcp_lammps_runner("units metal\nrun 0\n", str(session), "run.in"))
    assert "canary-9182736450" not in (r.get("output") or "")
    assert str(session) in (r.get("output") or "")  # HOME redirected into jail


def test_lammps_rejects_shell_command(jail):
    root, session = jail
    r = json.loads(T.mcp_lammps_runner("units metal\nshell ls\n", str(session), "run.in"))
    assert "'shell'" in (r.get("error") or "")


def test_lammps_rejects_python_command(jail):
    root, session = jail
    script = "units metal\n  python post here invoke\n"
    r = json.loads(T.mcp_lammps_runner(script, str(session), "run.in"))
    assert "'python'" in (r.get("error") or "")


def test_lammps_allows_shell_word_in_comment(jail, monkeypatch):
    root, session = jail
    stub = _stub_launcher(session)
    monkeypatch.setenv("ATOMAGENTS_LAMMPS_CMD", str(stub))
    script = "units metal\n# shell is only mentioned in this comment\nrun 0\n"
    r = json.loads(T.mcp_lammps_runner(script, str(session), "run.in"))
    assert "Refusing to run this LAMMPS script" not in (r.get("error") or "")


def test_lammps_timeout_env_malformed_is_structured(jail, monkeypatch):
    root, session = jail
    stub = _stub_launcher(session)
    monkeypatch.setenv("ATOMAGENTS_LAMMPS_CMD", str(stub))
    monkeypatch.setenv("ATOMAGENTS_LAMMPS_TIMEOUT", "abc")  # used to raise out
    r = json.loads(T.mcp_lammps_runner("units metal\nrun 0\n", str(session), "run.in"))
    assert r.get("returncode") == 0


# ---------------------------------------------------------------------------
# S1 — seatbelt wrapper units
# ---------------------------------------------------------------------------


def test_sandbox_profile_is_canonical_and_escaped(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    prof = T._sandbox_profile(link)
    assert f'(subpath "{real.resolve()}")' in prof  # canonicalized, not the link
    weird = tmp_path / 'we"ird'
    weird.mkdir()
    assert '\\"' in T._sbpl_path(weird)  # embedded quote escaped


def test_sandbox_profile_has_no_blanket_temp_allowance(jail):
    root, _ = jail
    prof = T._sandbox_profile(root)
    for banned in ('(subpath "/private/tmp")', '(subpath "/private/var/tmp")',
                   '(subpath "/private/var/folders")', '(subpath "/dev")',
                   '(subpath "/tmp")'):
        assert banned not in prof, banned
    assert "(deny file-write*)" in prof


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS-only")
def test_sandbox_refuses_when_unavailable(jail, monkeypatch):
    monkeypatch.setattr(T, "_sandbox_probe", False)
    argv, refusal = T._wrap_sandbox(["/bin/echo", "hi"])
    assert refusal and "sandbox" in refusal


def test_sandbox_disabled_by_env_escape_hatch(jail, monkeypatch):
    monkeypatch.setenv("ATOMAGENTS_SANDBOX_EXEC", "0")
    argv, refusal = T._wrap_sandbox(["/bin/echo", "hi"])
    assert refusal is None
    assert argv == ["/bin/echo", "hi"]


# ---------------------------------------------------------------------------
# S1 — seatbelt end-to-end (the boundary itself)
# ---------------------------------------------------------------------------

needs_seatbelt = pytest.mark.skipif(
    sys.platform != "darwin" or not T._sandbox_available(),
    reason="macOS sandbox-exec required",
)


@needs_seatbelt
def test_sandbox_blocks_write_outside_jail_bash(jail, tmp_path):
    root, session = jail
    victim = tmp_path / "victim"
    victim.mkdir()
    out = json.loads(
        T.mcp_bash_runner(f"echo pwned > {victim}/f.txt", str(session), 30)
    )
    assert out["returncode"] != 0
    assert not (victim / "f.txt").exists()


@needs_seatbelt
def test_sandbox_allows_write_inside_jail_bash(jail):
    root, session = jail
    out = json.loads(T.mcp_bash_runner(f"echo ok > {session}/in.txt", str(session), 30))
    assert out["returncode"] == 0
    assert (session / "in.txt").read_text().strip() == "ok"


@needs_seatbelt
def test_sandbox_blocks_write_outside_jail_code_runner(jail, tmp_path):
    root, session = jail
    victim = tmp_path / "victim"
    victim.mkdir()
    code = (
        "try:\n"
        f"    open({str(victim / 'f.txt')!r}, 'w').write('pwned')\n"
        "    print('WROTE')\n"
        "except Exception as e:\n"
        "    print('BLOCKED', type(e).__name__)\n"
    )
    out = json.loads(T.mcp_code_runner(code, str(session), 30))
    assert "WROTE" not in out["stdout"]
    assert "BLOCKED" in out["stdout"]
    assert not (victim / "f.txt").exists()


@needs_seatbelt
def test_sandbox_git_incident_repro_is_blocked(jail, tmp_path, monkeypatch):
    """The recorded incident, replayed: `git -C <outside>` walks out of the
    jailed cwd and `-c user.email=` re-supplies the identity the env scrub
    removed. The denylist is deliberately disabled here so this asserts the
    SEATBELT, not the guardrail in front of it."""
    root, session = jail
    victim = tmp_path / "victim"
    victim.mkdir()
    subprocess.run(["git", "init", "-q", str(victim)], check=True)
    monkeypatch.setenv("ATOMAGENTS_ALLOW_GIT", "1")
    cmd = (
        f"git -C {victim} -c user.email=a@b -c user.name=a "
        "commit --allow-empty -m pwned"
    )
    out = json.loads(T.mcp_bash_runner(cmd, str(session), 30))
    assert out["returncode"] != 0
    count = subprocess.run(
        ["git", "-C", str(victim), "rev-list", "--count", "--all"],
        capture_output=True, text=True,
    )
    assert count.stdout.strip() in ("", "0")


# ---------------------------------------------------------------------------
# S9 — the time budget must catch backgrounded children too
# ---------------------------------------------------------------------------


def test_bash_timeout_kills_process_group(jail):
    root, session = jail
    marker = session / "orphan.marker"
    # The orphan outlives the direct child; only a process-GROUP kill stops it.
    cmd = f"(sleep 30; touch {marker}) & sleep 30"
    out = json.loads(T.mcp_bash_runner(cmd, str(session), 2))
    assert out["returncode"] == -1
    assert "timed out" in out["stderr"]
    survivors = subprocess.run(
        ["pgrep", "-f", f"touch {marker}"], capture_output=True, text=True
    )
    assert survivors.stdout.strip() == "", survivors.stdout


# ---------------------------------------------------------------------------
# Meta — no write/exec tool may ship unguarded
# ---------------------------------------------------------------------------

_WRITE_EXEC_PARAMS = {"cwd", "path", "dest", "save_file", "save_dir", "work_dir"}


def _is_mcp_tool(node: ast.FunctionDef) -> bool:
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute) and target.attr == "tool":
            return True
    return False


def test_every_write_exec_tool_is_guarded():
    """Every @mcp.tool() taking a write/exec path must call the guard — in
    its own body or in the tools.py function it delegates to. Guards drifted
    apart historically (some live in the wrappers, some in tools.py), so the
    next tool added is one omission away from unguarded.

    The read-mode param `file` is excluded on purpose: reads are deliberately
    unjailed, and md_visualize's unguarded input `file` is the documented S12
    gap, not a regression this test should mask.
    """
    svc_dir = Path(T.__file__).parent
    tools_tree = ast.parse((svc_dir / "tools.py").read_text())
    svc_tree = ast.parse((svc_dir / "mcp_service.py").read_text())

    def _guards(node) -> bool:
        return any(
            isinstance(c.func, ast.Name) and c.func.id == "_guard_workspace_path"
            for c in ast.walk(node)
            if isinstance(c, ast.Call)
        )

    guarded_helpers = {
        n.name for n in ast.walk(tools_tree)
        if isinstance(n, ast.FunctionDef) and _guards(n)
    }

    exposed = [
        n for n in ast.walk(svc_tree)
        if isinstance(n, ast.FunctionDef) and _is_mcp_tool(n)
    ]
    assert exposed, "no @mcp.tool() functions found — did the surface move?"

    unguarded = []
    for node in exposed:
        if not ({a.arg for a in node.args.args} & _WRITE_EXEC_PARAMS):
            continue
        callees = {
            c.func.id for c in ast.walk(node)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        }
        if not _guards(node) and not (callees & guarded_helpers):
            unguarded.append(node.name)
    assert not unguarded, f"write/exec tools missing a path guard: {unguarded}"
