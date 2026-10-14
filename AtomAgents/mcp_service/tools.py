from typing import List, Dict, Tuple
from json import dumps
import ast
import logging
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

_logger = logging.getLogger("AtomAgents.mcp_service.tools")


# --- Shared helpers ----------------------------------------------------
# Cap applied to every text payload returned to an LLM (env-overridable so the
# service stays self-contained when deployed without the AtomAgents package).
TOOL_OUTPUT_MAX = int(os.environ.get("ATOMAGENTS_TOOL_OUTPUT_MAX", 64 * 1024))

# Paths a tool must never write to or create. Guards against the observed
# failure where an empty project variable expanded to filesystem root
# (`mkdir /input`).
# NOTE: paths are checked AFTER .resolve() — on macOS /etc,/tmp,/var are
# symlinks into /private/*, so the private twins must be listed too (a test
# caught /etc/passwd resolving to /private/etc/passwd and sailing through).
_FORBIDDEN_PREFIXES = (
    "/bin",
    "/sbin",
    "/usr",
    "/etc",
    "/System",
    "/Library",
    "/var/db",
    "/private/etc",
    "/private/var/db",
)

_warned_no_root = False
_warned_bad_root = False


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _tool_root() -> Path | None:
    """The write/exec jail root (env ATOMAGENTS_TOOL_ROOT), or None.

    Read per call so tests and long-lived services pick up env changes. When
    unset (or pointing at a non-directory) the guard FAILS CLOSED unless
    ATOMAGENTS_ALLOW_UNJAILED=1 opts back into the legacy denylist-only
    behaviour — a cwd-less bash_run once landed in this repo and committed
    to git, which is the incident this jail exists to prevent.
    """
    global _warned_no_root, _warned_bad_root
    root = os.environ.get("ATOMAGENTS_TOOL_ROOT", "").strip()
    if not root:
        if not _warned_no_root:
            _logger.warning(
                "ATOMAGENTS_TOOL_ROOT is not set — exec/write tools are NOT "
                "jailed to a session root (legacy denylist only). Set it to "
                "the jobs directory in production deployments."
            )
            _warned_no_root = True
        return None
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        if not _warned_bad_root:
            _logger.warning(
                "ATOMAGENTS_TOOL_ROOT=%s is not an existing directory — "
                "treating the jail as unset.",
                root_path,
            )
            _warned_bad_root = True
        return None
    return root_path


def _truncate_text(text: str, limit: int = TOOL_OUTPUT_MAX) -> str:
    """Head+tail window so both the start and the end of long output survive."""
    if text is None or len(text) <= limit:
        return text
    head = text[: int(limit * 0.75)]
    tail = text[-int(limit * 0.25) :]
    dropped = len(text) - len(head) - len(tail)
    return f"{head}\n... [{dropped} chars truncated] ...\n{tail}"


def _tail_lines(text: str, n: int = 200, limit: int = TOOL_OUTPUT_MAX) -> str:
    tail = "\n".join(text.splitlines()[-n:])
    if len(tail) > limit:
        tail = tail[-limit:]
    return tail


def _guard_workspace_path(
    path: str, must_be_absolute: bool = True, jail: bool = True
) -> str | None:
    """Return an error message if *path* is unsafe to operate under, else None.

    ``jail=True`` (write/exec call sites) additionally requires the resolved
    path to live under ATOMAGENTS_TOOL_ROOT when that env var is set.
    ``jail=False`` (read call sites) applies only the system-dir denylist —
    reads of e.g. downloaded potential files outside the session are allowed
    by design.
    """
    if not path or not str(path).strip():
        return "Empty path. Pass an absolute path inside the job workspace."
    p = Path(path)
    if must_be_absolute and not p.is_absolute():
        return f"Relative path {path!r}. Pass an absolute path inside the job workspace."
    resolved = p.resolve()
    if resolved == Path("/") or len(resolved.parts) <= 2:
        return (
            f"Refusing to operate directly under filesystem root: {resolved}. "
            "This usually means a project-root variable was empty."
        )
    for prefix in _FORBIDDEN_PREFIXES:
        if str(resolved).startswith(prefix):
            return f"Refusing to write into system directory: {resolved}"
    if jail:
        root = _tool_root()
        if root is None:
            if not _env_flag("ATOMAGENTS_ALLOW_UNJAILED"):
                return (
                    "Refusing to write or execute without a workspace jail: "
                    "ATOMAGENTS_TOOL_ROOT is unset or not a directory, so "
                    f"{resolved} cannot be verified to live inside the job "
                    "workspace. Start the MCP service with ATOMAGENTS_TOOL_ROOT "
                    "set (`make mcp`, or see README 'Security (tool sandbox)'), "
                    "or set ATOMAGENTS_ALLOW_UNJAILED=1 to accept the legacy "
                    "denylist-only behaviour."
                )
        elif not resolved.is_relative_to(root):
            return (
                f"Refusing to operate outside the session root {root}: "
                f"{resolved}. All writes and executions must stay under the "
                "job workspace."
            )
    return None


# Keys preserved in scrubbed child environments; everything else — including
# every API key and the real HOME — is dropped.
_ENV_KEEP = ("PATH", "LANG", "LC_ALL", "TERM")


def _scrubbed_env(cwd: str) -> dict:
    """Minimal child env for exec tools.

    HOME/TMPDIR are redirected INTO the session dir so ``~/.gitconfig``
    identity and user dotfiles can never resolve in a subprocess, and no
    service API keys leak into agent-authored code.
    """
    env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
    fake_home = Path(cwd) / ".home"
    try:
        fake_home.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    env["HOME"] = str(fake_home)
    env["TMPDIR"] = str(fake_home)
    # The fake HOME is inside the WRITABLE jail, so a planted
    # .home/.gitconfig would restore a git identity — pin git config to
    # /dev/null so no config file resolves at all.
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    env["GIT_CONFIG_SYSTEM"] = "/dev/null"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


# Dangerous-command denylist for bash_run and code_runner. Token-based:
# every line is scanned, quotes/backslashes are stripped, path prefixes are
# dropped and comparison is case-folded (macOS filesystems are
# case-insensitive — `GIT` runs git), so `/usr/bin/git`, `env git`,
# post-newline commands, `g"i"t` and backtick forms are all caught while
# `legit_tool --sudo-mode` and filenames like `.gitignore` still pass.
# This is a guardrail, not the boundary — obfuscated evasion (base64 piped
# to a shell) is out of scope; the seatbelt profile is the boundary.
_DENY_BASENAMES = frozenset(
    {"sudo", "doas", "su", "shutdown", "reboot", "halt", "poweroff",
     "ssh", "scp", "sftp"}
)
_DENY_PREFIXES = ("mkfs",)  # mkfs, mkfs.ext4, ...
_TOKEN_SPLIT = re.compile(r"[\s;&|()<>{}`\n\r\t]+")
_QUOTE_SPLIT = re.compile(r"['\"`\\]+")
_QUOTE_CHARS = "'\"`\\"
# Belt-and-braces for the classic root-rm form; MULTILINE anchors every line.
_RM_ROOT = re.compile(
    r"(?:^|[;&|(]\s*|\$\(\s*)rm\s+(?:-[a-zA-Z]+\s+|--[a-z-]+\s+)*/(?:\s|$|\*)",
    re.MULTILINE,
)


def _command_tokens(text: str) -> list[str]:
    """Raw tokens: split on whitespace and shell metacharacters (all lines)."""
    return [t for t in _TOKEN_SPLIT.split(text or "") if t]


def _command_words(text: str) -> set[str]:
    """Case-folded basenames of every plausible command word in *text*.

    Two passes per token so both obfuscations are covered:
      (a) quote/backslash characters REMOVED inside the token -> g"i"t -> git
      (b) token SPLIT on quote characters                     -> echo"git -> git
    """
    words: set[str] = set()
    for raw in _command_tokens(text):
        pieces = ["".join(c for c in raw if c not in _QUOTE_CHARS)]
        pieces += [p for p in _QUOTE_SPLIT.split(raw) if p]
        for piece in pieces:
            piece = piece.strip()
            if piece:
                words.add(piece.rsplit("/", 1)[-1].casefold())
    return words


def _deny_rm(text: str) -> bool:
    """rm targeting the filesystem root or a top-level directory."""
    if _RM_ROOT.search(text):
        return True
    tokens = _command_tokens(text)
    for i, tok in enumerate(tokens):
        base = "".join(c for c in tok if c not in _QUOTE_CHARS).rsplit("/", 1)[-1]
        if base.casefold() != "rm":
            continue
        for arg in tokens[i + 1:]:
            a = "".join(c for c in arg if c not in _QUOTE_CHARS)
            if a.startswith("-"):
                continue  # flag, keep scanning
            if a in ("/", "/*") or (
                a.startswith("/") and len(Path(a.rstrip("*")).parts) <= 2
            ):
                return True
            break  # first positional arg decides
    return False


def _deny_command(command: str) -> str | None:
    """Return an error message when *command* matches the denylist."""
    words = _command_words(command)
    if not _env_flag("ATOMAGENTS_ALLOW_GIT") and "git" in words:
        return (
            "git commands are disabled for agent shells (the service "
            "operator can set ATOMAGENTS_ALLOW_GIT=1 to permit them)."
        )
    if words & _DENY_BASENAMES or any(
        w.startswith(_DENY_PREFIXES) for w in words
    ):
        return "Refusing command: it matches the dangerous-command denylist."
    if _deny_rm(command):
        return "Refusing command: it matches the dangerous-command denylist."
    return None


# --- Python-source scan (code_runner) ----------------------------------
# The shell denylist above reads its input as a SHELL COMMAND. Applying it to
# Python source refused ordinary science code — `Su = 250e6  # ultimate
# strength`, `halt = False`, a "shutdown" in a comment (S16: 4 of 91 repo .py
# files were rejected). Python is therefore parsed, and only the command
# strings the script would hand to a shell/exec API are denylisted.
_PY_EXEC_CALLS = frozenset(
    {"system", "popen", "run", "call", "check_call", "check_output",
     "getoutput", "getstatusoutput", "Popen", "spawn", "spawnl", "spawnv",
     "spawnlp", "spawnvp", "execl", "execlp", "execle", "execv", "execvp",
     "execvpe", "startfile"}
)
_PY_EXEC_MODULES = frozenset({"os", "subprocess", "pty", "commands", "sh"})
# Attribute names specific enough to flag on their own (`run`/`call` are far
# too common as method names to match without a known module root).
_PY_EXEC_BARE = frozenset({"system", "popen", "Popen"})
# argv[0] values whose REMAINING elements are themselves command text and must
# be scanned in full (`["bash", "-c", "git push"]`, `["env", "git", "st"]`).
_PY_SHELL_WRAPPERS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "env"})


def _import_bindings(tree: ast.AST) -> tuple[dict, dict]:
    """(alias -> top-level module, bare name -> (module, attr)) import maps.

    `import subprocess as sp` binds sp->subprocess; `from subprocess import
    run as r` binds r->(subprocess, run) — without these, aliased and
    from-imported exec calls bypassed the scan entirely. Dynamic forms
    (importlib, __import__, getattr) stay out of scope — this is a guardrail,
    not the boundary (the seatbelt profile is).
    """
    alias_map: dict[str, str] = {}
    from_map: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                alias_map[a.asname or top] = top
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            for a in node.names:
                from_map[a.asname or a.name] = (top, a.name)
    return alias_map, from_map


def _callee_root_and_attr(func: ast.AST) -> tuple[str | None, str | None]:
    """(root name, attribute) of a call target: subprocess.run -> (subprocess, run)."""
    if isinstance(func, ast.Name):
        return None, func.id
    if isinstance(func, ast.Attribute):
        node = func.value
        while isinstance(node, ast.Attribute):
            node = node.value
        root = node.id if isinstance(node, ast.Name) else None
        return root, func.attr
    return None, None


def _shell_scan_texts(node: ast.AST) -> list[str]:
    """Denylist-scannable text of a literal command argument.

    A string (or f-string constant parts) argument is shell text and is
    scanned whole. For a list/tuple argv only argv[0] is *executed*, so only
    argv[0] is scanned — `["grep", "halt", "log"]` must not trip on the
    data word "halt" (the S16 false-positive class) — EXCEPT when argv[0] is
    a shell/wrapper (sh/bash/env/...), whose remaining elements are command
    text and are scanned in full. Non-literal pieces are skipped — this is a
    guardrail, not the boundary (the seatbelt profile is).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        parts = [
            v.value for v in node.values
            if isinstance(v, ast.Constant) and isinstance(v.value, str)
        ]
        return [" ".join(parts)] if parts else []
    if isinstance(node, (ast.List, ast.Tuple)):
        if not node.elts:
            return []
        head = node.elts[0]
        if not (isinstance(head, ast.Constant) and isinstance(head.value, str)):
            return []  # dynamic executable — out of scanning scope
        argv0 = head.value
        if argv0.rsplit("/", 1)[-1].casefold() in _PY_SHELL_WRAPPERS:
            rest = [
                e.value for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
            return [" ".join(rest)]
        return [argv0]
    return []


def _python_shell_commands(code: str) -> list[str]:
    """Command strings *code* would hand to a shell/exec API.

    Parses the source (resolving import aliases and from-imports) and
    inspects only os/subprocess/pty exec call sites, so the shell denylist
    never sees ordinary Python identifiers, strings or comments (S16).
    Uncompilable source returns [] — SyntaxError/ValueError, and the
    RecursionError/MemoryError CPython's compiler raises on pathological
    nesting, all mean the interpreter would reject it too.
    """
    try:
        tree = ast.parse(code or "")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    alias_map, from_map = _import_bindings(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        root, attr = _callee_root_and_attr(node.func)
        if attr is None:
            continue
        if isinstance(node.func, ast.Name):
            # bare call: from-imported exec API (`from subprocess import
            # run`), or the always-suspicious system/popen/Popen names.
            mod, orig = from_map.get(attr, (None, None))
            if not (
                (mod in _PY_EXEC_MODULES and orig in _PY_EXEC_CALLS)
                or attr in _PY_EXEC_BARE
            ):
                continue
        else:
            root = alias_map.get(root, root)
            if not (
                attr in _PY_EXEC_CALLS
                and (root in _PY_EXEC_MODULES or attr in _PY_EXEC_BARE)
            ):
                continue
        # The command is the first positional argument (or args=<literal>).
        targets = list(node.args[:1]) or [
            kw.value for kw in node.keywords if kw.arg == "args"
        ]
        for arg in targets:
            found.extend(t for t in _shell_scan_texts(arg) if t)
    return found


def _deny_python_code(code: str) -> str | None:
    """Denylist verdict for a Python script submitted to code_runner.

    Applies the shell denylist ONLY to the commands the script would shell
    out with. Before S16 the whole source was token-scanned, so domain code
    (`Su = 250e6`, `halt = False`) was refused while the actual containment
    boundary — the seatbelt profile — was unaffected either way.
    """
    for cmd in _python_shell_commands(code):
        denied = _deny_command(cmd)
        if denied:
            return denied
    return None


def _clamp_timeout(requested, env_var: str, default_ceiling: int) -> int:
    """Server-side timeout ceiling (env-controlled, never LLM-controlled —
    the lammps_run precedent)."""
    try:
        ceiling = int(os.environ.get(env_var, default_ceiling))
    except (TypeError, ValueError):
        ceiling = default_ceiling
    try:
        req = int(requested) if requested else ceiling
    except (TypeError, ValueError):
        req = ceiling
    return max(1, min(req, ceiling))


# --- Child-process sandbox (macOS seatbelt) ----------------------------
# The path guards above validate tool ARGUMENTS; they place no boundary on
# what a spawned child may touch (`git -C /anywhere` walks straight out of a
# jailed cwd). On macOS every exec-tool child is therefore wrapped in
# sandbox-exec with a profile that denies filesystem writes outside the jail.
# The kernel matches canonical paths (symlink tricks fail), confinement is
# inherited by grandchildren (mpirun -> lmp_mpi), and a sandboxed child
# cannot re-invoke sandbox-exec to widen its own profile. Reads and network
# stay open by design (potential files, doc queries live outside sessions).
_SANDBOX_EXEC = "/usr/bin/sandbox-exec"
_sandbox_probe: bool | None = None
_warned_no_sandbox = False


def _sbpl_path(p) -> str:
    """SBPL string literal for a canonical, escaped path."""
    s = str(Path(p).resolve()).rstrip("/") or "/"
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _sandbox_profile(root: Path) -> str:
    """Seatbelt profile: everything allowed EXCEPT writes outside *root*.

    /dev entries are enumerated (not `(subpath "/dev")`) so device nodes are
    unreachable, and there is deliberately NO blanket /tmp, /var/tmp or
    /var/folders allowance — HOME and TMPDIR are already redirected into the
    jail by _scrubbed_env, and mpirun/lmp_mpi/matplotlib were verified to run
    with only the jail writable. ATOMAGENTS_SANDBOX_EXTRA_WRITE (colon-
    separated paths) adds extra writable roots if a tool ever needs one.
    """
    extra = [
        x
        for x in os.environ.get("ATOMAGENTS_SANDBOX_EXTRA_WRITE", "").split(":")
        if x.strip()
    ]
    subpaths = "\n    ".join(f"(subpath {_sbpl_path(p)})" for p in [root, *extra])
    return f"""(version 1)
(allow default)
(deny file-write*)
(allow file-write*
    {subpaths}
    (subpath "/dev/fd")
    (literal "/dev/null") (literal "/dev/zero")
    (literal "/dev/random") (literal "/dev/urandom")
    (literal "/dev/stdout") (literal "/dev/stderr") (literal "/dev/tty"))
"""


def _sandbox_available() -> bool:
    """One-shot capability probe (`sandbox-exec -h` exits 0 even when unusable)."""
    global _sandbox_probe
    if _sandbox_probe is None:
        try:
            _sandbox_probe = (
                os.path.exists(_SANDBOX_EXEC)
                and subprocess.run(
                    [_SANDBOX_EXEC, "-p", "(version 1)(allow default)", "/usr/bin/true"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                ).returncode
                == 0
            )
        except Exception:
            _sandbox_probe = False
    return bool(_sandbox_probe)


def _wrap_sandbox(argv: list[str]) -> tuple[list[str], str | None]:
    """Return (argv_to_spawn, refusal). A non-None refusal means DO NOT spawn."""
    global _warned_no_sandbox
    mode = os.environ.get("ATOMAGENTS_SANDBOX_EXEC", "auto").strip().lower()
    if mode in ("0", "off", "false", "no"):
        return argv, None
    if sys.platform != "darwin":
        if not _warned_no_sandbox:
            _logger.warning(
                "sandbox-exec is macOS-only; child processes are NOT "
                "write-confined here (the container is the boundary)."
            )
            _warned_no_sandbox = True
        return argv, None
    root = _tool_root()
    if root is None:  # only reachable with ATOMAGENTS_ALLOW_UNJAILED=1
        if not _warned_no_sandbox:
            _logger.warning(
                "No ATOMAGENTS_TOOL_ROOT: nothing to confine children to."
            )
            _warned_no_sandbox = True
        return argv, None
    if len(root.parts) <= 2:
        return argv, (
            f"Refusing to execute: ATOMAGENTS_TOOL_ROOT={root} is too "
            "shallow to be a meaningful sandbox boundary."
        )
    if not _sandbox_available():
        return argv, (
            "Refusing to execute: macOS sandbox (/usr/bin/sandbox-exec) is "
            "unavailable, so child processes cannot be confined to "
            f"{root}. Set ATOMAGENTS_SANDBOX_EXEC=0 to run unconfined."
        )
    return [_SANDBOX_EXEC, "-p", _sandbox_profile(root), *argv], None


def _sandbox_status() -> str:
    """One-line sandbox posture for the startup banner."""
    mode = os.environ.get("ATOMAGENTS_SANDBOX_EXEC", "auto").strip().lower()
    if mode in ("0", "off", "false", "no"):
        return "off (ATOMAGENTS_SANDBOX_EXEC=0)"
    if sys.platform != "darwin":
        return "off (non-darwin — the container is the boundary)"
    root = _tool_root()
    if root is None:
        return "off (no jail root)"
    if _sandbox_available():
        return f"sandbox-exec (writes confined to {root})"
    return (
        "UNAVAILABLE — exec tools will refuse to spawn "
        "(set ATOMAGENTS_SANDBOX_EXEC=0 to run unconfined)"
    )


def _as_text(v) -> str:
    if not v:
        return ""
    return v if isinstance(v, str) else v.decode("utf-8", errors="replace")


def _run_child(argv, *, cwd, env, timeout, text=True):
    """Spawn in its own process group; on timeout kill the GROUP (a plain
    subprocess.run kills only the direct child, so `nohup x &` orphans
    survive), then re-raise TimeoutExpired carrying whatever output exists."""
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                out, err = proc.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                out, err = proc.communicate()
        except (ProcessLookupError, PermissionError):
            out, err = proc.communicate()
        raise subprocess.TimeoutExpired(argv, timeout, output=out, stderr=err)
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


# --- Render MD structure visual using OVITO; returns base64 PNG ---
def mcp_render_md_model(
    file: str,
    orientation: Tuple[float, float, float],
    cam_pos: Tuple[float, float, float],
    save_dir: str,
) -> str:
    """
    Agent Tool: Render Molecular Model (OVITO, returns PNG as base64)
    -----------------------------------------------------------------
    Takes a molecular structure file and renders it as a PNG using OVITO.
    Args:
        file (bytes): Molecular structure file (.xyz, .cfg, .lmp, etc)
        orientation (Tuple[float, float, float]): Camera direction vector (OVITO viewport direction) for example (-1,-1,-1).
        cam_pos (Tuple[float, float, float]): Camera position (OVITO viewport position) for example (1000,1000,1000).
        save_dir (str): Directory to save the rendered image png format
    Returns:
        str: base64 PNG of the rendered image, or error string if fails.
    Requirements: ovito (pip install ovito), working OpenGL context.
    """
    try:
        import base64
        from ovito.io import import_file
        from ovito.vis import Viewport
        from ovito.vis import ParticlesVis
        import tempfile
        from PySide6.QtCore import QBuffer, QByteArray, QIODevice
        from pathlib import Path

        with open(file, "rb") as f:
            # Read the file into OVITO
            f.seek(0)
            file_ = f.read()
        # Write file to a temp on disk for OVITO. Keep the original suffix —
        # OVITO infers the file format from the extension.
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file).suffix or ".xyz"
        ) as f:
            f.write(file_)
            fname = f.name

        pipeline = import_file(fname)
        # Style selection
        pipeline.compute().particles_.vis = ParticlesVis(
            shape=ParticlesVis.Shape.Sphere
        )
        # Add to global scene but keep a handle so we can remove it later
        try:
            node = pipeline.add_to_scene()
        except Exception:
            node = None
        # Add more styles as wanted...
        print("number of particles:", pipeline.compute().particles_.count)
        vp = Viewport()
        vp.type = Viewport.Type.Perspective
        vp.camera_pos = cam_pos  # e.g., (0, 0, 10) for a top-down view
        vp.camera_dir = orientation
        vp.zoom_all()

        img = vp.render_image(size=(512, 512), background=(1, 1, 1))

        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.WriteOnly)

        # Save the image to the buffer in PNG format (you can use other formats like "JPG")
        img.save(buffer, "PNG")
        # also save to a file if needed
        save_path = Path(save_dir)
        # get the parent directory
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        # Save the image to the specified directory
        img.save(save_dir, "PNG")
        print(f"Rendered image saved to: {save_dir}")
        image_bytes = byte_array.data()
        # Convert to base64
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        print(f"Rendered image size: {len(img_b64)} bytes")
        os.remove(fname)
        number_of_particles = pipeline.compute().particles_.count
        # Clean up scene to avoid stacking/overlap in subsequent renders
        try:
            if node is not None:
                # Prefer removing the specific node we added
                try:
                    node.remove_from_scene()  # OVITO >= 3 API
                except Exception:
                    try:
                        node.remove()  # Fallback API name in some versions
                    except Exception:
                        pass
            else:
                # If we couldn't capture a node, attempt to remove via pipeline
                try:
                    pipeline.remove_from_scene()
                except Exception:
                    pass
            # As a last resort, try clearing the whole scene (safe in this service context)
            try:
                from ovito import scene as _scene  # type: ignore

                _scene.clear()
            except Exception:
                pass
        finally:
            # Ensure local objects are released
            del pipeline
            del vp
        return True, img_b64, number_of_particles
    except Exception as e:
        print(f"Error rendering model: {e}")
        try:
            if "fname" in locals():
                os.remove(fname)
        except OSError:
            pass
        return False, f"ovito_render error: {e}", 0


"""
MCP Tools: Tool functions for the AtomAgents MCP Service.
Organize code execution, web search, file reading, etc. here, ready for FastAPI endpoints.
"""
from typing import Any
import tempfile
import shutil
from base64 import b64encode


# --- file_reader tool ---
def mcp_file_reader(file: str, offset: int = 0, limit_bytes: int = 0) -> str:
    """
    Agent Tool: File Reader.
    -----------------------------------------------
    Reads any text file (detected by content sniffing, not extension — .data,
    .in, .xdmf, .lammpstrj, etc. all work) plus PNG/JPG/PDF as multimodal parts.
    Large text files are windowed head+tail; use *offset*/*limit_bytes* to page
    through a specific region.
    Args:
        file (str): file path
        offset (int): byte offset to start reading text content from (default 0)
        limit_bytes (int): max bytes of text to return (default: tool cap)
    Returns:
        (success, str): JSON-encoded content parts or error.
    """
    # READ-mode guard: system-dir denylist only, deliberately NOT jailed —
    # runs legitimately read potential/data files outside the session dir.
    guard = _guard_workspace_path(file, must_be_absolute=False, jail=False)
    if guard:
        return False, dumps({"error": guard})
    path = Path(file)
    if not path.exists():
        return False, dumps({"error": f"File not found: {file}"})
    if path.is_dir():
        try:
            listing = "\n".join(sorted(p.name for p in path.iterdir())[:500])
            return True, dumps(
                [{"type": "text", "text": f"Directory listing of {file}:\n{listing}"}]
            )
        except Exception as e:
            return False, dumps({"error": f"file_reader error: {e}"})

    suffix = path.suffix.lower()
    # Base64 payload cap: 3/4 of the encoded budget (env-overridable). The
    # image/PDF branches were previously unbounded.
    max_binary = int(os.environ.get("ATOMAGENTS_TOOL_BINARY_MAX", 8 * 1024 * 1024))
    try:
        if suffix in (".png", ".jpg", ".jpeg"):
            size = path.stat().st_size
            if size > max_binary:
                return False, dumps(
                    {
                        "error": (
                            f"Image too large to inline ({size} bytes > "
                            f"{max_binary}). Downscale it first."
                        )
                    }
                )
            with open(file, "rb") as f:
                data = f.read()
            mime = "image/png" if suffix == ".png" else "image/jpeg"
            base64_string = b64encode(data).decode("utf-8")
            parts = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{base64_string}"},
                }
            ]
            return True, dumps(parts)
        if suffix == ".pdf":
            size = path.stat().st_size
            if size > max_binary:
                return False, dumps(
                    {
                        "error": (
                            f"PDF too large to inline ({size} bytes > "
                            f"{max_binary}). Extract the relevant pages first."
                        )
                    }
                )
            with open(file, "rb") as f:
                data = f.read()
            base64_string = b64encode(data).decode("utf-8")
            parts = [
                {
                    "type": "file",
                    "file": {
                        "filename": file,
                        "file_data": f"data:application/pdf;base64,{base64_string}",
                    },
                }
            ]
            return True, dumps(parts)

        # Everything else: sniff whether it is text.
        size = path.stat().st_size
        with open(file, "rb") as f:
            probe = f.read(8192)
        if b"\x00" in probe:
            return False, dumps(
                {
                    "error": (
                        f"Binary file ({size} bytes): {file}. "
                        "Use a format-specific tool or a python script to inspect it."
                    )
                }
            )
        cap = min(limit_bytes, TOOL_OUTPUT_MAX) if limit_bytes else TOOL_OUTPUT_MAX
        with open(file, "r", encoding="utf-8", errors="replace") as f:
            if offset:
                f.seek(offset)
                content = f.read(cap)
                notice = f"[window: bytes {offset}-{offset + len(content)} of {size}]\n"
                text = notice + content
            else:
                content = f.read()
                text = _truncate_text(content, cap)
                if len(content) > cap:
                    text = (
                        f"[file is {size} bytes; showing head+tail window — "
                        f"re-call with offset/limit_bytes for other regions]\n" + text
                    )
        return True, dumps([{"type": "text", "text": text}])
    except Exception as e:
        return False, dumps({"error": f"file_reader error: {e}"})


# --- write_file tool ---
def mcp_write_file(path: str, content: str, mode: str = "w") -> str:
    """
    Agent Tool: Write File.
    -----------------------------------------------
    Write *content* to *path* verbatim (UTF-8). This is the reliable way to
    author scripts/inputs/configs — no shell quoting or heredoc pitfalls.
    Args:
        path (str): absolute destination path inside the job workspace
        content (str): exact file content
        mode (str): "w" (overwrite, default) or "a" (append)
    Returns:
        JSON string with 'path', 'bytes_written' or 'error'.
    """
    guard = _guard_workspace_path(path)
    if guard:
        return dumps({"error": guard})
    if mode not in ("w", "a"):
        return dumps({"error": f"Unsupported mode {mode!r}; use 'w' or 'a'."})
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, mode, encoding="utf-8") as f:
            f.write(content)
        return dumps({"path": str(p), "bytes_written": len(content.encode("utf-8"))})
    except Exception as e:
        return dumps({"error": f"write_file error: {e}"})


# --- download_file tool ---
def mcp_download_file(url: str, dest: str, expect: str = "auto") -> str:
    """
    Agent Tool: Download File (validated).
    -----------------------------------------------
    Download *url* to *dest* and validate the payload so an HTML error/landing
    page is never silently saved as a data file (observed failure: an HTML page
    stored as an EAM potential).
    Args:
        url (str): http(s) URL
        dest (str): absolute destination path
        expect (str): "text", "binary", or "auto"
    Returns:
        JSON string with 'path', 'bytes', 'sha256', 'content_type' or 'error'.
    """
    import hashlib
    import urllib.request

    guard = _guard_workspace_path(dest)
    if guard:
        return dumps({"error": guard})
    if not url.startswith(("http://", "https://")):
        return dumps({"error": f"Unsupported URL scheme: {url}"})
    MAX_DOWNLOAD = 500 * 1024 * 1024
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AtomAgents/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read(MAX_DOWNLOAD + 1)
        if len(data) > MAX_DOWNLOAD:
            return dumps({"error": f"Download exceeds {MAX_DOWNLOAD} bytes: {url}"})
        head = data[:2048].lstrip().lower()
        looks_html = head.startswith((b"<!doctype", b"<html")) or b"<head>" in head
        if looks_html and not url.endswith((".html", ".htm")):
            return dumps(
                {
                    "error": (
                        f"Server returned an HTML page ({len(data)} bytes, "
                        f"Content-Type: {content_type!r}) instead of a data file. "
                        "This is usually a landing/error page — extract the real "
                        "download link from it first."
                    ),
                    "preview": data[:1000].decode("utf-8", errors="replace"),
                }
            )
        if expect == "text" and b"\x00" in data[:8192]:
            return dumps({"error": "Expected text but payload is binary."})
        p = Path(dest)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
        return dumps(
            {
                "path": str(p),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "content_type": content_type,
            }
        )
    except Exception as e:
        return dumps({"error": f"download_file error: {e}"})


# --- Code execution tool: safe sandbox for code (for agent use) ---


def mcp_code_runner(code: str, cwd: str | None = None, timeout_sec: int = 300) -> Dict[str, Any]:
    """
    Agent Tool: Python Code Executor (sandboxed)
    --------------------------------------------
    Execute a Python script (as a string) in a subprocess. `cwd` is REQUIRED
    and must be an absolute path inside the session workspace; the child runs
    with a scrubbed minimal environment (no API keys, HOME redirected into
    the workspace). Network access remains available — write scripts
    accordingly. Output streams are truncated. Only shell-escape call sites
    (os/subprocess exec APIs) are denylist-scanned, not the whole source —
    ordinary identifiers and comments are never treated as shell commands.
    Args:
        code (str): Python code (a script as string)
        cwd (str): REQUIRED absolute working directory (the job workspace)
        timeout_sec (int): max seconds before the run is killed (server-side ceiling applies)
    Returns:
        JSON string with keys 'stdout', 'stderr', 'returncode'.
    """

    if not cwd:
        return dumps(
            {
                "stdout": "",
                "stderr": (
                    "cwd is required: pass the absolute session workspace "
                    "directory to run scripts in."
                ),
                "returncode": -1,
            }
        )
    guard = _guard_workspace_path(cwd)
    if guard:
        return dumps({"stdout": "", "stderr": guard, "returncode": -1})
    denied = _deny_python_code(code)
    if denied:
        return dumps({"stdout": "", "stderr": denied, "returncode": -1})

    timeout_sec = _clamp_timeout(timeout_sec, "ATOMAGENTS_CODE_TIMEOUT_MAX", 1800)
    env = _scrubbed_env(cwd)

    # The script lives INSIDE the session workspace (not the system temp
    # dir), so everything the run produces stays in the jail.
    script_dir = Path(cwd) / ".scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".py", delete=True, dir=str(script_dir)
    ) as tmp_file:
        tmp_file.write(code)
        tmp_file.flush()
        argv, refusal = _wrap_sandbox([sys.executable, tmp_file.name])
        if refusal:
            return dumps({"stdout": "", "stderr": refusal, "returncode": -1})
        try:
            result = _run_child(argv, cwd=cwd, env=env, timeout=timeout_sec)
            return dumps(
                {
                    "stdout": _truncate_text(result.stdout),
                    "stderr": _truncate_text(result.stderr),
                    "returncode": result.returncode,
                }
            )
        except subprocess.TimeoutExpired as e:
            return dumps(
                {
                    "stdout": _truncate_text(_as_text(e.stdout)),
                    "stderr": f"Execution timed out after {timeout_sec} seconds.",
                    "returncode": -1,
                }
            )
        except Exception as e:
            return dumps(
                {
                    "stdout": "",
                    "stderr": f"Execution error: {str(e)}",
                    "returncode": -1,
                }
            )


# --- Web search tool (Tavily) ---
def mcp_web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Agent Tool: Web Search (Tavily)
    ------------------------------------------------------
    Runs a text search via the Tavily API (requires TAVILY_API_KEY).
    Args:
        query (str): search keywords
        max_results (int): Max results to return (default 5)
    Returns:
        JSON string with the Tavily response (truncated to the tool output cap).
    Usage: For agents/tools needing current info, web links, or knowledge discovery tasks.
    """
    try:
        from tavily import TavilyClient

        tavily_client = TavilyClient()
        response = tavily_client.search(query, max_results=max_results)

        return _truncate_text(dumps(response))
    except Exception as e:
        return dumps([{"title": "tavily error", "href": "", "body": str(e)}])


# --- Bash command runner (current environment) ---
def mcp_bash_runner(command: str, cwd: str | None = None, timeout_sec: int = 60) -> str:
    """
    Agent Tool: Bash Command Runner (sandboxed)
    -------------------------------------------
    Execute a Bash command inside the session workspace. `cwd` is REQUIRED
    and must be an absolute path under the session root; the child runs with
    a scrubbed minimal environment (no API keys, HOME redirected into the
    workspace) and git/dangerous commands are denied. Do not use for running
    python scripts.
    Args:
        command: Command string to run (interpreted by bash).
        cwd: REQUIRED absolute working directory inside the job workspace.
        timeout_sec: Max seconds before timing out (server-side ceiling applies).
    Returns:
        JSON string with keys: 'stdout', 'stderr', 'returncode', 'cwd', and 'elapsed_sec'.
    """
    import time

    # The incident vector: a cwd-less bash_run inherited the SERVICE cwd
    # (this repo) and the full env (git identity, API keys). cwd is now
    # mandatory + jailed, the env is scrubbed, and git is denied.
    if not cwd:
        return dumps(
            {
                "stdout": "",
                "stderr": (
                    "cwd is required: pass the absolute session workspace "
                    "directory to run commands in."
                ),
                "returncode": -1,
                "cwd": "",
            }
        )
    guard = _guard_workspace_path(cwd)
    if guard:
        return dumps({"stdout": "", "stderr": guard, "returncode": -1, "cwd": cwd})
    denied = _deny_command(command)
    if denied:
        return dumps({"stdout": "", "stderr": denied, "returncode": -1, "cwd": cwd})

    timeout_sec = _clamp_timeout(timeout_sec, "ATOMAGENTS_BASH_TIMEOUT_MAX", 600)
    env = _scrubbed_env(cwd)

    # Argv form of the former shell=True/executable="/bin/bash" spawn — same
    # semantics, but lets the sandbox wrapper prefix the command.
    argv, refusal = _wrap_sandbox(["/bin/bash", "-c", command])
    if refusal:
        return dumps({"stdout": "", "stderr": refusal, "returncode": -1, "cwd": cwd})

    try:
        start = time.time()
        result = _run_child(argv, cwd=cwd, env=env, timeout=timeout_sec)
        elapsed = time.time() - start

        return dumps(
            {
                "stdout": _truncate_text(result.stdout),
                "stderr": _truncate_text(result.stderr),
                "returncode": result.returncode,
                "cwd": str(Path(cwd).resolve()),
                "elapsed_sec": round(elapsed, 3),
            }
        )
    except subprocess.TimeoutExpired:
        return dumps(
            {
                "stdout": "",
                "stderr": f"Command timed out after {timeout_sec} seconds.",
                "returncode": -1,
                "cwd": str(Path(cwd).resolve()),
            }
        )
    except Exception as e:
        return dumps(
            {
                "stdout": "",
                "stderr": f"Execution error: {e}",
                "returncode": -1,
                "cwd": str(Path(cwd).resolve()),
            }
        )


# --- LAMMPS runner tool: runs a LAMMPS simulation given input file string, returns output (from MCP) ---


# LAMMPS input built-ins that execute arbitrary code from inside the
# simulation (in-script RCE with whatever env the runner has). Comments are
# stripped before scanning so `# shell ...` doesn't false-positive.
_LAMMPS_RCE = re.compile(r"^\s*(shell|python)\b", re.MULTILINE)


def _lammps_command(input_file: Path, log_file: Path) -> list[str] | None:
    """Resolve the LAMMPS launcher: ATOMAGENTS_LAMMPS_CMD env override, else
    mpirun+lmp_mpi if both are available, else serial lmp. None if nothing found."""
    import shlex

    override = os.environ.get("ATOMAGENTS_LAMMPS_CMD", "").strip()
    if override:
        return shlex.split(override) + ["-in", str(input_file), "-log", str(log_file)]
    nproc = os.environ.get("ATOMAGENTS_LAMMPS_NP", "12")
    if shutil.which("mpirun") and shutil.which("lmp_mpi"):
        return [
            "mpirun", "-np", nproc, "lmp_mpi",
            "-in", str(input_file), "-log", str(log_file),
        ]
    for serial in ("lmp", "lmp_serial"):
        if shutil.which(serial):
            return [serial, "-in", str(input_file), "-log", str(log_file)]
    return None


def _read_lammps_log(script: str, work_dir: Path, default_log: Path) -> tuple[str, str]:
    """Return (log_text, log_path_used). Honors an in-script `log <path>` command,
    which redirects LAMMPS away from the `-log` target — the historical bug was
    reading only the (near-empty) `-log` file."""
    import re

    log_path = default_log
    for m in re.finditer(r"^\s*log\s+(\S+)", script, re.MULTILINE):
        candidate = Path(m.group(1))
        if not candidate.is_absolute():
            candidate = work_dir / candidate
        # Containment: an in-script `log /anywhere` must not turn this into
        # an arbitrary-file read — only follow candidates inside work_dir.
        try:
            if not candidate.resolve().is_relative_to(Path(work_dir).resolve()):
                continue
        except (OSError, ValueError):
            continue
        log_path = candidate
    texts = []
    for p in dict.fromkeys([log_path, default_log]):  # ordered, unique
        try:
            texts.append(p.read_text(errors="replace"))
        except OSError:
            continue
    return (max(texts, key=len) if texts else ""), str(log_path)


def mcp_lammps_runner(infile_string: str, work_dir, infile: str) -> dict:
    """
    Run a LAMMPS simulation using the given input file content string.
    Runs with cwd=work_dir (relative paths in the script stay inside it), honors
    an in-script `log` command when collecting output, and returns the parsed
    ERROR/WARNING lines plus the log tail — also on failure and timeout.
    Args:
        infile_string (str): LAMMPS input script as a string.
        work_dir (str): Directory to run the simulation in (absolute).
        infile (str): Name of the input file to be created in the work directory.
    Returns:
        JSON string with 'returncode', 'errors', 'warnings', 'output' (log tail),
        'log_file', 'stderr', 'error' (summary or None).
    Usage: For agents/tools needing to run LAMMPS simulations and analyze results.
    """
    import re

    guard = _guard_workspace_path(str(work_dir))
    if guard:
        return dumps({"error": guard, "output": None})
    if not infile or "/" in infile or infile.startswith("."):
        return dumps(
            {"error": f"Invalid input file name {infile!r} (plain filename expected).",
             "output": None}
        )
    hit = _LAMMPS_RCE.search(re.sub(r"#.*", "", infile_string))
    if hit:
        return dumps(
            {
                "error": (
                    f"Refusing to run this LAMMPS script: the {hit.group(1)!r} "
                    "command executes arbitrary code from inside the simulation. "
                    "Remove it and do that work via code_runner/bash_run instead."
                ),
                "output": None,
            }
        )
    # Parse-safe server-side ceiling (a malformed env value used to raise
    # straight out of the tool).
    timeout_sec = _clamp_timeout(None, "ATOMAGENTS_LAMMPS_TIMEOUT", 3000)
    try:
        temp_path = Path(work_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        input_file = temp_path / infile
        output_file = temp_path / f"{infile}_lammps_output.txt"
        with open(input_file, "w") as f:
            f.write(infile_string)

        command = _lammps_command(input_file, output_file)
        if command is None:
            return dumps(
                {
                    "error": (
                        "No LAMMPS executable found (tried mpirun+lmp_mpi, lmp, "
                        "lmp_serial). Set ATOMAGENTS_LAMMPS_CMD to the launcher."
                    ),
                    "output": None,
                }
            )

        # Same containment as every other exec tool: scrubbed env (no API
        # keys, HOME/TMPDIR inside the workspace) + seatbelt + group kill.
        env = _scrubbed_env(str(temp_path))
        argv, refusal = _wrap_sandbox(command)
        if refusal:
            return dumps({"error": refusal, "output": None})

        timed_out = False
        stderr_text = ""
        returncode = -1
        try:
            result = _run_child(
                argv, cwd=str(temp_path), env=env, timeout=timeout_sec
            )
            stderr_text = result.stderr or ""
            returncode = result.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            stderr_text = _as_text(e.stderr)

        log_text, log_path_used = _read_lammps_log(
            infile_string, temp_path, output_file
        )
        errors = re.findall(r"^ERROR.*$", log_text, re.MULTILINE)
        warnings = re.findall(r"^WARNING.*$", log_text, re.MULTILINE)

        error_summary = None
        if timed_out:
            error_summary = f"LAMMPS timed out after {timeout_sec}s (partial log returned)."
        elif returncode != 0:
            error_summary = (
                (errors[-1] if errors else "")
                or _tail_lines(stderr_text, 20, 4096)
                or f"LAMMPS exited with code {returncode}."
            )

        payload = {
            "returncode": -1 if timed_out else returncode,
            "errors": errors[:50],
            "warnings": warnings[:50],
            "output": _tail_lines(log_text, 200),
            "log_file": log_path_used,
            "stderr": _tail_lines(stderr_text, 40, 8192) if error_summary else "",
            "error": error_summary,
        }
        # Physics validation gate: attach a machine-readable verdict so a run
        # that "completed" but produced garbage (no integrator, NaN, blow-up)
        # cannot be reported as PASS. Optional in standalone deployments.
        try:
            try:
                from AtomAgents.utils.run_validation import validate_lammps_run
            except ImportError:
                from ..utils.run_validation import validate_lammps_run  # type: ignore
            payload["validation"] = validate_lammps_run(
                infile_string, log_text, work_dir=str(temp_path)
            )
        except Exception as val_err:  # validation must never mask the run result
            payload["validation"] = {"overall": "UNKNOWN", "error": str(val_err)}
        print(
            f"LAMMPS run rc={payload['returncode']} log={log_path_used} "
            f"errors={len(errors)} warnings={len(warnings)}"
        )
        return dumps(payload)
    except Exception as e:
        return dumps({"error": str(e), "output": None})


# --- Pure DOLFINx versions of the FEniCS helpers ---
# Drop-in replacements for your three functions that previously had
# dolfin/dolfinx split paths. These now depend only on:
#   dolfinx, ufl, mpi4py, petsc4py, matplotlib, numpy
# Signatures & returns are unchanged.

from typing import Tuple, Dict, Any


def mcp_fenics_mesh_visualize(
    save_file: str,
    domain: str = "unit_square",
    nx: int = 32,
    ny: int = 32,
    nz: int | None = None,
) -> Tuple[bool, str]:
    """
    Visualize a simple DOLFINx mesh and return a base64-encoded PNG.
    """
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpi4py import MPI
    from dolfinx import mesh as dxmesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from base64 import b64encode
    from pathlib import Path

    def _ensure_parent_dir(path: str) -> None:
        p = Path(path)
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)

    def _encode_png_to_b64(png_path: str) -> str:
        with open(png_path, "rb") as f:
            return b64encode(f.read()).decode("utf-8")

    try:
        _ensure_parent_dir(save_file)

        if domain.lower() == "unit_square":
            m = dxmesh.create_unit_square(MPI.COMM_SELF, nx, ny)
            # Draw all triangle edges
            tdim = m.topology.dim
            m.topology.create_connectivity(tdim, 0)
            c2v = m.topology.connectivity(tdim, 0)
            num_cells_local = m.topology.index_map(tdim).size_local
            triangles = [c2v.links(i) for i in range(num_cells_local)]
            coords = m.geometry.x

            fig = plt.figure(figsize=(5, 5), dpi=150)
            ax = fig.add_subplot(111)
            for tri in triangles:
                pts = coords[tri]
                loop = np.vstack([pts, pts[0]])
                ax.plot(loop[:, 0], loop[:, 1], linewidth=0.3)
            ax.set_aspect("equal")
            ax.set_title(f"Mesh (dolfinx): {domain}, res=({nx},{ny})")
            plt.tight_layout()
            plt.savefig(save_file, bbox_inches="tight")
            plt.close(fig)
            return True, _encode_png_to_b64(save_file)

        elif domain.lower() == "unit_cube":
            if nz is None:
                return False, "For unit_cube, 'nz' resolution is required."
            m = dxmesh.create_unit_cube(MPI.COMM_SELF, nx, ny, nz)
            tdim = m.topology.dim
            # Exterior facets via boundary locator (avoid c2f.links due to env issues)
            m.topology.create_connectivity(tdim - 1, 0)
            c2v = m.topology.connectivity(tdim - 1, 0)
            facets_bnd = dxmesh.locate_entities_boundary(
                m,
                tdim - 1,
                lambda x: np.isclose(x[0], 0)
                | np.isclose(x[0], 1)
                | np.isclose(x[1], 0)
                | np.isclose(x[1], 1)
                | np.isclose(x[2], 0)
                | np.isclose(x[2], 1),
            )
            exterior = facets_bnd

            coords = m.geometry.x
            faces = [c2v.links(fi) for fi in exterior]
            polys = [coords[f] for f in faces]

            fig = plt.figure(figsize=(6, 6), dpi=150)
            ax = fig.add_subplot(111, projection="3d")
            coll = Poly3DCollection(polys, linewidths=0.05, alpha=0.15)
            ax.add_collection3d(coll)
            xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
            ax.set_xlim(xs.min(), xs.max())
            ax.set_ylim(ys.min(), ys.max())
            ax.set_zlim(zs.min(), zs.max())
            ax.set_title(f"Mesh (dolfinx 3D): {domain}, res=({nx},{ny},{nz})")
            plt.tight_layout()
            plt.savefig(save_file, bbox_inches="tight")
            plt.close(fig)
            return True, _encode_png_to_b64(save_file)
        else:
            return (
                False,
                f"Unsupported domain: {domain}. Use 'unit_square' or 'unit_cube'.",
            )
    except Exception as e:
        return False, f"fenics_mesh_visualize error: {e}"


def mcp_fenics_poisson_visualize(
    save_file: str,
    domain: str = "unit_square",
    nx: int = 32,
    ny: int = 32,
    f_value: float = 1.0,
    dirichlet_value: float = 0.0,
    nz: int | None = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Solve -Δu = f with u = dirichlet_value on the boundary (DOLFINx only).
    Returns base64 PNG and simple metrics.
    """
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import mesh as dxmesh, fem
    import ufl
    from matplotlib.tri import Triangulation
    from base64 import b64encode
    from pathlib import Path

    def _ensure_parent_dir(path: str) -> None:
        p = Path(path)
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)

    def _encode_png_to_b64(png_path: str) -> str:
        with open(png_path, "rb") as f:
            return b64encode(f.read()).decode("utf-8")

    try:
        _ensure_parent_dir(save_file)
        if domain.lower() == "unit_square":
            m = dxmesh.create_unit_square(MPI.COMM_SELF, nx, ny)
        elif domain.lower() == "unit_cube":
            if nz is None:
                return False, "For unit_cube, 'nz' resolution is required.", {}
            m = dxmesh.create_unit_cube(MPI.COMM_SELF, nx, ny, nz)
        else:
            return False, f"Unsupported domain: {domain}.", {}

        V = fem.functionspace(m, ("Lagrange", 1))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        f_const = fem.Constant(m, PETSc.ScalarType(f_value))
        uD = fem.Constant(m, PETSc.ScalarType(dirichlet_value))

        tdim = m.topology.dim
        # All boundary facets
        facets = dxmesh.locate_entities_boundary(
            m,
            tdim - 1,
            lambda x: np.logical_or.reduce(
                [np.isclose(x[i], 0.0) | np.isclose(x[i], 1.0) for i in range(tdim)]
            ),
        )
        bdofs = fem.locate_dofs_topological(V, tdim - 1, facets)
        bc = fem.dirichletbc(uD, bdofs, V)

        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = f_const * v * ufl.dx

        try:
            from dolfinx.fem.petsc import LinearProblem  # type: ignore

            problem = LinearProblem(
                a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
            )
            uh = problem.solve()
        except Exception:
            a_form = fem.form(a)
            L_form = fem.form(L)
            try:
                from dolfinx.fem.petsc import (
                    assemble_matrix,
                    assemble_vector,
                    apply_lifting,
                    set_bc,
                    create_vector,
                )  # type: ignore
            except Exception:
                assemble_matrix = fem.assemble_matrix  # type: ignore
                assemble_vector = fem.assemble_vector  # type: ignore
                apply_lifting = fem.apply_lifting  # type: ignore
                set_bc = fem.set_bc  # type: ignore
                create_vector = fem.create_vector  # type: ignore
            A = assemble_matrix(a_form, bcs=[bc])
            A.assemble()
            b = create_vector(L_form)
            with b.localForm() as loc:
                loc.set(0)
            assemble_vector(b, L_form)
            apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc])
            uh = fem.Function(V)
            xvec = getattr(uh, "vector", None) or uh.x
            ksp = PETSc.KSP().create(m.comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.solve(b, xvec)
            xvec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

        # Metrics
        l2_sq = fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ufl.dx))
        l2 = float(np.sqrt(l2_sq))
        grad_energy = float(
            fem.assemble_scalar(
                fem.form(ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx)
            )
        )
        metrics = {"l2_norm": l2, "grad_energy": grad_energy}

        # Visualization
        if tdim == 2:
            dof_coords = V.tabulate_dof_coordinates()
            x = dof_coords[:, 0]
            y = dof_coords[:, 1]
            z = uh.x.array.real
            num_cells = m.topology.index_map(tdim).size_local
            cell_dofs = np.array(
                [V.dofmap.cell_dofs(ci) for ci in range(num_cells)], dtype=np.int32
            )
            tri = Triangulation(x, y, cell_dofs)
            fig = plt.figure(figsize=(5, 5), dpi=150)
            ax = fig.add_subplot(111)
            tpc = ax.tripcolor(tri, z, shading="gouraud")
            fig.colorbar(tpc)
            ax.set_aspect("equal")
            ax.set_title(
                f"Poisson (dolfinx) domain={domain}, f={f_value}, uD={dirichlet_value}"
            )
            plt.tight_layout()
            plt.savefig(save_file, bbox_inches="tight")
            plt.close(fig)
            return True, _encode_png_to_b64(save_file), metrics
        else:
            # 3D: color exterior facets by nearest scalar dof
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from matplotlib import cm
            from matplotlib.colors import Normalize

            m.topology.create_connectivity(tdim - 1, 0)
            c2v = m.topology.connectivity(tdim - 1, 0)
            facets_bnd = dxmesh.locate_entities_boundary(
                m,
                tdim - 1,
                lambda x: np.isclose(x[0], 0)
                | np.isclose(x[0], 1)
                | np.isclose(x[1], 0)
                | np.isclose(x[1], 1)
                | np.isclose(x[2], 0)
                | np.isclose(x[2], 1),
            )
            exterior = facets_bnd
            coords = m.geometry.x
            faces = [c2v.links(fi) for fi in exterior]
            dof_coords = V.tabulate_dof_coordinates()
            vals = uh.x.array.real
            polys = []
            face_values = []
            for f in faces:
                vert_xyz = coords[f]
                polys.append(vert_xyz)
                diffs = dof_coords[:, None, :] - vert_xyz[None, :, :]
                d2 = (diffs * diffs).sum(-1)
                nn = np.argmin(d2, axis=0)
                face_values.append(float(np.mean(vals[nn])))
            norm = Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
            cmap = cm.viridis
            facecolors = [cmap(norm(v)) for v in face_values]
            fig = plt.figure(figsize=(7, 6), dpi=150)
            ax = fig.add_subplot(111, projection="3d")
            coll = Poly3DCollection(
                polys, facecolors=facecolors, linewidths=0.05, alpha=0.95
            )
            ax.add_collection3d(coll)
            xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
            ax.set_xlim(xs.min(), xs.max())
            ax.set_ylim(ys.min(), ys.max())
            ax.set_zlim(zs.min(), zs.max())
            ax.set_title(
                f"Poisson (dolfinx 3D) domain={domain}, f={f_value}, uD={dirichlet_value}"
            )
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
            fig.colorbar(mappable, ax=ax, shrink=0.75)
            plt.tight_layout()
            plt.savefig(save_file, bbox_inches="tight")
            plt.close(fig)
            return True, _encode_png_to_b64(save_file), metrics
    except Exception as e:
        return False, f"fenics_poisson_visualize error (dolfinx): {e}", {}


def mcp_fenics_elasticity_visualize(
    save_file: str,
    domain: str = "unit_square",
    nx: int = 32,
    ny: int = 32,
    nz: int | None = None,
    E: float = 210e9,
    nu: float = 0.3,
    traction: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    plane_stress: bool = False,
    field: str = "u_mag",
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Linear elasticity in DOLFINx with u=0 on x=0, traction on x=1.
    Visualize displacement magnitude (u_mag) or von Mises (von_mises).
    """
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import mesh as dxmesh, fem
    from matplotlib.tri import Triangulation
    from base64 import b64encode
    from pathlib import Path

    def _ensure_parent_dir(path: str) -> None:
        p = Path(path)
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)

    def _encode_png_to_b64(png_path: str) -> str:
        with open(png_path, "rb") as f:
            return b64encode(f.read()).decode("utf-8")

    # Helper for exterior surface measure tagging (right side only)
    def _dolfinx_measure_ds_for_sides(m, side_ids: dict[int, np.ndarray]):
        from dolfinx import mesh as _dxm
        import numpy as _np

        tdim = m.topology.dim
        tags = []
        vals = []
        for val, idxs in side_ids.items():
            if idxs.size:
                tags.append(idxs.astype(_np.int32))
                vals.append(_np.full_like(idxs, val, dtype=_np.int32))
        if len(tags) == 0:
            mt = _dxm.meshtags(
                m,
                tdim - 1,
                _np.array([], dtype=_np.int32),
                _np.array([], dtype=_np.int32),
            )
        else:
            indices = _np.hstack(tags)
            values = _np.hstack(vals)
            mt = _dxm.meshtags(m, tdim - 1, indices, values)
        import ufl as _ufl

        ds = _ufl.Measure("ds", domain=m, subdomain_data=mt)
        return mt, ds

    try:
        _ensure_parent_dir(save_file)
        if domain.lower() == "unit_square":
            m = dxmesh.create_unit_square(MPI.COMM_SELF, nx, ny)
            tdim = 2
        elif domain.lower() == "unit_cube":
            if nz is None:
                return False, "For unit_cube, 'nz' resolution is required.", {}
            m = dxmesh.create_unit_cube(MPI.COMM_SELF, nx, ny, nz)
            tdim = 3
        else:
            return False, f"Unsupported domain: {domain}", {}

        V = fem.functionspace(m, ("Lagrange", 1, (tdim,)))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        mu = fem.Constant(m, PETSc.ScalarType(E / (2 * (1 + nu))))
        if tdim == 2 and plane_stress:
            lam_val = (E * nu) / (1 - nu * nu)
        else:
            lam_val = E * nu / ((1 + nu) * (1 - 2 * nu))
        lam = fem.Constant(m, PETSc.ScalarType(lam_val))

        def eps(w):
            return ufl.sym(ufl.grad(w))

        def sigma(w):
            return 2 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(tdim)

        b = fem.Constant(m, PETSc.ScalarType((0.0,) * tdim))

        # Boundary: left Dirichlet u=0
        facets_left = dxmesh.locate_entities_boundary(
            m, tdim - 1, lambda x: np.isclose(x[0], 0.0)
        )
        bdofs = fem.locate_dofs_topological(V, tdim - 1, facets_left)
        bc = fem.dirichletbc(fem.Constant(m, PETSc.ScalarType((0.0,) * tdim)), bdofs, V)

        # Traction on right boundary (x=1)
        facets_right = dxmesh.locate_entities_boundary(
            m, tdim - 1, lambda x: np.isclose(x[0], 1.0)
        )
        _, ds = _dolfinx_measure_ds_for_sides(m, {1: facets_right})
        t_vec = fem.Constant(m, PETSc.ScalarType(traction[:tdim]))

        a = ufl.inner(sigma(u), eps(v)) * ufl.dx
        L = ufl.dot(b, v) * ufl.dx + ufl.dot(t_vec, v) * ds(1)

        try:
            from dolfinx.fem.petsc import LinearProblem  # type: ignore

            problem = LinearProblem(
                a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
            )
            uh = problem.solve()
        except Exception:
            a_form = fem.form(a)
            L_form = fem.form(L)
            try:
                from dolfinx.fem.petsc import (
                    assemble_matrix,
                    assemble_vector,
                    apply_lifting,
                    set_bc,
                    create_vector,
                )  # type: ignore
            except Exception:
                assemble_matrix = fem.assemble_matrix  # type: ignore
                assemble_vector = fem.assemble_vector  # type: ignore
                apply_lifting = fem.apply_lifting  # type: ignore
                set_bc = fem.set_bc  # type: ignore
                create_vector = fem.create_vector  # type: ignore
            A = assemble_matrix(a_form, bcs=[bc])
            A.assemble()
            bvec = create_vector(L_form)
            with bvec.localForm() as loc:
                loc.set(0)
            assemble_vector(bvec, L_form)
            apply_lifting(bvec, [a_form], bcs=[[bc]])
            bvec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            set_bc(bvec, [bc])
            uh = fem.Function(V)
            xvec = getattr(uh, "vector", None) or uh.x
            ksp = PETSc.KSP().create(m.comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.solve(bvec, xvec)
            xvec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

        # Metrics & visualization fields (project to scalar P1 space)
        W = fem.functionspace(m, ("Lagrange", 1))
        u_mag_expr = fem.Expression(
            ufl.sqrt(ufl.inner(uh, uh)), W.element.interpolation_points()
        )
        u_mag = fem.Function(W)
        u_mag.interpolate(u_mag_expr)

        dev = lambda T: T - ufl.tr(T) / tdim * ufl.Identity(tdim)
        vm_expr = fem.Expression(
            ufl.sqrt(1.5 * ufl.inner(dev(sigma(uh)), dev(sigma(uh)))),
            W.element.interpolation_points(),
        )
        vm = fem.Function(W)
        vm.interpolate(vm_expr)

        l2 = float(np.sqrt(fem.assemble_scalar(fem.form(u_mag * u_mag * ufl.dx))))
        vm_l2 = float(np.sqrt(fem.assemble_scalar(fem.form(vm * vm * ufl.dx))))
        metrics = {"u_mag_L2": l2, "von_mises_L2": vm_l2}

        if tdim == 2:
            z = (vm if field == "von_mises" else u_mag).x.array.real
            dof_coords = W.tabulate_dof_coordinates()
            x = dof_coords[:, 0]
            y = dof_coords[:, 1]
            num_cells = m.topology.index_map(tdim).size_local
            cell_dofs = np.array(
                [W.dofmap.cell_dofs(ci) for ci in range(num_cells)], dtype=np.int32
            )
            tri = Triangulation(x, y, cell_dofs)
            fig = plt.figure(figsize=(6, 5), dpi=150)
            ax = fig.add_subplot(111)
            tpc = ax.tripcolor(tri, z, shading="gouraud")
            fig.colorbar(tpc)
            ax.set_aspect("equal")
            ax.set_title(f"Elasticity (dolfinx) field={field}")
            plt.tight_layout()
            plt.savefig(save_file, bbox_inches="tight")
            plt.close(fig)
            return True, _encode_png_to_b64(save_file), metrics
        else:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from matplotlib.colors import Normalize

            m.topology.create_connectivity(tdim - 1, 0)
            c2v = m.topology.connectivity(tdim - 1, 0)
            facets_bnd = dxmesh.locate_entities_boundary(
                m,
                tdim - 1,
                lambda x: np.isclose(x[0], 0)
                | np.isclose(x[0], 1)
                | np.isclose(x[1], 0)
                | np.isclose(x[1], 1)
                | np.isclose(x[2], 0)
                | np.isclose(x[2], 1),
            )
            exterior = facets_bnd
            coords = m.geometry.x
            faces = [c2v.links(fi) for fi in exterior]

            vals = (vm if field == "von_mises" else u_mag).x.array.real
            dof_coords = W.tabulate_dof_coordinates()
            polys = []
            face_values = []
            for f in faces:
                vert_xyz = coords[f]
                polys.append(vert_xyz)
                diffs = dof_coords[:, None, :] - vert_xyz[None, :, :]
                d2 = (diffs * diffs).sum(-1)
                nn = np.argmin(d2, axis=0)
                face_values.append(float(np.mean(vals[nn])))
            norm = Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
            from matplotlib import cm as _cm

            facecolors = [_cm.viridis(norm(v)) for v in face_values]
            fig = plt.figure(figsize=(7, 6), dpi=150)
            ax = fig.add_subplot(111, projection="3d")
            coll = Poly3DCollection(
                polys, facecolors=facecolors, linewidths=0.05, alpha=0.95
            )
            ax.add_collection3d(coll)
            xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
            ax.set_xlim(xs.min(), xs.max())
            ax.set_ylim(ys.min(), ys.max())
            ax.set_zlim(zs.min(), zs.max())
            ax.set_title(f"Elasticity (dolfinx 3D) field={field}")
            mappable = _cm.ScalarMappable(norm=norm, cmap=_cm.viridis)
            mappable.set_array([])
            fig.colorbar(mappable, ax=ax, shrink=0.75)
            plt.tight_layout()
            plt.savefig(save_file, bbox_inches="tight")
            plt.close(fig)
            return True, _encode_png_to_b64(save_file), metrics
    except Exception as e:
        # Keep the 3-tuple contract the service wrapper unpacks — raising here
        # crashed the tool call instead of reporting the error to the agent.
        return False, f"fenics_elasticity_visualize error (dolfinx): {e}", {}
