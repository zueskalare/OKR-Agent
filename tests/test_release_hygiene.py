"""Release hygiene: the public tree must stay free of secrets and machine paths.

These checks are generic on purpose (they name patterns, not people) so the
test itself leaks nothing. They run over `git ls-files`, i.e. exactly what a
commit would publish; outside a git checkout they skip.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),          # OpenAI-style keys (test dummies are shorter)
    re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}"),
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    re.compile(r"tvly-[A-Za-z0-9]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[bpas]-[A-Za-z0-9-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]
HOME_PATH = re.compile(r"(/Users/[^/\s'\"`]+/|/home/[^/\s'\"`]+/)")
# Files allowed to contain a deliberately made-up home path (fixtures of the
# path-guard tests). Anything else with one is a leak of somebody's checkout.
HOME_PATH_ALLOWLIST = {
    "tests/test_mcp_guards.py",
    "tests/test_release_hygiene.py",   # the pattern above matches its own source
}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".cff", ".sh",
    ".in", ".log", ".dat", ".ipynb", ".example", ".cfg", ".ini", "",
}


def _tracked() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout — hygiene checks need `git ls-files`")
    return [REPO / p for p in out.split("\0") if p]


def _text_files() -> list[Path]:
    return [p for p in _tracked() if p.suffix in TEXT_SUFFIXES and p.is_file()]


def test_no_env_or_credential_files_are_tracked():
    names = {p.name for p in _tracked()}
    suffixes = {p.suffix for p in _tracked()}
    assert ".env" not in names, ".env must never be committed"
    assert not {n for n in names if n.startswith(".env.") and n != ".env.example"}
    assert ".env.example" in names, "the template must ship"
    assert not ({".pem", ".key", ".db", ".sqlite"} & suffixes)


def test_no_key_shaped_strings_in_tracked_text():
    hits = []
    for p in _text_files():
        text = p.read_text(errors="replace")
        for pat in KEY_PATTERNS:
            for m in pat.finditer(text):
                hits.append(f"{p.relative_to(REPO)}: {m.group(0)[:12]}…")
    assert not hits, "credential-shaped strings in the tree:\n" + "\n".join(hits)


def test_no_personal_checkout_paths_in_tracked_text():
    hits = []
    for p in _text_files():
        rel = str(p.relative_to(REPO))
        if rel in HOME_PATH_ALLOWLIST:
            continue
        for ln, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if HOME_PATH.search(line):
                hits.append(f"{rel}:{ln}")
    assert not hits, "absolute home-directory paths in the tree:\n" + "\n".join(hits)


def test_notebooks_are_committed_without_outputs():
    dirty = []
    for p in _tracked():
        if p.suffix != ".ipynb":
            continue
        nb = json.loads(p.read_text())
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("outputs") or cell.get("execution_count") is not None:
                dirty.append(f"{p.relative_to(REPO)} cell {i}")
    assert not dirty, "notebooks must be stripped before commit:\n" + "\n".join(dirty)
