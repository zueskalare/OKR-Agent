from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional


HEADER = "# Project Task Board"


def _markers(section: str) -> tuple[str, str]:
    start = f"<!-- START {section} -->"
    end = f"<!-- END {section} -->"
    return start, end


def ensure_readme(project_root: str | Path) -> Path:
    """Ensure a README.md exists with default sections.

    Creates `<project_root>/README.md` if missing and scaffolds sections for
    Supervisor, Modeler, Simulator, and Tasks with marker ranges that allow
    idempotent updates.
    """
    root = Path(project_root)
    root.mkdir(parents=True, exist_ok=True)
    readme = root / "README.md"
    if not readme.exists():
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        content = [
            HEADER,
            "",
            f"Initialized at {timestamp} (UTC)",
            "",
        ]
        for sec in ("Tasks", "Supervisor", "Modeler", "Simulator"):
            s, e = _markers(sec)
            content.extend([
                f"## {sec}",
                s,
                f"_No entries for {sec} yet._",
                e,
                "",
            ])
        readme.write_text("\n".join(content), encoding="utf-8")
    return readme


def update_section(
    project_root: str | Path,
    section: str,
    markdown: str,
    heading_level: int = 2,
) -> Path:
    """Replace or append a section block in README delimited by markers.

    The block is placed between `<!-- START {section} -->` and
    `<!-- END {section} -->`. If the section heading doesn't exist, it is
    appended at the end under a `## {section}` heading (or a custom level).
    Returns the README path.
    """
    readme = ensure_readme(project_root)
    text = readme.read_text(encoding="utf-8")
    start, end = _markers(section)

    # Make sure a heading exists for the section
    heading = f"{'#' * heading_level} {section}"
    if heading not in text:
        # Append a new empty block with markers
        if not text.endswith("\n"):
            text += "\n"
        text += "\n".join([heading, start, f"_No entries for {section} yet._", end, ""]) + "\n"

    # Replace marker block content
    pattern = re.compile(rf"({re.escape(start)})([\s\S]*?)({re.escape(end)})", re.MULTILINE)
    replacement = f"{start}\n{markdown.strip()}\n{end}"
    if pattern.search(text):
        text = pattern.sub(replacement, text)
    else:
        # If markers somehow missing, append a fresh block under the heading
        text += "\n" + heading + "\n" + replacement + "\n"

    text = _dedupe_stray_headings(text, section, heading_level)
    readme.write_text(text, encoding="utf-8")
    return readme


def _dedupe_stray_headings(text: str, section: str, heading_level: int = 2) -> str:
    """Remove duplicate `## <section>` blocks that live OUTSIDE the marker span.

    Historically agents also appended their own section headings via shell,
    producing boards with the same section 3-4 times. The marker-delimited
    block is authoritative; any other block under an identical heading is
    dropped (up to the next same-or-higher-level heading).
    """
    start, _ = _markers(section)
    heading = f"{'#' * heading_level} {section}"
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    seen_authoritative = False
    while i < len(lines):
        line = lines[i]
        if line.strip() == heading:
            # Authoritative iff the marker start appears before the next heading
            j = i + 1
            is_auth = False
            while j < len(lines):
                stripped = lines[j].strip()
                if stripped == start:
                    is_auth = True
                    break
                if stripped.startswith("#") and not stripped.startswith("#" * (heading_level + 1)):
                    break
                j += 1
            if is_auth and not seen_authoritative:
                seen_authoritative = True
                out.append(line)
                i += 1
                continue
            # Stray duplicate: skip until the next same-or-higher-level heading
            i += 1
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith("#") and not stripped.startswith(
                    "#" * (heading_level + 1)
                ):
                    break
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def get_section(project_root: str | Path, section: str) -> Optional[str]:
    """Return the current markdown content within a section's markers.

    If README or markers are missing, returns None.
    """
    readme = Path(project_root) / "README.md"
    if not readme.exists():
        return None
    text = readme.read_text(encoding="utf-8")
    start, end = _markers(section)
    pattern = re.compile(rf"{re.escape(start)}\n([\s\S]*?)\n{re.escape(end)}", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return None
    return m.group(1).strip()


def render_tasks_markdown(
    tasks: list[dict],
    subtasks: list[dict],
    key_results: list[dict] | None = None,
) -> str:
    """Render tasks/subtasks/key-results into the README overview.

    SQL is the detail store; this projection is the human-readable summary
    with per-task progress rollups.
    """
    key_results = key_results or []
    lines: list[str] = []
    lines.append("Updated: " + datetime.utcnow().isoformat(timespec="seconds") + "Z")
    lines.append("")
    done_states = {"done", "completed", "achieved", "success"}
    for t in tasks:
        tid = t.get("id")
        children = [s for s in subtasks if s.get("task_id") == tid]
        n_done = sum(
            1 for s in children if str(s.get("status", "")).lower() in done_states
        )
        rollup = f" ({n_done}/{len(children)} subtasks done)" if children else ""
        lines.append(
            f"- [Task #{tid}][{t.get('status','').upper()}] {t.get('description','')}{rollup}"
        )
        for st in children:
            lines.append(
                f"  - [Sub #{st.get('id')}][{st.get('status','').upper()}] {st.get('description','')}"
            )
        for kr in (k for k in key_results if k.get("task_id") == tid):
            mark = "x" if kr.get("achieved") else " "
            lines.append(
                f"  - KR #{kr.get('id')} [{mark}] {kr.get('description','')}"
            )
    if len(lines) == 2:
        return "_No tasks found._"
    return "\n".join(lines)


def render_artefacts_markdown(artefacts: list[dict], metrics: list[dict]) -> str:
    """Render registered artefacts and validation metrics."""
    lines: list[str] = []
    lines.append("Updated: " + datetime.utcnow().isoformat(timespec="seconds") + "Z")
    if artefacts:
        lines.append("")
        lines.append("| Artefact | Kind | Bytes | KR |")
        lines.append("|---|---|---:|---|")
        for a in artefacts:
            kr = f"KR-{a.get('kr_id')}" if a.get("kr_id") else ""
            lines.append(
                f"| `{a.get('path','')}` | {a.get('kind','')} | {a.get('bytes','')} | {kr} |"
            )
    if metrics:
        lines.append("")
        lines.append("| Metric | Value | Unit | KR |")
        lines.append("|---|---:|---|---|")
        for m in metrics:
            kr = f"KR-{m.get('kr_id')}" if m.get("kr_id") else ""
            lines.append(
                f"| {m.get('metric','')} | {m.get('value','')} | {m.get('unit','')} | {kr} |"
            )
    if len(lines) == 1:
        return "_No artefacts registered yet._"
    return "\n".join(lines)

