"""Dry-run tool emulation (WS20.3).

``emulate_tools`` swaps selected tools for same-name/same-schema stand-ins
whose "execution" asks an LLM to fabricate one plausible output. This lets a
full multi-agent flow run end-to-end (routing, prompts, budgets, HITL)
without paying for real MD/FEM runs — the agents cannot tell the difference.

Gated by ``ATOMAGENTS_EMULATE_TOOLS`` (comma-separated tool names, default
empty = everything real). Board/OKR tools are merged AFTER emulation in the
backend, so they always stay real.
"""

from __future__ import annotations

import logging
from typing import Sequence

from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger("AtomAgents.tool_emulation")

_PROMPT = (
    "You are emulating the tool `{name}` for a dry run of an agent system.\n"
    "Tool description:\n{description}\n\n"
    "The tool was called with these arguments:\n{args}\n\n"
    "Reply with ONE plausible, realistic output exactly as the real tool "
    "would return it (same format, believable values, no commentary, no "
    "markdown fences). If the real tool would return JSON, return JSON."
)


def _emulated_copy(tool: BaseTool, llm) -> StructuredTool:
    def _render(kwargs) -> str:
        return _PROMPT.format(
            name=tool.name, description=tool.description or "(none)", args=kwargs
        )

    def _sync(**kwargs) -> str:
        resp = llm.invoke(_render(kwargs))
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    async def _async(**kwargs) -> str:
        resp = await llm.ainvoke(_render(kwargs))
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=_sync,
        coroutine=_async,
    )


def emulate_tools(
    tools: Sequence[BaseTool], names: Sequence[str], llm
) -> list[BaseTool]:
    """Return ``tools`` with every tool named in ``names`` replaced by an
    LLM-backed emulator (same name, description, and argument schema)."""
    wanted = {n.strip() for n in names if n and n.strip()}
    if not wanted:
        return list(tools)
    out: list[BaseTool] = []
    replaced = []
    for tool in tools:
        if tool.name in wanted:
            out.append(_emulated_copy(tool, llm))
            replaced.append(tool.name)
        else:
            out.append(tool)
    missing = wanted - set(replaced)
    if replaced:
        logger.warning("DRY RUN — emulating tools (no real execution): %s", replaced)
    if missing:
        logger.warning("emulate_tools: names not found among tools: %s", sorted(missing))
    return out


__all__ = ["emulate_tools"]
