"""Tool classification for per-agent act-stage tool policies (WS25, A7).

One place that answers "is this an execution tool?" — used by the Supervisor
to bind only coordination tools to its act stage, and by the policy tests to
pin the classification of every real tool name. Keep the two name sets in
sync with the MCP service (`mcp_service/mcp_service.py`) and the OKR tools
(`utils/okr_tools.py`); `tests/test_supervisor_tool_policy.py` fails on a
rename rather than letting the policy silently degrade.
"""

import logging
from typing import Any, List, Tuple

logger = logging.getLogger("AtomAgents.tool_policy")

# Read/search/board tools: safe for a coordinator — they observe or annotate,
# they never create artifacts or run code.
READ_ONLY_TOOLS = frozenset(
    {
        "file_reader",
        "web_search",
        "fenicsx_doc_query",
        "lammps_doc_query",
        "okr_update",
        "register_artefact",
        "record_metric",
        "record_attempt",
    }
)

# Tools that create/modify files or execute code/simulations.
EXECUTION_TOOLS = frozenset(
    {
        "code_runner",
        "bash_run",
        "lammps_run",
        "write_file",
        "download_file",
        "md_visualize",
        "fem_mesh_visualize",
        "fem_poisson_solve_visualize",
        "fem_elasticity_solve_visualize",
    }
)

# Fallback heuristics for tools in neither set. Parameter names that
# UNAMBIGUOUSLY indicate writing/executing — deliberately NOT the generic
# "path"/"infile" (read-only tools like a page reader or doc fetcher commonly
# take a path argument; withholding those from the Supervisor silently broke
# its read-and-verify charter duties). Name substrings supplement the params.
_WRITE_EXEC_PARAMS = frozenset(
    {
        "cwd",
        "dest",
        "save_file",
        "save_dir",
        "work_dir",
        "infile_string",
        "code",
        "cmd",
    }
)
_EXEC_NAME_HINTS = (
    "write",
    "run",
    "exec",
    "delete",
    "remove",
    "create",
    "download",
    "upload",
    "install",
    "spawn",
)


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or "")


def _tool_params(tool: Any) -> frozenset:
    """Best-effort parameter names of a tool's schema; empty when unreadable."""
    try:
        args = getattr(tool, "args", None)
        if isinstance(args, dict) and args:
            return frozenset(args.keys())
        schema = getattr(tool, "args_schema", None)
        if schema is not None:
            fields = getattr(schema, "model_fields", None)
            if isinstance(fields, dict):
                return frozenset(fields.keys())
    except Exception as e:  # pragma: no cover - introspection is best-effort
        logger.debug("Tool schema introspection failed for %r: %s", tool, e)
    return frozenset()


def is_execution_tool(tool: Any) -> bool:
    """True when the tool creates artifacts or executes code.

    Order matters: the allowlist wins first — `register_artefact(path=...)`
    takes a path *reference* without writing anything and would otherwise be
    misclassified by a heuristic. Unknown tools classify as execution only on
    an unambiguous signal (exec/write param names or name substrings); a
    read-only tool that merely takes a `path` argument stays with the
    coordinator. Unknown tools are logged so the operator can pin them in the
    name sets (never silently withhold a tool on a guess).
    """
    name = _tool_name(tool)
    if name in READ_ONLY_TOOLS:
        return False
    if name in EXECUTION_TOOLS:
        return True
    lowered = name.lower()
    verdict = any(h in lowered for h in _EXEC_NAME_HINTS) or bool(
        _tool_params(tool) & _WRITE_EXEC_PARAMS
    )
    logger.warning(
        "Tool %r is not in the tool_policy name sets — heuristically "
        "classified as %s; pin it in READ_ONLY_TOOLS/EXECUTION_TOOLS.",
        name or tool,
        "execution (withheld from the Supervisor)" if verdict else "coordination",
    )
    return verdict


def split_tools(tools: List[Any]) -> Tuple[List[Any], List[Any]]:
    """Split into (coordination, execution) preserving order."""
    coordination: List[Any] = []
    execution: List[Any] = []
    for tool in tools or []:
        (execution if is_execution_tool(tool) else coordination).append(tool)
    return coordination, execution
