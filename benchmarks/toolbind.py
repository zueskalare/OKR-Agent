"""The only place a benchmark arm's tools are constructed.

Every arm gets the same 17 tools: the 13 published by the MCP service plus the
4 bookkeeping tools in `okr_tools`. The names, descriptions and argument
schemas are derived from one source (the live `tools/list` plus one spec table)
and hashed into `tools.lock.json`, so "the arms had the same tools" is a
checked fact rather than a claim in a report.

Two failure modes this module exists to make impossible:

* **Silently binding nothing.** The AG2 scaffold returns `[]` when the MCP
  connection fails, so a run exits 0 having called no tools and reads as a
  successful-but-lazy agent. Here, an empty or short tool list raises.
* **Silent drift between arms.** If the service gains or loses a tool between
  two runs, the lockfile mismatches and the campaign stops instead of
  comparing two different environments.

The framework adapters are deliberately thin and imported lazily, because no
single venv has both langchain_core and crewai installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional

from benchmarks.mcp_client import MCPClient
from benchmarks.okr_tools import OKR_TOOL_SPECS, OKRStore

LOCKFILE = Path(__file__).resolve().parent / "tools.lock.json"

# Environment parity with ARM A, not a baseline concession.
#
# AtomAgents caps every tool result client-side at `TOOL_OUTPUT_MAX_BYTES`
# (AtomAgents/config.py:108, default 64 KiB) inside `MCPToolNode._cap_output`,
# with a 75 % head / 25 % tail split and an explicit marker. Its stored messages
# top out at 65 581 chars, confirming the cap is live.
#
# Without the same cap the baselines swallow whole multi-MB payloads: a single
# `md_visualize` base64 PNG added ~250 000 tokens to ARM R's context in one
# call, and two such calls took it from 200 k to 762 k. Measured, twice:
# `prompt is too long: 1007846` then `1068114 tokens > 1000000 maximum`.
#
# Same constant, same env var, same truncation shape as ARM A.
TOOL_OUTPUT_MAX_BYTES = int(os.environ.get("ATOMAGENTS_TOOL_OUTPUT_MAX", 64 * 1024))

EXPECTED_MCP_TOOLS = 13
EXPECTED_OKR_TOOLS = 4
EXPECTED_TOTAL = EXPECTED_MCP_TOOLS + EXPECTED_OKR_TOOLS

_JSON_TYPES: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


class ToolBindError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------


class ToolSpec:
    """A framework-neutral tool: a name, a description, a JSON Schema, a call."""

    __slots__ = ("name", "description", "schema", "fn", "source")

    def __init__(self, name: str, description: str, schema: dict, fn: Callable[..., str], source: str):
        self.name = name
        self.description = description
        self.schema = schema
        self.fn = fn
        self.source = source

    def fingerprint(self) -> str:
        blob = json.dumps(
            {"name": self.name, "description": self.description, "schema": self.schema},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _okr_schema(args: dict[str, tuple]) -> dict:
    props: dict[str, Any] = {}
    required: list[str] = []
    for arg, (typ, default, desc) in args.items():
        props[arg] = {"type": _JSON_TYPES.get(typ, "string"), "description": desc}
        if default is ...:
            required.append(arg)
        else:
            props[arg]["default"] = default
    return {"type": "object", "properties": props, "required": required}


def cap_output(text: str, cap: int = TOOL_OUTPUT_MAX_BYTES) -> str:
    """Mirror of `MCPToolNode._truncate_text` — head 75 %, tail 25 %, marker.

    Kept byte-identical in shape to ARM A's so a truncated result reads the
    same to every arm. The tail matters: LAMMPS writes its verdict at the END
    of a log, so a head-only cap would hide exactly the line that says whether
    the run succeeded.
    """
    # openai route only: long base64 runs (md_visualize returns whole PNGs as
    # base64 JSON) survive the cap as TRUNCATED base64. The claude wire carries
    # that as inert text, but the codex backend behind the gateway promotes
    # data-URL-looking content to an image_url item and then rejects the
    # invalid base64 with a 400 — which killed ws29b-demo2-react_gpt-t1 at
    # input[43] after 12 healthy minutes. Scrub the payload BEFORE capping so
    # nothing image-like ever reaches that wire. Gated on BENCH_PROVIDER so the
    # recorded claude-route arms remain byte-identical to their campaigns.
    if os.environ.get("BENCH_PROVIDER") == "openai":
        # The gateway's codex backend RE-PARSES tool-result strings that look
        # like content-block JSON and reconstructs typed image items, then
        # validates them. Scrubbing the base64 (attempt 3) or the data-URL
        # prefix (attempt 4) still 400s — the `"image_url"` STRUCTURE is the
        # trigger. An image-shaped result carries no other information, so on
        # this wire replace it wholesale with plain prose (no braces at all).
        if '"image_url"' in text or "data:image/" in text:
            return (
                "[image tool result omitted on the openai wire: the gateway "
                "backend rejects reconstructed image blocks. The image file "
                "itself was written to disk by the tool and can be verified "
                "with a size/existence check.]"
            )
        text = re.sub(
            r"[A-Za-z0-9+/=]{4096,}",
            "[base64 payload omitted for the openai wire]",
            text,
        )
    if not cap or len(text) <= cap:
        return text
    head = text[: int(cap * 0.75)]
    tail = text[-int(cap * 0.25):]
    return (
        f"{head}\n... [{len(text) - len(head) - len(tail)} chars "
        f"truncated client-side] ...\n{tail}"
    )


def tool_specs(client: MCPClient, store: OKRStore) -> list[ToolSpec]:
    """The 17. MCP tools first, in the service's own order, then bookkeeping."""
    specs: list[ToolSpec] = []

    for t in client.list_tools():
        name = t["name"]
        specs.append(
            ToolSpec(
                name=name,
                description=(t.get("description") or "").strip(),
                schema=t.get("inputSchema") or {"type": "object", "properties": {}},
                fn=(lambda _n: (lambda **kw: cap_output(client.call_tool(_n, kw))))(name),
                source="mcp",
            )
        )

    for spec in OKR_TOOL_SPECS:
        method = getattr(store, spec["method"])
        specs.append(
            ToolSpec(
                name=spec["name"],
                description=spec["description"],
                schema=_okr_schema(spec["args"]),
                fn=(lambda _m: (lambda **kw: _m(**kw)))(method),
                source="okr",
            )
        )

    _assert_shape(specs)
    return specs


def _assert_shape(specs: list[ToolSpec]) -> None:
    if not specs:
        raise ToolBindError(
            "bound 0 tools — the MCP service at :9999 is unreachable or empty. "
            "Refusing to start a run that would look like a lazy agent."
        )
    n_mcp = sum(1 for s in specs if s.source == "mcp")
    n_okr = sum(1 for s in specs if s.source == "okr")
    if n_mcp != EXPECTED_MCP_TOOLS or n_okr != EXPECTED_OKR_TOOLS:
        raise ToolBindError(
            f"expected {EXPECTED_MCP_TOOLS} MCP + {EXPECTED_OKR_TOOLS} OKR tools, "
            f"got {n_mcp} + {n_okr}"
        )
    dupes = {s.name for s in specs if sum(1 for o in specs if o.name == s.name) > 1}
    if dupes:
        raise ToolBindError(f"duplicate tool names: {sorted(dupes)}")


# ---------------------------------------------------------------------------
# lockfile
# ---------------------------------------------------------------------------


def lock_payload(specs: list[ToolSpec]) -> dict:
    return {
        "note": (
            "Regenerate with `python -m benchmarks.toolbind --write`. Every arm "
            "asserts against this before a billed run."
        ),
        "count": len(specs),
        "tools": {s.name: {"source": s.source, "fingerprint": s.fingerprint()} for s in specs},
        "digest": hashlib.sha256(
            "".join(f"{s.name}:{s.fingerprint()}" for s in sorted(specs, key=lambda x: x.name)).encode()
        ).hexdigest(),
    }


def write_lock(specs: list[ToolSpec]) -> dict:
    payload = lock_payload(specs)
    LOCKFILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def assert_lock(specs: list[ToolSpec]) -> None:
    """Refuse to run if the tool surface has drifted from the pinned one."""
    if not LOCKFILE.exists():
        raise ToolBindError(f"{LOCKFILE.name} missing — run `python -m benchmarks.toolbind --write`")
    pinned = json.loads(LOCKFILE.read_text())
    current = lock_payload(specs)
    if current["digest"] == pinned.get("digest"):
        return
    added = sorted(set(current["tools"]) - set(pinned.get("tools", {})))
    removed = sorted(set(pinned.get("tools", {})) - set(current["tools"]))
    changed = sorted(
        n
        for n in set(current["tools"]) & set(pinned.get("tools", {}))
        if current["tools"][n]["fingerprint"] != pinned["tools"][n]["fingerprint"]
    )
    raise ToolBindError(
        "tool surface drifted from tools.lock.json — the arms would not be "
        f"comparable. added={added} removed={removed} changed={changed}"
    )


# ---------------------------------------------------------------------------
# framework adapters
# ---------------------------------------------------------------------------


def _pydantic_model(name: str, schema: dict):
    """JSON Schema -> pydantic v2 model, for frameworks that demand a class.

    Only the scalar subset the MCP service actually publishes is handled; an
    unknown type falls back to `str` rather than raising, because a tool that
    binds with a slightly loose schema is far better than an arm that binds
    nothing.
    """
    from pydantic import Field, create_model

    py: dict[str, Any] = {"string": str, "integer": int, "number": float, "boolean": bool}
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, tuple] = {}
    for arg, spec in props.items():
        # FastMCP publishes `nz: int | None = None` as {"anyOf": [{"type":
        # "integer"}, {"type": "null"}]} with NO top-level "type"; falling
        # straight to str would then REJECT the integer the tool wants
        # (pydantic v2 does not coerce int -> str), so resolve anyOf first.
        t = spec.get("type")
        if not t:
            t = next(
                (b.get("type") for b in spec.get("anyOf", [])
                 if isinstance(b, dict) and b.get("type") not in (None, "null")),
                "string",
            )
        typ = py.get(t, str)
        desc = spec.get("description", "")
        if arg in required:
            fields[arg] = (typ, Field(..., description=desc))
        else:
            default = spec.get("default", None)
            fields[arg] = (Optional[typ], Field(default, description=desc))
    if not fields:
        # create_model with no fields is legal and yields a no-arg tool.
        return create_model(f"{name}_Args")
    return create_model(f"{name}_Args", **fields)


def build_langchain_tools(specs: list[ToolSpec]) -> list:
    """ARM R. StructuredTool over the same specs."""
    from langchain_core.tools import StructuredTool

    tools = []
    for s in specs:
        tools.append(
            StructuredTool.from_function(
                func=s.fn,
                name=s.name,
                description=s.description,
                args_schema=_pydantic_model(s.name, s.schema),
                handle_tool_error=True,          # a bad arg must not kill the loop
                handle_validation_error=True,
            )
        )
    return tools


def build_crewai_tools(specs: list[ToolSpec]) -> list:
    """ARM C. crewai BaseTool subclasses over the same specs."""
    from crewai.tools import BaseTool

    tools = []
    for s in specs:
        args_model = _pydantic_model(s.name, s.schema)

        def _make(spec: ToolSpec, model):
            class _Tool(BaseTool):
                name: str = spec.name
                description: str = spec.description
                args_schema: type = model

                def _run(self, **kwargs: Any) -> str:
                    # Drop the Nones pydantic inserts for omitted optionals, so
                    # the service sees its own defaults rather than explicit
                    # nulls (`timeout_sec=None` would fail its type check).
                    clean = {k: v for k, v in kwargs.items() if v is not None}
                    return spec.fn(**clean)

            _Tool.__name__ = f"{spec.name}_Tool"
            return _Tool()

        tools.append(_make(s, args_model))
    return tools


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="regenerate tools.lock.json")
    ap.add_argument("--url", default="http://localhost:9999/mcp")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        store = OKRStore(Path(tmp) / "probe.db", tmp)
        with MCPClient(args.url) as client:
            specs = tool_specs(client, store)
            if args.write:
                payload = write_lock(specs)
                print(f"wrote {LOCKFILE.name}: {payload['count']} tools, digest {payload['digest'][:16]}")
            else:
                assert_lock(specs)
                print(f"ok: {len(specs)} tools match {LOCKFILE.name}")
            for s in specs:
                print(f"  {s.source:4s} {s.name:32s} {s.fingerprint()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
