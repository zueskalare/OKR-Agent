"""A dependency-light streamable-HTTP MCP client, usable from every arm's venv.

Why this exists rather than each framework's own MCP integration:

* The service is streamable-HTTP only -- `AtomAgents/mcp_service/mcp_service.py`
  ends with `app = mcp.streamable_http_app()`, and exposes no stdio or SSE
  endpoint. `crewai_tools.MCPServerAdapter` (0.75.0) speaks stdio and SSE only,
  so CrewAI's native MCP path cannot reach this server at all.
* Even if it could, routing each arm through a different client would confound
  the experiment: a CrewAI deficit caused by `crewai_tools`' transport support
  is not a finding about agent architecture. One client, one tool surface, and
  the frameworks differ only where we intend them to.

Requires httpx and nothing else, which both the repo venv (3.11) and the CrewAI
venv (3.12) already have. Synchronous on purpose -- CrewAI tools are sync, and
the async-tool-called-synchronously mismatch is exactly the bug that makes the
AG2 scaffold silently bind zero tools.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Optional

import httpx

DEFAULT_URL = "http://localhost:9999/mcp"
PROTOCOL_VERSION = "2025-06-18"

# Long enough for a blocking `lammps_run`: the tool itself caps at 3000 s, and a
# client timeout below that would report a failure for a simulation that is
# still running -- turning a successful arm into a false negative.
DEFAULT_TIMEOUT = 3600.0


class MCPError(RuntimeError):
    pass


def _parse(resp: httpx.Response) -> Optional[dict]:
    """Streamable HTTP replies either as JSON or as an SSE stream.

    The stream may carry server notifications (logging, progress) BEFORE the
    JSON-RPC response; those have neither `result` nor `error` and must be
    skipped, not returned — returning one would hand the caller None in place
    of the tool result.
    """
    ctype = resp.headers.get("content-type", "")
    if "event-stream" in ctype:
        for line in resp.text.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                payload = json.loads(line[5:].strip())
            except ValueError:
                continue
            if isinstance(payload, dict) and ("result" in payload or "error" in payload):
                return payload
        return None
    if not resp.text.strip():
        return None
    return resp.json()


class MCPClient:
    """One session against the MCP service.

    Thread-safe: a lock serialises requests so two agent threads (CrewAI runs
    tools from its own executor) cannot interleave on one HTTP session.
    """

    def __init__(self, url: str = DEFAULT_URL, timeout: float = DEFAULT_TIMEOUT):
        self.url = url.rstrip("/")
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)
        self._session_id: Optional[str] = None
        self._id = 0
        self._lock = threading.Lock()
        self.server_info: dict[str, Any] = {}

    # -- plumbing ----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            h["Mcp-Session-Id"] = self._session_id
        return h

    def _rpc(self, method: str, params: Optional[dict] = None, notify: bool = False) -> Any:
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            body["params"] = params
        if not notify:
            self._id += 1
            body["id"] = self._id

        resp = self._client.post(self.url, headers=self._headers(), json=body)
        if resp.status_code >= 400:
            raise MCPError(f"{method} -> HTTP {resp.status_code}: {resp.text[:300]}")
        if notify:
            return None

        sid = resp.headers.get("mcp-session-id")
        if sid:
            self._session_id = sid

        payload = _parse(resp)
        if payload is None:
            raise MCPError(f"{method} -> empty response")
        if "error" in payload:
            raise MCPError(f"{method} -> {payload['error']}")
        return payload.get("result")

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> "MCPClient":
        result = self._rpc(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "atomagents-benchmark", "version": "1"},
            },
        )
        self.server_info = result.get("serverInfo", {})
        self._rpc("notifications/initialized", notify=True)
        return self

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "MCPClient":
        return self.connect()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- API ---------------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._rpc("tools/list")["tools"]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool and flatten the content blocks to a single string.

        A transport failure is returned as a JSON error string rather than
        raised: an exception escaping a tool kills the agent loop, which would
        score as an architecture failure instead of the infrastructure blip it
        is. The string is shaped like the tools' own error results so the model
        can react to it.
        """
        try:
            with self._lock:
                try:
                    result = self._rpc("tools/call", {"name": name, "arguments": arguments})
                except MCPError as exc:
                    # 404 on an established session means the server lost it
                    # (restarted). One re-initialize, one retry: a campaign
                    # must not have every remaining tool call poisoned by a
                    # service bounce.
                    if "HTTP 404" not in str(exc) or not self._session_id:
                        raise
                    self._session_id = None
                    self.connect()
                    result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"status": "error", "error": f"MCP transport: {exc}"})

        if not isinstance(result, dict):
            return json.dumps(
                {"status": "error", "error": f"MCP transport: unexpected response {result!r}"}
            )
        if result.get("isError"):
            blocks = result.get("content") or []
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            return json.dumps({"status": "error", "error": text or "tool reported an error"})

        blocks = result.get("content") or []
        parts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        if parts:
            return "\n".join(parts)
        # Structured-only result: hand back the raw payload rather than "".
        return json.dumps(result.get("structuredContent", result))
