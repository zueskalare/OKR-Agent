"""Scripted fake LLM for agent-level tests (no network, no tokens).

The script is a list of responses consumed in order:

- ``AIMessage``          → returned by plain ``.invoke()``
- pydantic instance      → returned by ``.with_structured_output(S).invoke()``;
                           the test fails loudly if it isn't an ``S``
- ``Exception`` instance → raised by the next call (for failure-path tests)

Every call is recorded in ``.calls`` so tests can assert the CALL BUDGET —
the point of the triage work is how few LLM calls a turn costs.
"""

from typing import Any, List


class _Structured:
    def __init__(self, fake: "FakeLLM", schema):
        self.fake = fake
        self.schema = schema

    def invoke(self, messages, *args, **kwargs):
        self.fake.calls.append(("structured", self.schema.__name__))
        item = self.fake._next()
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, self.schema), (
            f"FakeLLM script drift: expected {self.schema.__name__}, "
            f"got {type(item).__name__}"
        )
        return item

    async def ainvoke(self, messages, *args, **kwargs):
        # Same ledger entries as the sync path so call-budget assertions
        # are path-agnostic.
        return self.invoke(messages, *args, **kwargs)


class FakeLLM:
    def __init__(self, script: List[Any] = ()):  # noqa: B006 - copied below
        self.script: List[Any] = list(script)
        self.calls: List[Any] = []
        self.bound_tools = None

    def _next(self):
        if not self.script:
            raise AssertionError(
                "FakeLLM script exhausted — an unexpected extra LLM call was made"
            )
        return self.script.pop(0)

    def invoke(self, messages, *args, **kwargs):
        self.calls.append(("invoke",))
        item = self._next()
        if isinstance(item, BaseException):
            raise item
        return item

    async def ainvoke(self, messages, *args, **kwargs):
        return self.invoke(messages, *args, **kwargs)

    async def astream(self, messages, *args, **kwargs):
        """Async generator yielding one AIMessageChunk per whitespace word of
        the next scripted AIMessage — for opt-in token tests."""
        from langchain_core.messages import AIMessageChunk

        self.calls.append(("astream",))
        item = self._next()
        if isinstance(item, BaseException):
            raise item
        words = str(getattr(item, "content", "") or "").split(" ")
        for i, w in enumerate(words):
            yield AIMessageChunk(content=(w if i == len(words) - 1 else w + " "))

    def with_structured_output(self, schema):
        return _Structured(self, schema)

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def with_config(self, *args, **kwargs):
        # Tagged prose call sites (token bridge) work against the fake.
        return self

    def bind(self, **kwargs):
        return self
