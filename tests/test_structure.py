"""Structural import checks with REAL assertions.

The previous version wrapped every import in try/except-print, so the suite
stayed green even when the whole backend failed to import.
"""


def test_agents_package_exports():
    from AtomAgents.agents import Modeler, SimulatorAgent, Supervisor, ReportMessage

    assert Modeler.__name__ == "Modeler"
    assert SimulatorAgent.__name__ == "SimulatorAgent"
    assert Supervisor.__name__ == "Supervisor"
    assert ReportMessage is not None


def test_core_modules_import():
    import AtomAgents.config
    import AtomAgents.core.simulation_agent_system
    import AtomAgents.memory.sqlite_rag_memory
    import AtomAgents.utils.okr_tools
    import AtomAgents.utils.readme_sync
    import AtomAgents.utils.run_validation

    assert AtomAgents.config.MCP_URL.startswith("http")


def test_backend_imports():
    import backend.openai_backend as ob

    assert hasattr(ob, "app")
    assert hasattr(ob, "_assemble_graph")
    # the reduced schema shim must stay deleted
    assert not hasattr(ob, "_ensure_memory_schema")


def test_real_langchain_openai_not_shadowed():
    import inspect

    import langchain_openai

    src = inspect.getsourcefile(langchain_openai)
    assert src and "site-packages" in src, f"stubbed langchain_openai active: {src}"


def test_mcp_service_tools_importable():
    import importlib.util
    from pathlib import Path

    tools_path = (
        Path(__file__).parent.parent / "AtomAgents" / "mcp_service" / "tools.py"
    )
    spec = importlib.util.spec_from_file_location("tools_struct_check", tools_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in (
        "mcp_file_reader",
        "mcp_write_file",
        "mcp_download_file",
        "mcp_lammps_runner",
        "mcp_bash_runner",
        "mcp_code_runner",
    ):
        assert callable(getattr(mod, name)), name


def test_prompts_with_braces_do_not_break_formatting():
    """Tool descriptions contain JSON braces; prompts must not be parsed as
    f-string templates (caught live: 'Invalid variable name' on every turn)."""
    from langchain_core.messages import AIMessage, HumanMessage

    from AtomAgents.agents import Modeler
    from langchain_core.tools import tool

    @tool
    def braces_tool(x: str) -> str:
        """Returns {"error": ...} or {"path": [0]} on failure."""
        return x

    class FakeStructured:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, msgs, *args, **kwargs):
            fields = self.schema.model_fields
            kwargs = {}
            if "plan" in fields:
                kwargs["plan"] = "p"
            if "send_to" in fields:
                args = getattr(fields["send_to"].annotation, "__args__", ("report",))
                kwargs["send_to"] = "report" if "report" in args else args[0]
            if "report" in fields:
                kwargs["report"] = "r"
            return self.schema(**kwargs)

    class FakeLLM:
        def invoke(self, msgs, *args, **kwargs):
            return AIMessage(content="ok")

        def bind_tools(self, tools):
            return self

        def with_structured_output(self, schema):
            return FakeStructured(schema)

    import tempfile

    d = tempfile.mkdtemp()
    m = Modeler(
        llm=FakeLLM(), memory=None, Collages=["Supervisor", "END"],
        tools=[braces_tool], project_root=d,
    )
    result = m.run({"messages": [HumanMessage(content='payload with {"json": [1]} braces')]})
    assert result["send_to"] in ("Supervisor", "END")


def test_act_cycle_budget_forces_termination(tmp_path, monkeypatch):
    """Live-smoke regression: varying tool calls looped past the recursion
    limit (44 okr_update calls in one run). Past the act budget, tools are
    withheld and the run must reach report."""
    monkeypatch.setenv("ATOMAGENTS_MAX_ACT_CYCLES", "3")
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool

    from AtomAgents.agents import Modeler

    @tool
    def noisy_tool(x: str) -> str:
        """A tool the fake model wants to call forever."""
        return "ok"

    class FakeStructured:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, msgs, *args, **kwargs):
            fields = self.schema.model_fields
            kwargs = {}
            if "plan" in fields:
                kwargs["plan"] = "keep going"
            if "send_to" in fields:
                args = getattr(fields["send_to"].annotation, "__args__", ("act",))
                kwargs["send_to"] = "act" if "act" in args else args[0]
            if "report" in fields:
                kwargs["report"] = "done"
                sargs = getattr(fields["send_to"].annotation, "__args__", ("END",))
                kwargs["send_to"] = sargs[0]
            return self.schema(**kwargs)

    class LoopingLLM:
        """Calls the tool with DIFFERENT args every act — evades the
        identical-retry guard; only the budget can stop it."""

        def __init__(self):
            self.n = 0
            self.tools_bound = False

        def invoke(self, msgs, *args, **kwargs):
            if self.tools_bound:
                self.n += 1
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "noisy_tool", "args": {"x": f"call-{self.n}"}, "id": f"t{self.n}"}],
                )
            return AIMessage(content="final summary without tools")

        def bind_tools(self, tools):
            import copy

            bound = copy.copy(self)
            bound.tools_bound = True
            return bound

        def with_structured_output(self, schema):
            return FakeStructured(schema)

    m = Modeler(
        llm=LoopingLLM(), memory=None, Collages=["Supervisor", "END"],
        tools=[noisy_tool], project_root=str(tmp_path),
        config={"recursion_limit": 60},
    )
    result = m.run({"messages": [HumanMessage(content="loop forever please")]})
    # Terminated by the budget (not the recursion limit), ending in a report
    assert result["send_to"] in ("Supervisor", "END")
    assert any(type(x).__name__ == "ReportMessage" for x in result["messages"])
