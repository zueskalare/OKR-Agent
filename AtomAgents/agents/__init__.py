# The public symbols are imported lazily so that optional third-party
# dependencies that are required by some agents (e.g. ``langchain_openai``)
# do **not** break the whole package import.  This makes the codebase usable in
# minimal test environments where these heavy dependencies are missing.


def _safe_import(name: str):
    import importlib

    try:
        return importlib.import_module(name, __package__)
    except ModuleNotFoundError:
        return None


Modeler_mod = _safe_import(".model_building_agent")
Human_mod = _safe_import(".human_interput")
Supervisor_mod = _safe_import(".simulation_manager")
SimulatorAgent_mod = _safe_import(".simulator_agent")
ReportMessage_mod = _safe_import(".messages")

if Modeler_mod and hasattr(Modeler_mod, "Modeler"):
    Modeler = Modeler_mod.Modeler  # type: ignore
if Human_mod and hasattr(Human_mod, "human_node"):
    human_node = Human_mod.human_node  # type: ignore
if SimulatorAgent_mod and hasattr(SimulatorAgent_mod, "SimulatorAgent"):
    SimulatorAgent = SimulatorAgent_mod.SimulatorAgent  # type: ignore
if ReportMessage_mod and hasattr(ReportMessage_mod, "ReportMessage"):
    ReportMessage = ReportMessage_mod.ReportMessage  # type: ignore
if Supervisor_mod and hasattr(Supervisor_mod, "Supervisor"):
    Supervisor = Supervisor_mod.Supervisor  # type: ignore

__all__ = [
    name
    for name in (
        "Modeler",
        "human_node",
        "Supervisor",
        "SimulatorAgent",
        "ReportMessage",
    )
    if name in globals()
]

# ---------------------------------------------------------------------------
# Optional stubs for missing third-party test dependencies
# ---------------------------------------------------------------------------

import sys
from types import ModuleType
from contextlib import contextmanager

if "blockbuster" not in sys.modules:
    stub = ModuleType("blockbuster")

    class _BlockBuster:  # noqa: D401
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    @contextmanager
    def _blockbuster_ctx(*args, **kwargs):  # noqa: D401
        yield _BlockBuster()

    stub.BlockBuster = _BlockBuster  # type: ignore[attr-defined]
    stub.blockbuster_ctx = _blockbuster_ctx  # type: ignore[attr-defined]
    sys.modules["blockbuster"] = stub
