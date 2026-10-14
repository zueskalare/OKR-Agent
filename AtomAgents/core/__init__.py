def _safe_import(mod, name):
    try:
        module = __import__(mod, fromlist=[name])
        return getattr(module, name)
    except Exception:
        return None


lammps_runner = _safe_import("AtomAgents.core.lammps_runner", "lammps_runner")
file_reader = _safe_import("AtomAgents.core.tools", "file_reader")
from .simulation_agent_system import SimulationAgentSystem, build_simulation_graph  # always available here

__all__ = [
    n
    for n in (
        "lammps_runner",
        "file_reader",
        "SimulationAgentSystem",
        "build_simulation_graph",
    )
    if globals().get(n) is not None
]
