from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from AtomAgents.utils.routing import handoff_router
from AtomAgents.agents.simulation_manager import Supervisor
from AtomAgents.agents.model_building_agent import Modeler
from AtomAgents.agents.simulator_agent import SimulatorAgent
from AtomAgents.memory.sqlite_rag_memory import (
    SQLiteRAGMemory,
)
from AtomAgents.utils.logger import configure_agent_logger
from AtomAgents.workflows.state_definitions import State as GraphState
from AtomAgents.utils.prompt_loader import PromptLoader
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

agent_config_defaults = dict(
    model="gpt-5",  # nmd 太贵了
    temperature=0.9,
    max_tokens=10240,
    reasoning_effort="medium",  # "minimal" or "low" or "medium" or "high"
)


def _assemble_graph(
    project_root: Path,
    memory: SQLiteRAGMemory,
    tools: List[Any],
    llm_config: Dict[str, Any],
    *,
    prompt_loaders: Optional[Dict[str, PromptLoader]] = None,
):
    graph = StateGraph(state_schema=GraphState)

    sup_kwargs: Dict[str, Any] = dict(
        llm_config=llm_config,
        memory=memory,
        Collages=["Modeler", "Simulator", "END"],
        tools=tools,
        project_root=str(project_root.resolve()),
        enable_okr_table=True,
    )
    if prompt_loaders and prompt_loaders.get("Supervisor"):
        sup_kwargs["prompt_loader"] = prompt_loaders["Supervisor"]
    sup = Supervisor(**sup_kwargs)

    mod_kwargs: Dict[str, Any] = dict(
        llm_config=llm_config,
        memory=memory,
        Collages=["Supervisor", "Simulator", "END"],
        tools=tools,
        project_root=str(project_root.resolve()),
    )
    if prompt_loaders and prompt_loaders.get("Modeler"):
        mod_kwargs["prompt_loader"] = prompt_loaders["Modeler"]
    mod = Modeler(**mod_kwargs)

    sim_kwargs: Dict[str, Any] = dict(
        llm_config=llm_config,
        memory=memory,
        Collages=["Supervisor", "Modeler", "END"],
        tools=tools,
        config={"recursion_limit": 30},
        project_root=str(project_root.resolve()),
    )
    if prompt_loaders and prompt_loaders.get("Simulator"):
        sim_kwargs["prompt_loader"] = prompt_loaders["Simulator"]
    sim = SimulatorAgent(**sim_kwargs)

    graph.add_node("Supervisor", sup.run)
    graph.add_node("Modeler", mod.run)
    graph.add_node("Simulator", sim.run)
    graph.set_entry_point("Supervisor")

    graph.add_conditional_edges(
        "Supervisor",
        partial(handoff_router, agent="Supervisor"),
        {"Modeler": "Modeler", "Simulator": "Simulator", "END": END},
    )
    graph.add_conditional_edges(
        "Modeler",
        partial(handoff_router, agent="Modeler"),
        {"Supervisor": "Supervisor", "Simulator": "Simulator", "END": END},
    )
    graph.add_conditional_edges(
        "Simulator",
        partial(handoff_router, agent="Simulator"),
        {"Supervisor": "Supervisor", "Modeler": "Modeler", "END": END},
    )

    return graph.compile()


def _coerce_to_message(msg: Union[str, Dict[str, Any], BaseMessage]) -> BaseMessage:
    """Convert common message representations into a BaseMessage.

    Supports:
    - str → HumanMessage
    - {"role": "user"|"assistant", "content": "..."}
    - BaseMessage passthrough
    """
    if isinstance(msg, BaseMessage):
        return msg
    if isinstance(msg, str):
        return HumanMessage(content=msg)
    if isinstance(msg, dict):
        role = (msg.get("role") or msg.get("type") or "").lower()
        content = msg.get("content") or msg.get("text") or ""
        if role in ("user", "human"):
            return HumanMessage(content=content)
        if role in ("assistant", "ai", "system"):
            # treat system as AI-written context for storage
            return AIMessage(content=content)
    # Fallback
    return HumanMessage(content=str(msg))


def add_conversation_to_memory(
    memory: SQLiteRAGMemory,
    messages: Iterable[Union[str, Dict[str, Any], BaseMessage]],
    *,
    job_id: Optional[str] = None,
    task_id: Optional[int] = None,
    key_result_id: Optional[int] = None,
    agent_name: Optional[str] = None,
) -> int:
    """Append a conversation (sequence of messages) into existing memory.

    Returns number of messages stored.
    """
    count = 0
    for raw in messages:
        msg = _coerce_to_message(raw)
        # Use private method to preserve exact roles; save_context pairs I/O
        memory._add_message(  # noqa: SLF001
            msg,
            job_id=job_id,
            task_id=task_id,
            key_result_id=key_result_id,
            agent_name=agent_name,
        )
        count += 1
    return count


def AtomAgent_build(
    work_dir: str,
    memory_file: str = "memory.db",
    *,
    model_id: str = "atomagent",
):
    """Build the AtomAgents graph using an existing memory DB and return (graph, memory, project_root).

    - Ensures tools return direct responses.
    - Uses provided SQLite file for long‑term memory (created if missing).
    - Provides helper `add_conversation_to_memory` for appending past chats.
    """
    # Logging default: quiet
    configure_agent_logger(verbose=False)

    # Resolve paths
    project_root = Path(work_dir).resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    sqlite_path = project_root / memory_file

    # LLM / Embedding configuration chosen by model_id to align with frontend
    # - "atomagent"        → cloud/OpenAI-style (env overrideable)
    # - "atomagent_local"  → local OpenAI-compatible server (e.g., Ollama)

    model_norm = (model_id or "atomagent").lower()
    if model_norm == "atomagent":
        llm_config = agent_config_defaults
        embedding_model = None  # Use default OpenAI embedding
    elif model_norm == "fenicsagent":
        # Keep cloud/OpenAI-style defaults; only prompt templates differ
        llm_config = agent_config_defaults
        embedding_model = None
    else:
        # "atomagent_local" (and any other value) → local defaults
        from AtomAgents.config import LOCAL_LLM_MODEL, LOCAL_LLM_BASE_URL

        llm_config = dict(
            model=LOCAL_LLM_MODEL,
            temperature=1.0,
            max_tokens=16000,
            base_url=LOCAL_LLM_BASE_URL,
        )
        embedding_model = None
        # embedding_model = GPTEmbedding(
        #     model="text-embedding-granite-embedding-278m-multilingual",
        #     base_url="http://localhost:11434/v1/",
        # )
    memory = SQLiteRAGMemory(
        sqlite_path=str(sqlite_path.resolve()),
        project_root=str(project_root),
        embedding_model=embedding_model,
    )

    # MCP tools
    from AtomAgents.config import make_mcp_client

    client = make_mcp_client()
    tools = asyncio.run(client.get_tools())
    # NOTE: no tool.return_direct here — it is only honored by prebuilt ReAct
    # agents and was inert (and misleading) in this hand-rolled graph.

    # Configure prompt loaders per agent based on model
    prompt_loaders: Optional[Dict[str, PromptLoader]] = None
    if model_norm == "fenicsagent":
        root = Path(__file__).parent.parent
        fenics_dir = str((root / "AtomAgents" / "templates" / "fenicsx").resolve())
        prompt_loaders = {
            "Supervisor": PromptLoader(templates_dir=fenics_dir),
            "Simulator": PromptLoader(templates_dir=fenics_dir),
            "Modeler": PromptLoader(templates_dir=fenics_dir),
        }

    app_graph = _assemble_graph(
        project_root, memory, tools, llm_config, prompt_loaders=prompt_loaders
    )

    return app_graph, memory, project_root


# def run_graph_once(
#     prompt: str,
#     *,
#     work_dir: str,
#     memory_file: str = "memory.db",
#     model_id: str = "atomagent",
#     conversation: Optional[Iterable[Union[str, Dict[str, Any], BaseMessage]]] = None,
#     job_id: Optional[str] = None,
#     task_id: Optional[int] = None,
#     key_result_id: Optional[int] = None,
#     agent_name: Optional[str] = None,
#     recursion_limit: int = 100,
#     on_update: Optional[Callable[[str], None]] = None,
# ) -> Dict[str, Any]:
#     """Build the graph (if needed), optionally seed memory with conversation, and run once.

#     Returns a dict with the final assistant message, project_root, and bookkeeping.
#     """
#     app_graph, memory, project_root = AtomAgent_build(
#         work_dir=work_dir, memory_file=memory_file, model_id=model_id
#     )

#     added = 0
#     if conversation:
#         added = add_conversation_to_memory(
#             memory,
#             conversation,
#             job_id=job_id,
#             task_id=task_id,
#             key_result_id=key_result_id,
#             agent_name=agent_name,
#         )

#     input_msg = HumanMessage(content=prompt)
#     last_text = ""
#     updates: List[str] = []
#     for step in app_graph.stream(
#         input={"messages": [input_msg], "config": {"recursion_limit": recursion_limit}},
#         stream_mode="updates",
#     ):
#         for v in step.values():
#             latest_ai = [
#                 m for m in v.get("messages", []) if getattr(m, "type", "") == "ai"
#             ]
#             if latest_ai:
#                 current = latest_ai[-1].content
#                 if current != last_text:
#                     last_text = current
#                     updates.append(last_text)
#                     if on_update:
#                         try:
#                             on_update(last_text)
#                         except Exception:
#                             pass

#     return {
#         "message": last_text,
#         "project_root": str(project_root),
#         "added_messages": added,
#         "updates": updates,
#     }


# def run_graph_stream(
#     prompt: str,
#     *,
#     work_dir: str,
#     memory_file: str = "memory.db",
#     model_id: str = "atomagent",
#     conversation: Optional[Iterable[Union[str, Dict[str, Any], BaseMessage]]] = None,
#     job_id: Optional[str] = None,
#     task_id: Optional[int] = None,
#     key_result_id: Optional[int] = None,
#     agent_name: Optional[str] = None,
#     recursion_limit: int = 100,
# ) -> Iterator[Dict[str, Any]]:
#     """Stream continuous updates while running the graph.

#     Yields dicts of the shape {"message": <partial_text>, "project_root": <str>}
#     each time the assistant text grows.
#     """
#     app_graph, memory, project_root = AtomAgent_build(
#         work_dir=work_dir, memory_file=memory_file, model_id=model_id
#     )

#     if conversation:
#         add_conversation_to_memory(
#             memory,
#             conversation,
#             job_id=job_id,
#             task_id=task_id,
#             key_result_id=key_result_id,
#             agent_name=agent_name,
#         )
    

#     input_msg = HumanMessage(content=prompt)
#     last_text = ""
#     for step in app_graph.stream(
#         input={"messages": [input_msg], "config": {"recursion_limit": recursion_limit}},
#         stream_mode="updates",
#     ):
#         for v in step.values():
#             latest_ai = [
#                 m for m in v.get("messages", []) if getattr(m, "type", "") == "ai"
#             ]
#             if latest_ai:
#                 current = latest_ai[-1].content
#                 if current != last_text:
#                     last_text = current
#                     yield {"message": last_text, "project_root": str(project_root)}
