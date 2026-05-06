from __future__ import annotations

from typing import Any

from app.agent.state import AgentState
from app.agent.nodes.answer import answer
from app.agent.nodes.chat import chat
from app.agent.nodes.clarify import clarify
from app.agent.nodes.decide import decide
from app.agent.nodes.refuse import refuse
from app.agent.nodes.retrieve import retrieve
from app.agent.nodes.tool import tool


def build_graph() -> Any:
    """Build a LangGraph graph if available, otherwise return None.

    The rest of the system is written so `run_agent` still works without LangGraph.
    """

    try:
        from langgraph.graph import END, START, StateGraph
    except Exception:
        return None

    graph = StateGraph(AgentState)

    graph.add_node("decide", decide)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_node("chat", chat)
    graph.add_node("clarify", clarify)
    graph.add_node("refuse", refuse)
    graph.add_node("tool", tool)

    graph.add_edge(START, "decide")

    def route(state: AgentState) -> str:
        return (state.get("decision") or {}).get("action", "answer")

    graph.add_conditional_edges(
        "decide",
        route,
        {
            "retrieve": "retrieve",
            "clarify": "clarify",
            "tool": "tool",
            "refuse": "refuse",
            "answer": "chat",
        },
    )

    graph.add_edge("retrieve", "answer")
    graph.add_edge("tool", "answer")

    graph.add_edge("answer", END)
    graph.add_edge("chat", END)
    graph.add_edge("clarify", END)
    graph.add_edge("refuse", END)

    return graph.compile()


def run_agent(
    compiled_graph: Any,
    question: str,
    history: str = "",
    memory_summary: str = "",
    semantic_memory: str = "",
) -> dict[str, Any]:
    initial: AgentState = {
        "question": question,
        "trace": [],
        "history": history,
        "memory_summary": memory_summary,
        "semantic_memory": semantic_memory,
    }

    if compiled_graph is not None:
        result = compiled_graph.invoke(initial)
        return dict(result)

    state: AgentState = dict(initial)
    state.update(decide(state))

    action = (state.get("decision") or {}).get("action", "answer")
    if action == "retrieve":
        state.update(retrieve(state))
        state.update(answer(state))
    elif action == "tool":
        state.update(tool(state))
        state.update(answer(state))
    elif action == "clarify":
        state.update(clarify(state))
    elif action == "refuse":
        state.update(refuse(state))
    else:
        state.update(chat(state))

    return dict(state)


def make_graph(config: Any | None = None) -> Any:
    """Factory for LangGraph CLI / Agent Server.

    The CLI can reference this as `./app/agent/graph.py:make_graph`.
    """

    _ = config
    return build_graph()
