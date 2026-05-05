from __future__ import annotations

from app.agent.state import AgentState


def clarify(state: AgentState) -> dict:
    state.setdefault("trace", []).append("clarify")

    q = (state.get("question") or "").strip()
    if not q:
        return {"answer": "What would you like to ask?"}

    return {
        "answer": (
            "I need a bit more detail to answer well. Could you clarify the topic, "
            "the specific aspect you care about, or the context (e.g., which paper "
            "or which method)?"
        )
    }
