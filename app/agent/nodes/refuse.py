from __future__ import annotations

from app.agent.state import AgentState


def refuse(state: AgentState) -> dict:
    state.setdefault("trace", []).append("refuse")
    return {
        "answer": (
            "I can't help with that request. It looks like it involves sensitive "
            "credentials or content outside this assistant's scope. If you meant "
            "something different, please rephrase."
        )
    }
