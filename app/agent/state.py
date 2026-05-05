from __future__ import annotations

from typing import Any, Literal, TypedDict


Action = Literal["retrieve", "clarify", "tool", "refuse", "answer"]


class AgentDecision(TypedDict, total=False):
    action: Action
    query: str
    tool_name: str
    tool_args: dict[str, Any]


class RetrievedChunk(TypedDict):
    id: str
    text: str
    source: str
    score: float


class AgentState(TypedDict, total=False):
    question: str
    decision: AgentDecision
    documents: list[RetrievedChunk]
    answer: str
    trace: list[str]
    # Conversation memory: prior-turn transcript injected at graph entry.
    history: str
    # Rolling LLM-summarized memory of older turns.
    memory_summary: str
