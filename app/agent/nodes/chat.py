from __future__ import annotations

from typing import Generator

from app.agent.prompts import CHAT_SYSTEM
from app.agent.state import AgentState
from app.llm.client import get_llm_client
from app.observability.logger import get_logger


_logger = get_logger("agent.chat")


def _format_memory(state: AgentState) -> str:
    summary = (state.get("memory_summary") or "").strip()
    history = (state.get("history") or "").strip()
    if summary and history and summary == history:
        summary = ""
    parts: list[str] = []
    if summary:
        parts.append(f"Summary of earlier turns:\n{summary}")
    if history:
        parts.append(f"Recent conversation:\n{history}")
    return "\n\n".join(parts) if parts else "(no prior conversation)"


def _build_user_prompt(state: AgentState) -> str:
    return (
        f"Conversation memory:\n{_format_memory(state)}\n\n"
        f"User: {state.get('question', '')}\n\n"
        f"Assistant:"
    )


def chat(state: AgentState) -> dict:
    state.setdefault("trace", []).append("chat")
    question = state.get("question", "")

    llm = get_llm_client()
    user_prompt = _build_user_prompt(state)

    try:
        text = llm.complete_text(system=CHAT_SYSTEM, user=user_prompt)
        _logger.info(
            "chat.ok question_len=%d response_len=%d",
            len(question),
            len(text or ""),
        )
        return {"answer": (text or "").strip() or "Hello! How can I help?"}
    except Exception as exc:
        _logger.error("chat.llm_failure error=%s", exc)
        return {"answer": "I'm unable to respond right now."}


def chat_stream(state: AgentState) -> Generator[str, None, None]:
    """Stream chat tokens, yielding each text chunk."""

    state.setdefault("trace", []).append("chat")
    llm = get_llm_client()
    user_prompt = _build_user_prompt(state)

    try:
        yield from llm.stream_text(system=CHAT_SYSTEM, user=user_prompt)
    except Exception as exc:
        _logger.error("chat.stream_failure error=%s", exc)
        yield "I'm unable to respond right now."
