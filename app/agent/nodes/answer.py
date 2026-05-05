from __future__ import annotations

from typing import Generator

from app.agent.prompts import ANSWER_SYSTEM
from app.agent.state import AgentState
from app.llm.client import get_llm_client
from app.observability.logger import get_logger


_logger = get_logger("agent.answer")


def _format_context(state: AgentState) -> str:
    docs = state.get("documents") or []
    if not docs:
        return "(no context)"
    lines: list[str] = []
    chunk_number = 1
    for d in docs:
        src = d.get("source", "unknown")
        score = d.get("score")
        score_str = f" score={score:.2f}" if isinstance(score, (int, float)) else ""
        text = d.get("text", "")
        if str(src).startswith("tool:"):
            lines.append(f"[TOOL RESULT] ({src}{score_str}) {text}")
            continue
        lines.append(f"[{chunk_number}] ({src}{score_str}) {text}")
        chunk_number += 1
    return "\n\n".join(lines)


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
        f"Retrieved context:\n{_format_context(state)}\n\n"
        f"Question: {state.get('question', '')}\n\n"
        f"Answer:"
    )


def answer(state: AgentState) -> dict:
    state.setdefault("trace", []).append("answer")
    question = state.get("question", "")
    n_docs = len(state.get("documents") or [])
    llm = get_llm_client()
    user_prompt = _build_user_prompt(state)

    try:
        text = llm.complete_text(system=ANSWER_SYSTEM, user=user_prompt)
        _logger.info(
            "answer.ok question_len=%d n_docs=%d response_len=%d",
            len(question), n_docs, len(text or ""),
        )
        return {"answer": (text or "").strip() or "I don't know."}
    except Exception as exc:
        _logger.error("answer.llm_failure error=%s", exc)
        if n_docs == 0:
            return {"answer": "I don't have enough information to answer that confidently."}
        snippet = (state["documents"][0].get("text") or "")[:400]
        return {
            "answer": (
                "The language model is temporarily unavailable. Most relevant passage:\n\n"
                + snippet
            )
        }


def answer_stream(state: AgentState) -> Generator[str, None, None]:
    """Stream answer tokens, yielding each text chunk."""
    state.setdefault("trace", []).append("answer")
    llm = get_llm_client()
    user_prompt = _build_user_prompt(state)
    try:
        yield from llm.stream_text(system=ANSWER_SYSTEM, user=user_prompt)
    except Exception as exc:
        _logger.error("answer.stream_failure error=%s", exc)
        yield "I'm unable to generate a response right now."
