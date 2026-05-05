from __future__ import annotations

import json
import re

from app.agent.prompts import DECIDE_SYSTEM
from app.agent.state import AgentDecision, AgentState
from app.llm.client import get_llm_client
from app.llm.schemas import Decision
from app.observability.logger import get_logger


_logger = get_logger("agent.decide")

_REFUSE_TRIGGERS = (
    "password",
    "credit card",
    "ssn",
    "social security",
    "api key",
    "private key",
)
_VAGUE_PHRASES = (
    "tell me more",
    "explain it",
    "expand on that",
    "what about that",
    "the second one",
)
_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "hiya",
    "yo",
    "sup",
}
_CALC_RE = re.compile(r"^[\s\d\.\+\-\*\/\(\)\^]+$")


def _heuristic_decision(question: str) -> AgentDecision:
    q = question.strip().lower()

    if not q or len(q) < 4:
        return {"action": "answer"} if q in _GREETINGS else {"action": "clarify"}

    if q in _GREETINGS:
        return {"action": "answer"}

    if any(word in q for word in _REFUSE_TRIGGERS):
        return {"action": "refuse"}

    if any(p in q for p in _VAGUE_PHRASES):
        return {"action": "clarify"}

    if "arxiv" in q or "find paper" in q or "recent papers" in q:
        return {
            "action": "tool",
            "tool_name": "arxiv_search",
            "tool_args": {"query": question},
        }

    # Pure arithmetic expression?
    expr_candidate = q.replace("**", "^")
    if _CALC_RE.match(expr_candidate) and any(c.isdigit() for c in q):
        return {
            "action": "tool",
            "tool_name": "calculator",
            "tool_args": {"expression": question},
        }

    return {"action": "retrieve", "query": question}


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    return t.strip()


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _heuristic_reasoning(question: str, decision: AgentDecision) -> str:
    q = question.strip().lower()
    action = (decision or {}).get("action")
    if not q:
        return "Empty input; ask for clarification."
    if q in _GREETINGS:
        return "Greeting; answer directly."
    if any(word in q for word in _REFUSE_TRIGGERS):
        return "Sensitive/credential trigger; refuse."
    if any(p in q for p in _VAGUE_PHRASES):
        return "Vague follow-up; ask for details."
    if "arxiv" in q or "find paper" in q or "recent papers" in q:
        return "Explicit arXiv request; use arxiv_search tool."
    expr_candidate = q.replace("**", "^")
    if _CALC_RE.match(expr_candidate) and any(c.isdigit() for c in q):
        return "Arithmetic expression detected; use calculator tool."
    if action == "retrieve":
        return "Default: knowledge question; retrieve context."
    return "Heuristic fallback."


def decide(state: AgentState) -> dict:
    question = state.get("question", "")
    history = (state.get("history") or "").strip()
    summary = (state.get("memory_summary") or "").strip()
    state.setdefault("trace", []).append("decide")

    llm = get_llm_client()

    memory_block = ""
    if summary and history and summary == history:
        summary = ""
    if summary:
        memory_block += f"\nEarlier-turn summary:\n{summary}"
    if history:
        memory_block += f"\nRecent turns:\n{history}"

    user_payload = (
        f"{memory_block}\n\nUser question: {question}\n\nReturn JSON decision only."
    ).strip()

    try:
        raw = llm.complete_json(system=DECIDE_SYSTEM, user=user_payload)
        cleaned = _strip_json_fences(raw)
        payload = _extract_json_object(cleaned) or cleaned
        data = json.loads(payload)
        decision = Decision.model_validate(data)
        agent_decision = decision.to_agent_decision()
        # Surface LLM reasoning + structured-output metadata for debug/UI
        agent_decision["_reasoning"] = data.get("reasoning") or ""
        agent_decision["_strict_schema"] = bool(data.get("_strict_schema", False))
        agent_decision["_mock"] = False
        _logger.info(
            "decide.llm action=%s strict_schema=%s reasoning=%r",
            agent_decision.get("action"),
            agent_decision.get("_strict_schema"),
            agent_decision.get("_reasoning", "")[:80],
        )
        return {"decision": agent_decision}
    except json.JSONDecodeError as exc:
        _logger.warning(
            "decide.invalid_json raw_prefix=%r error=%s",
            (locals().get("raw", "") or "")[:160],
            exc,
        )
        agent_decision = _heuristic_decision(question)
        agent_decision["_reasoning"] = _heuristic_reasoning(question, agent_decision)
        agent_decision["_strict_schema"] = False
        agent_decision["_mock"] = True
        return {"decision": agent_decision}
    except Exception as exc:
        agent_decision = _heuristic_decision(question)
        agent_decision["_reasoning"] = _heuristic_reasoning(question, agent_decision)
        agent_decision["_strict_schema"] = False
        agent_decision["_mock"] = True
        _logger.warning(
            "decide.fallback_heuristic action=%s reason=%s",
            agent_decision.get("action"),
            type(exc).__name__,
        )
        return {"decision": agent_decision}
