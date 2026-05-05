"""LLM client — Groq API with strict structured outputs and real streaming.

Model: openai/gpt-oss-120b
  - 500 tok/s, 131 072-token context
  - Supports strict JSON-schema structured outputs (constrained decoding)
  - Supports streaming (stream=True)
  - NOTE: streaming and structured outputs are mutually exclusive on Groq

USE_REAL_LLM=true  → always calls real Groq API, mocks disabled
USE_REAL_LLM=false → uses mock responses when api_key is absent (test-safe default)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Generator

from app.config import settings
from app.observability.logger import get_logger

# ---------------------------------------------------------------------------
# Strict JSON schema for the routing decision — used with gpt-oss-120b
# ---------------------------------------------------------------------------
_DECISION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "AgentDecision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["retrieve", "clarify", "tool", "refuse", "answer"],
                },
                "query": {"type": ["string", "null"]},
                "tool_name": {"type": ["string", "null"]},
                "tool_args": {
                    "type": "object",
                    "properties": {
                        "query": {"type": ["string", "null"]},
                        "expression": {"type": ["string", "null"]},
                    },
                    "required": ["query", "expression"],
                    "additionalProperties": False,
                },
                "reasoning": {"type": ["string", "null"]},
            },
            "required": ["action", "query", "tool_name", "tool_args", "reasoning"],
            "additionalProperties": False,
        },
    },
}


_logger = get_logger("llm.client")
_DECISION_MAX_TOKENS = 384


@dataclass
class LLMStats:
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    used_structured_output: bool = False
    used_streaming: bool = False


@dataclass
class LLMClient:
    api_key: str | None
    model: str
    # Tracks stats for the most recent call — exposed to UI/debug
    last_stats: LLMStats | None = field(default=None, repr=False)

    def _is_mock_mode(self) -> bool:
        """Mock mode: no key AND USE_REAL_LLM is false."""
        if settings.use_real_llm:
            if not self.api_key:
                raise RuntimeError(
                    "USE_REAL_LLM=true but GROQ_API_KEY is not set. "
                    "Add your key to .env or export GROQ_API_KEY=..."
                )
            return False
        return not bool(self.api_key)

    def _groq_client(self):
        try:
            from groq import Groq
        except ImportError as e:
            raise RuntimeError("groq package not installed. Run: pip install groq") from e
        return Groq(api_key=self.api_key)

    def _build_messages(self, system: str, user: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _annotated_json_payload(self, raw: str, used_strict: bool) -> str:
        candidate = (raw or "").strip()
        extracted = self._extract_json_object(candidate)
        for payload in (candidate, extracted):
            if not payload:
                continue
            try:
                data = json.loads(payload)
                data["_strict_schema"] = used_strict
                return json.dumps(data)
            except Exception:
                continue
        return candidate

    @staticmethod
    def _normalize_tool_args(tool_args: object) -> dict:
        if not isinstance(tool_args, dict):
            return {"query": None, "expression": None}
        return {
            "query": tool_args.get("query"),
            "expression": tool_args.get("expression"),
        }

    def _decision_from_question(self, question: str, reasoning: str = "heuristic fallback") -> dict:
        q = question.lower().strip()
        base: dict = {
            "query": None,
            "tool_name": None,
            "tool_args": {"query": None, "expression": None},
            "reasoning": reasoning,
            "_strict_schema": False,
        }

        if not q or q in {"tell me more", "explain it", "expand on that"}:
            return {**base, "action": "clarify"}
        if any(w in q for w in ("password", "credit card", "ssn", "social security")):
            return {**base, "action": "refuse"}
        if "arxiv" in q or "find paper" in q or "recent papers" in q:
            return {
                **base,
                "action": "tool",
                "tool_name": "arxiv_search",
                "tool_args": {"query": question, "expression": None},
                "reasoning": "user requested arxiv search",
            }
        if any(c.isdigit() for c in q) and any(op in q for op in ("+", "-", "*", "/", "**")):
            return {
                **base,
                "action": "tool",
                "tool_name": "calculator",
                "tool_args": {"query": None, "expression": question},
                "reasoning": "arithmetic expression detected",
            }
        if len(q) < 4 or q in {"hi", "hello", "hey"}:
            return {**base, "action": "answer"}
        return {
            **base,
            "action": "retrieve",
            "query": question,
            "reasoning": "knowledge question → retrieve",
        }

    @staticmethod
    def _extract_user_question(user: str) -> str:
        for line in user.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("user question:"):
                return stripped.split(":", 1)[1].strip()
        return user.strip()

    def _coerce_decision_json(self, raw: str, used_strict: bool, user: str) -> str:
        candidate = (raw or "").strip()
        extracted = self._extract_json_object(candidate)
        for payload in (candidate, extracted):
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except Exception:
                continue
            if isinstance(data, dict) and data.get("action") in ("retrieve", "clarify", "tool", "refuse", "answer"):
                normalized = {
                    "action": data["action"],
                    "query": data.get("query"),
                    "tool_name": data.get("tool_name"),
                    "tool_args": self._normalize_tool_args(data.get("tool_args")),
                    "reasoning": data.get("reasoning"),
                    "_strict_schema": used_strict,
                }
                return json.dumps(normalized)

        repaired = self._decision_from_question(
            self._extract_user_question(user),
            reasoning="heuristic schema repair",
        )
        _logger.warning(
            "complete_json.heuristic_repair used_strict=%s raw_prefix=%r",
            used_strict,
            candidate[:160],
        )
        return json.dumps(repaired)

    @staticmethod
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete_text(self, system: str, user: str) -> str:
        """Blocking text completion. Returns full response string."""
        if self._is_mock_mode():
            return self._mock_text(system, user)

        client = self._groq_client()
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(system, user),
            temperature=0.2,
            max_tokens=1024,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = resp.usage
        self.last_stats = LLMStats(
            model=self.model,
            latency_ms=round(latency_ms, 1),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )
        return (resp.choices[0].message.content or "").strip()

    def complete_json(self, system: str, user: str) -> str:
        """Structured-output JSON completion using strict JSON-schema mode.

        Uses Groq's constrained decoding (strict=True) with gpt-oss-120b.
        Falls back to best-effort json_object if the model doesn't support strict.
        """
        if self._is_mock_mode():
            return self._mock_json(system, user)

        client = self._groq_client()
        messages = self._build_messages(system, user)

        def _call_json(response_format: dict) -> tuple[object, float]:
            started = time.perf_counter()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=_DECISION_MAX_TOKENS,
                response_format=response_format,
            )
            return response, (time.perf_counter() - started) * 1000

        strict_error: Exception | None = None
        for attempt in range(2):
            try:
                resp, latency_ms = _call_json(_DECISION_SCHEMA)
                usage = resp.usage
                self.last_stats = LLMStats(
                    model=self.model,
                    latency_ms=round(latency_ms, 1),
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    used_structured_output=True,
                    used_streaming=False,
                )
                raw = (resp.choices[0].message.content or "").strip()
                annotated = self._coerce_decision_json(raw, used_strict=True, user=user)
                if self._extract_json_object(annotated) or annotated.startswith("{"):
                    return annotated
                _logger.warning(
                    "complete_json.strict_malformed attempt=%d raw_prefix=%r",
                    attempt + 1,
                    raw[:160],
                )
            except Exception as exc:
                strict_error = exc
                _logger.warning(
                    "complete_json.strict_failure attempt=%d error=%s",
                    attempt + 1,
                    exc,
                )

        # Best-effort fallback for providers/models that fail constrained decoding.
        resp, latency_ms = _call_json({"type": "json_object"})
        usage = resp.usage
        self.last_stats = LLMStats(
            model=self.model,
            latency_ms=round(latency_ms, 1),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            used_structured_output=True,
            used_streaming=False,
        )
        raw = (resp.choices[0].message.content or "").strip()
        annotated = self._coerce_decision_json(raw, used_strict=False, user=user)
        if self._extract_json_object(annotated) or annotated.startswith("{"):
            _logger.warning(
                "complete_json.best_effort_json strict_error=%s",
                type(strict_error).__name__ if strict_error else "strict_malformed",
            )
            return annotated

        _logger.error(
            "complete_json.unparseable strict_error=%s raw_prefix=%r",
            type(strict_error).__name__ if strict_error else "strict_malformed",
            raw[:160],
        )
        return raw

    def stream_text(self, system: str, user: str) -> Generator[str, None, None]:
        """Real Groq streaming — yields text chunks as they arrive.

        Falls back to yielding the full mock response as a single chunk
        so callers use the same generator interface in test mode.

        NOTE: Groq does not support structured outputs + streaming simultaneously,
        so streaming always uses plain text mode.
        """
        if self._is_mock_mode():
            yield self._mock_text(system, user)
            return

        client = self._groq_client()
        t0 = time.perf_counter()
        total_tokens = 0

        stream = client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(system, user),
            temperature=0.2,
            max_tokens=1024,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                total_tokens += 1
                yield delta

        latency_ms = (time.perf_counter() - t0) * 1000
        self.last_stats = LLMStats(
            model=self.model,
            latency_ms=round(latency_ms, 1),
            completion_tokens=total_tokens,
            used_streaming=True,
        )

    def stats_dict(self) -> dict:
        """Return last call stats for debug display in the UI."""
        if not self.last_stats:
            return {}
        s = self.last_stats
        return {
            "model": s.model,
            "latency_ms": s.latency_ms,
            "tokens": {
                "prompt": s.prompt_tokens,
                "completion": s.completion_tokens,
                "total": s.total_tokens,
            },
            "structured_output": s.used_structured_output,
            "streaming": s.used_streaming,
        }

    # ------------------------------------------------------------------
    # Mock responses — only used when _is_mock_mode() is True
    # ------------------------------------------------------------------

    def _mock_text(self, system: str, user: str) -> str:
        if "Retrieved context:" in user:
            if "(no context)" in user:
                return (
                    "I didn't retrieve any documents from the local index. "
                    "Try running ingestion first: python scripts/run_ingestion.py"
                )
            return (
                "Based on the retrieved context, here is a concise answer "
                "drawn from the provided passages. [1]"
            )
        return "Hello! I'm the Agentic RAG assistant. Ask me anything about cs.AI papers."

    def _mock_json(self, system: str, user: str) -> str:
        question = self._extract_user_question(user)
        return json.dumps(self._decision_from_question(question, reasoning="mock"))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is not None:
        return _client
    api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
    _client = LLMClient(api_key=api_key, model=settings.groq_model)
    return _client


def reset_llm_client() -> None:
    global _client
    _client = None
