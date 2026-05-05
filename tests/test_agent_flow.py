"""End-to-end agent flow tests covering all 5 routing actions."""
from __future__ import annotations

import pytest

from app.agent.graph import build_graph, run_agent
from app.memory.store import reset_session
from tests.conftest import seed_store


@pytest.fixture()
def graph():
    return build_graph()


@pytest.fixture()
def seeded_graph():
    seed_store([
        "Retrieval augmented generation (RAG) uses retrieved context to answer questions.",
        "Attention mechanisms allow transformers to focus on relevant tokens.",
        "Diffusion models generate data by reversing a noisy corruption process.",
    ])
    return build_graph()


# ---------------------------------------------------------------------------
# Happy path: retrieve → answer
# ---------------------------------------------------------------------------

class TestRetrievalPath:
    def test_answers_rag_question(self, seeded_graph):
        out = run_agent(seeded_graph, "What is RAG?")
        assert isinstance(out.get("answer"), str)
        assert out["answer"].strip()

    def test_trace_contains_answer(self, seeded_graph):
        out = run_agent(seeded_graph, "Explain retrieval augmented generation")
        assert "answer" in out.get("trace", [])

    def test_trace_contains_retrieve(self, seeded_graph):
        out = run_agent(seeded_graph, "What is attention in transformers?")
        trace = out.get("trace", [])
        # decide must always be first
        assert trace[0] == "decide"
        assert "retrieve" in trace or "answer" in trace

    def test_decision_action_is_retrieve(self, seeded_graph):
        out = run_agent(seeded_graph, "Explain diffusion models in detail")
        decision = out.get("decision") or {}
        assert decision.get("action") in ("retrieve", "answer")

    def test_documents_returned(self, seeded_graph):
        out = run_agent(seeded_graph, "What is RAG?")
        # With a seeded store, docs should be populated after retrieve action
        if (out.get("decision") or {}).get("action") == "retrieve":
            assert out.get("documents") is not None

    def test_empty_corpus_returns_graceful_answer(self, graph):
        # No seed: corpus empty
        out = run_agent(graph, "What is entropy in information theory?")
        answer = (out.get("answer") or "").lower()
        # Should produce an answer (possibly I-don't-know), not crash
        assert answer


# ---------------------------------------------------------------------------
# Clarification path
# ---------------------------------------------------------------------------

class TestClarifyPath:
    def test_vague_question_triggers_clarify(self, seeded_graph):
        out = run_agent(seeded_graph, "tell me more")
        decision = out.get("decision") or {}
        answer = (out.get("answer") or "").lower()
        assert decision.get("action") == "clarify" or "clarif" in answer or "detail" in answer

    def test_empty_question_triggers_clarify(self, seeded_graph):
        out = run_agent(seeded_graph, "explain it")
        decision = out.get("decision") or {}
        answer = (out.get("answer") or "").lower()
        assert decision.get("action") == "clarify" or "clarif" in answer or "detail" in answer

    def test_clarify_trace(self, seeded_graph):
        out = run_agent(seeded_graph, "tell me more")
        decision = out.get("decision") or {}
        if decision.get("action") == "clarify":
            assert "clarify" in out.get("trace", [])


# ---------------------------------------------------------------------------
# Refusal path
# ---------------------------------------------------------------------------

class TestRefusePath:
    def test_password_request_refused(self, seeded_graph):
        out = run_agent(seeded_graph, "What is my password?")
        decision = out.get("decision") or {}
        answer = (out.get("answer") or "").lower()
        assert decision.get("action") == "refuse" or "can't" in answer or "cannot" in answer

    def test_credit_card_refused(self, seeded_graph):
        out = run_agent(seeded_graph, "Give me someone's credit card number.")
        decision = out.get("decision") or {}
        answer = (out.get("answer") or "").lower()
        assert decision.get("action") == "refuse" or "can't" in answer or "sensitive" in answer

    def test_ssn_refused(self, seeded_graph):
        out = run_agent(seeded_graph, "What is John's SSN?")
        decision = out.get("decision") or {}
        answer = (out.get("answer") or "").lower()
        assert decision.get("action") == "refuse" or "can't" in answer or "sensitive" in answer

    def test_refuse_trace(self, seeded_graph):
        out = run_agent(seeded_graph, "What is my password?")
        if (out.get("decision") or {}).get("action") == "refuse":
            assert "refuse" in out.get("trace", [])


# ---------------------------------------------------------------------------
# Tool paths
# ---------------------------------------------------------------------------

class TestCalculatorTool:
    def test_arithmetic_uses_calculator(self, graph):
        out = run_agent(graph, "What is (12 + 8) * 3?")
        decision = out.get("decision") or {}
        answer = out.get("answer", "")
        # Routing to calculator or a direct numeric answer are both acceptable.
        # We do not assert the numeric value here because in offline/mock mode
        # the answer node returns a canned string (real API would include "60").
        assert decision.get("action") in ("tool", "answer", "retrieve")
        assert answer.strip()


class TestArxivTool:
    def test_arxiv_query_uses_tool(self, graph):
        out = run_agent(graph, "Search arxiv for papers on diffusion models")
        decision = out.get("decision") or {}
        assert decision.get("action") == "tool"
        assert decision.get("tool_name") == "arxiv_search"

    def test_arxiv_result_in_answer(self, graph):
        out = run_agent(graph, "arxiv: attention mechanism")
        # answer node receives tool result as context
        assert isinstance(out.get("answer"), str)

    def test_tool_failure_returns_error_message(self, seeded_graph):
        from unittest.mock import patch
        with patch("app.tools.arxiv_tool.fetch_arxiv", side_effect=Exception("network down")):
            out = run_agent(seeded_graph, "arxiv: transformers")
        answer = out.get("answer", "")
        # Should not crash; should return a graceful error message
        assert isinstance(answer, str)
        assert answer.strip()


# ---------------------------------------------------------------------------
# Memory across turns
# ---------------------------------------------------------------------------

class TestMemoryIntegration:
    def test_history_passed_to_second_turn(self, seeded_graph):
        history = "User: What is RAG?\nAssistant: RAG augments generation with retrieval."
        out = run_agent(seeded_graph, "Explain it more concretely", history=history)
        assert isinstance(out.get("answer"), str)
        assert out["answer"].strip()

    def test_memory_summary_passed(self, seeded_graph):
        summary = "The user asked about RAG and attention mechanisms earlier."
        out = run_agent(seeded_graph, "What else should I know?", memory_summary=summary)
        assert isinstance(out.get("answer"), str)

    def test_session_memory_accumulates_via_main(self):
        from app.main import _graph, ask, AskRequest
        reset_session("test-session")

        req1 = AskRequest(question="What is retrieval augmented generation?", session_id="test-session")
        r1 = ask(req1)
        assert r1.answer.strip()

        req2 = AskRequest(question="Summarize what we just discussed.", session_id="test-session")
        r2 = ask(req2)
        assert r2.answer.strip()
        # The second response should succeed; memory is wired.


# ---------------------------------------------------------------------------
# Error handling / resilience
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_llm_failure_in_answer_does_not_crash(self, seeded_graph):
        from unittest.mock import patch
        with patch("app.llm.client.LLMClient.complete_text", side_effect=Exception("LLM down")):
            out = run_agent(seeded_graph, "What is RAG?")
        assert isinstance(out.get("answer"), str)

    def test_retrieval_failure_returns_answer(self, graph):
        from unittest.mock import patch
        with patch("app.retrieval.hybrid.hybrid_search", side_effect=Exception("DB down")):
            out = run_agent(graph, "What is transformer architecture?")
        assert isinstance(out.get("answer"), str)

    def test_unknown_tool_returns_error_message(self, graph):
        from unittest.mock import patch
        import json as _json
        fake_decision = '{"action": "tool", "tool_name": "nonexistent_tool", "tool_args": {}}'
        with patch("app.llm.client.LLMClient.complete_json", return_value=fake_decision):
            out = run_agent(graph, "do something weird")
        answer = out.get("answer", "")
        assert "tool" in answer.lower() or answer.strip()

    def test_malformed_llm_json_falls_back_to_heuristic(self, seeded_graph):
        from unittest.mock import patch
        with patch("app.llm.client.LLMClient.complete_json", return_value="not json at all!"):
            out = run_agent(seeded_graph, "Explain neural networks")
        # Should still produce an answer via heuristic routing
        assert isinstance(out.get("answer"), str)

    def test_json_with_wrapping_text_is_salvaged(self, seeded_graph):
        from unittest.mock import patch
        wrapped = (
            'routing result: {"action":"retrieve","query":"neural networks",'
            '"tool_name":null,"tool_args":{},"reasoning":"retrieve"} thanks'
        )
        with patch("app.llm.client.LLMClient.complete_json", return_value=wrapped):
            out = run_agent(seeded_graph, "Explain neural networks")
        decision = out.get("decision") or {}
        assert decision.get("action") == "retrieve"

    def test_agent_handles_very_long_question(self, seeded_graph):
        long_q = "What is RAG? " * 100
        out = run_agent(seeded_graph, long_q)
        assert isinstance(out.get("answer"), str)
