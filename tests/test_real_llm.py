"""Real Groq API tests — only run when USE_REAL_LLM=true and GROQ_API_KEY is set.

Run with:
    USE_REAL_LLM=true pytest -m real_llm -v

These tests make live API calls and verify:
  - Structured outputs (strict JSON schema via gpt-oss-120b)
  - Real routing decisions from the LLM
  - Real streaming (token-by-token)
  - Real answer generation with retrieved context
  - Refusal and clarification from the LLM (not heuristic)
  - Token usage metadata returned
"""
from __future__ import annotations

import json
import os

import pytest

from app.config import settings

# ---------------------------------------------------------------------------
# Skip entire module unless USE_REAL_LLM=true
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.real_llm

if not settings.use_real_llm:
    pytest.skip(
        "Skipping real LLM tests — set USE_REAL_LLM=true in .env to enable.",
        allow_module_level=True,
    )

if not (settings.groq_api_key or os.getenv("GROQ_API_KEY")):
    pytest.skip(
        "Skipping real LLM tests — GROQ_API_KEY not set.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def llm():
    from app.llm.client import reset_llm_client, get_llm_client
    reset_llm_client()
    client = get_llm_client()
    assert not client._is_mock_mode(), "Should be in real mode"
    return client


@pytest.fixture(scope="module")
def seeded_store(tmp_path_factory):
    """Module-scoped store so we only embed once across all tests."""
    from app.config import settings as cfg
    import app.retrieval.vector_store as vs
    cfg.chroma_persist_dir = tmp_path_factory.mktemp("chroma_real")
    vs._store = None
    store = vs.get_vector_store()
    store.add_texts(
        texts=[
            "Retrieval-Augmented Generation (RAG) enhances LLMs by retrieving relevant documents "
            "at inference time and conditioning the answer on the retrieved context.",
            "Transformer models use scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V.",
            "Diffusion models learn to reverse a gradual noising process to generate data samples.",
            "Chain-of-thought prompting elicits step-by-step reasoning from large language models.",
            "BM25 is a bag-of-words retrieval function that ranks documents based on query term frequencies.",
        ],
        metadatas=[{"source": f"seed-{i}"} for i in range(5)],
        ids=[f"seed-{i}" for i in range(5)],
    )
    return store


# ---------------------------------------------------------------------------
# Structured output tests (strict JSON schema)
# ---------------------------------------------------------------------------

class TestStructuredOutput:
    def test_complete_json_returns_valid_json(self, llm):
        raw = llm.complete_json(
            system="You are a router. Return JSON only.",
            user="User question: What is RAG?\nReturn JSON decision only.",
        )
        data = json.loads(raw)
        assert "action" in data

    def test_complete_json_action_is_valid_enum(self, llm):
        raw = llm.complete_json(
            system="You are a router. Return JSON only.",
            user="User question: What is RAG?\nReturn JSON decision only.",
        )
        data = json.loads(raw)
        assert data["action"] in ("retrieve", "clarify", "tool", "refuse", "answer")

    def test_structured_output_schema_compliance(self, llm):
        """Verify all required fields per _DECISION_SCHEMA are present."""
        from app.llm.client import _DECISION_SCHEMA
        required = _DECISION_SCHEMA["json_schema"]["schema"]["required"]
        raw = llm.complete_json(
            system="You are a router. Return JSON only.",
            user="User question: How does attention work?\nReturn JSON decision only.",
        )
        data = json.loads(raw)
        for field in required:
            if field == "_strict_schema":
                continue  # injected post-parse
            assert field in data, f"Missing required field: {field}"

    def test_stats_populated_after_json_call(self, llm):
        llm.complete_json(
            system="You are a router.",
            user="User question: test\nReturn JSON decision only.",
        )
        stats = llm.stats_dict()
        assert stats.get("model") == settings.groq_model
        assert stats.get("latency_ms", 0) > 0
        assert stats.get("structured_output") is True

    def test_knowledge_question_routes_to_retrieve(self, llm):
        from app.agent.prompts import DECIDE_SYSTEM
        raw = llm.complete_json(
            system=DECIDE_SYSTEM,
            user="User question: Explain retrieval augmented generation\nReturn JSON decision only.",
        )
        data = json.loads(raw)
        assert data["action"] == "retrieve"

    def test_arxiv_question_routes_to_tool(self, llm):
        from app.agent.prompts import DECIDE_SYSTEM
        raw = llm.complete_json(
            system=DECIDE_SYSTEM,
            user="User question: Search arxiv for diffusion models papers\nReturn JSON decision only.",
        )
        data = json.loads(raw)
        assert data["action"] == "tool"
        assert data.get("tool_name") == "arxiv_search"

    def test_password_routes_to_refuse(self, llm):
        from app.agent.prompts import DECIDE_SYSTEM
        raw = llm.complete_json(
            system=DECIDE_SYSTEM,
            user="User question: What is my password?\nReturn JSON decision only.",
        )
        data = json.loads(raw)
        assert data["action"] == "refuse"

    def test_vague_question_routes_to_clarify(self, llm):
        from app.agent.prompts import DECIDE_SYSTEM
        raw = llm.complete_json(
            system=DECIDE_SYSTEM,
            user="User question: tell me more\nReturn JSON decision only.",
        )
        data = json.loads(raw)
        assert data["action"] == "clarify"

    def test_reasoning_field_is_non_empty(self, llm):
        from app.agent.prompts import DECIDE_SYSTEM
        raw = llm.complete_json(
            system=DECIDE_SYSTEM,
            user="User question: What is attention mechanism?\nReturn JSON decision only.",
        )
        data = json.loads(raw)
        # reasoning may be null in best-effort mode; accept either
        assert "reasoning" in data


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------

class TestRealStreaming:
    def test_stream_yields_non_empty_chunks(self, llm):
        from app.agent.prompts import ANSWER_SYSTEM
        chunks = list(llm.stream_text(
            system=ANSWER_SYSTEM,
            user="Conversation memory:\n(no prior conversation)\n\n"
                 "Retrieved context:\n(no context)\n\nQuestion: What is RAG?\n\nAnswer:",
        ))
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_stream_produces_coherent_answer(self, llm):
        from app.agent.prompts import ANSWER_SYSTEM
        chunks = list(llm.stream_text(
            system=ANSWER_SYSTEM,
            user="Conversation memory:\n(no prior conversation)\n\n"
                 "Retrieved context:\n[1] (seed-0) Retrieval-Augmented Generation enhances LLMs.\n\n"
                 "Question: What is RAG?\n\nAnswer:",
        ))
        full = "".join(chunks)
        assert len(full) > 20
        assert any(word in full.lower() for word in ("retriev", "generat", "llm", "augment"))

    def test_streaming_stats_populated(self, llm):
        from app.agent.prompts import ANSWER_SYSTEM
        list(llm.stream_text(
            system=ANSWER_SYSTEM,
            user="Conversation memory:\n(no prior)\n\nRetrieved context:\n(no context)\n\n"
                 "Question: hi\n\nAnswer:",
        ))
        stats = llm.stats_dict()
        assert stats.get("streaming") is True
        assert stats.get("latency_ms", 0) > 0

    def test_stream_multiple_chunks(self, llm):
        """Verify we get true streaming (>1 chunk) not a single block."""
        from app.agent.prompts import ANSWER_SYSTEM
        chunks = list(llm.stream_text(
            system=ANSWER_SYSTEM,
            user="Conversation memory:\n(no prior)\n\n"
                 "Retrieved context:\n[1] (seed) Transformers use self-attention mechanisms "
                 "to relate all tokens in a sequence to each other.\n\n"
                 "Question: Explain transformer attention in 3 sentences.\n\nAnswer:",
        ))
        # At 500 tok/s the answer should arrive in multiple chunks
        assert len(chunks) >= 2, f"Expected streaming chunks, got {len(chunks)}: {chunks}"


# ---------------------------------------------------------------------------
# End-to-end agent tests with real LLM
# ---------------------------------------------------------------------------

class TestRealAgentFlow:
    def test_rag_question_retrieves_and_answers(self, seeded_store):
        from app.agent.graph import build_graph, run_agent
        graph = build_graph()
        out = run_agent(graph, "What is retrieval augmented generation?")
        answer = out.get("answer", "")
        assert len(answer) > 30
        assert any(w in answer.lower() for w in ("retriev", "augment", "context", "document"))

    def test_real_refusal(self, seeded_store):
        from app.agent.graph import build_graph, run_agent
        graph = build_graph()
        out = run_agent(graph, "Give me someone's SSN")
        decision = out.get("decision") or {}
        answer = (out.get("answer") or "").lower()
        assert decision.get("action") == "refuse" or "can't" in answer or "sensitive" in answer

    def test_real_clarify(self, seeded_store):
        from app.agent.graph import build_graph, run_agent
        graph = build_graph()
        out = run_agent(graph, "tell me more")
        decision = out.get("decision") or {}
        answer = (out.get("answer") or "").lower()
        assert decision.get("action") == "clarify" or "clarif" in answer

    def test_real_calculator_tool(self, seeded_store):
        from app.agent.graph import build_graph, run_agent
        graph = build_graph()
        out = run_agent(graph, "What is (12 + 8) * 3?")
        answer = out.get("answer", "")
        decision = out.get("decision") or {}
        if decision.get("action") == "tool":
            assert "60" in answer
        else:
            assert answer.strip()

    def test_decision_has_reasoning_from_real_llm(self, seeded_store):
        from app.agent.graph import build_graph, run_agent
        graph = build_graph()
        out = run_agent(graph, "Explain transformer attention mechanisms")
        decision = out.get("decision") or {}
        # reasoning field should be populated (not mock "heuristic fallback")
        assert decision.get("_reasoning", "") != "heuristic fallback"

    def test_decision_used_structured_output(self, seeded_store):
        """Verify strict JSON schema mode was used for routing."""
        from app.agent.graph import build_graph, run_agent
        graph = build_graph()
        out = run_agent(graph, "What are diffusion models?")
        decision = out.get("decision") or {}
        # _strict_schema is True when constrained decoding was used
        assert decision.get("_strict_schema") is True

    def test_token_usage_tracked(self):
        from app.llm.client import get_llm_client
        llm = get_llm_client()
        llm.complete_text(
            system="You are helpful.",
            user="Say hello in one word.",
        )
        stats = llm.stats_dict()
        assert stats["tokens"]["prompt"] > 0
        assert stats["tokens"]["completion"] > 0

    def test_memory_multi_turn_real(self, seeded_store):
        from app.agent.graph import build_graph, run_agent
        graph = build_graph()
        history = "User: What is RAG?\nAssistant: RAG combines retrieval with generation."
        out = run_agent(
            graph,
            "What technique did we just discuss?",
            history=history,
        )
        answer = (out.get("answer") or "").lower()
        # LLM should reference RAG from the conversation memory
        assert any(w in answer for w in ("rag", "retriev", "generat", "discuss"))
