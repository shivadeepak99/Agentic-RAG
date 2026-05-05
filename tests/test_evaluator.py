"""Tests for the evaluation harness."""
from __future__ import annotations

from app.eval.dataset import DATASET
from app.eval.evaluator import (
    aggregate,
    run_eval,
    score_answer,
    _content_score,
    _matches_any,
    REFUSAL_MARKERS,
    CLARIFY_MARKERS,
    DONT_KNOW_MARKERS,
)


class TestScoringFunctions:
    def test_all_keywords_present_scores_1(self):
        assert _content_score("retrieval and context used here", ["retrieval", "context"]) == 1.0

    def test_no_keywords_present_scores_0(self):
        assert _content_score("completely unrelated answer", ["retrieval", "context"]) == 0.0

    def test_partial_match_scores_half(self):
        score = _content_score("only retrieval mentioned", ["retrieval", "context"])
        assert score == 0.5

    def test_empty_expected_scores_1(self):
        assert _content_score("anything", []) == 1.0

    def test_case_insensitive_match(self):
        assert _content_score("RETRIEVAL is great", ["retrieval"]) == 1.0

    def test_matches_any_refusal(self):
        assert _matches_any("Sorry, I can't help with that.", REFUSAL_MARKERS)

    def test_matches_any_clarify(self):
        assert _matches_any("Could you clarify your question?", CLARIFY_MARKERS)

    def test_matches_any_dont_know(self):
        assert _matches_any("I don't know the answer.", DONT_KNOW_MARKERS)


class TestDataset:
    def test_dataset_has_at_least_10_cases(self):
        assert len(DATASET) >= 10

    def test_each_case_has_id_and_question(self):
        for row in DATASET:
            assert "id" in row, f"Missing 'id' in {row}"
            assert "question" in row, f"Missing 'question' in {row}"

    def test_dataset_has_refusal_cases(self):
        refusals = [r for r in DATASET if r.get("kind") == "refuse"]
        assert len(refusals) >= 2, "Assignment requires at least 2 refusal/clarification cases"

    def test_dataset_has_clarify_cases(self):
        clarifies = [r for r in DATASET if r.get("kind") == "clarify"]
        assert len(clarifies) >= 2

    def test_dataset_has_tool_cases(self):
        tools = [r for r in DATASET if r.get("kind") in ("tool", "math")]
        assert len(tools) >= 1

    def test_dataset_has_knowledge_cases(self):
        knowledge = [r for r in DATASET if r.get("kind") == "knowledge"]
        assert len(knowledge) >= 2

    def test_all_kinds_are_valid(self):
        valid = {"knowledge", "tool", "math", "clarify", "refuse", "ood", "smalltalk"}
        for row in DATASET:
            kind = row.get("kind")
            if kind is not None:
                assert kind in valid, f"Unknown kind {kind!r} in row {row['id']}"


class TestAggregate:
    def test_empty_results_returns_zeros(self):
        result = aggregate([])
        assert result["n"] == 0
        assert result["avg_score"] == 0.0

    def test_all_perfect_scores(self):
        from app.eval.evaluator import EvalResult
        results = [
            EvalResult("q1", "q", "a", 1.0, expected_action="retrieve", actual_action="retrieve", action_match=True),
            EvalResult("q2", "q", "a", 1.0, expected_action="clarify", actual_action="clarify", action_match=True),
        ]
        agg = aggregate(results)
        assert agg["avg_score"] == 1.0
        assert agg["action_accuracy"] == 1.0

    def test_mixed_scores(self):
        from app.eval.evaluator import EvalResult
        results = [
            EvalResult("q1", "q", "a", 1.0, expected_action="retrieve", actual_action="retrieve", action_match=True),
            EvalResult("q2", "q", "a", 0.0, expected_action="clarify", actual_action="refuse", action_match=False),
        ]
        agg = aggregate(results)
        assert agg["avg_score"] == 0.5
        assert agg["action_accuracy"] == 0.5
