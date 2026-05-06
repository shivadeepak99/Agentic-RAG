"""Tests for the evaluation harness."""
from __future__ import annotations

from app.eval.dataset import DATASET
from app.eval.evaluator import (
    CLARIFY_FAMILIES,
    REFUSAL_FAMILIES,
    UNCERTAINTY_FAMILIES,
    EvalResult,
    _behavior_score,
    _content_score,
    _matches_family,
    _uncertainty_score,
    aggregate,
    score_answer,
)


class TestScoringFunctions:
    def test_content_score_matches_phrase_families(self):
        score = _content_score(
            "RAG retrieves external documents and uses that context during answer generation.",
            [
                ["retrieval", "retrieves"],
                ["documents", "context"],
                ["generation", "answer"],
            ],
        )
        assert score == 1.0

    def test_content_score_allows_partial_semantic_hits(self):
        score = _content_score(
            "It retrieves documents before the model responds.",
            [
                ["retrieval", "retrieves"],
                ["documents", "context"],
                ["generation", "answer"],
            ],
        )
        assert score == 2 / 3

    def test_content_score_returns_none_for_non_content_rows(self):
        assert _content_score("anything", []) is None

    def test_score_answer_keeps_backward_compatible_flat_lists(self):
        assert score_answer("retrieval uses context", ["retrieval", "context"]) == 1.0

    def test_matches_family_is_case_insensitive(self):
        assert _matches_family("Sorry, I CANNOT HELP with that.", REFUSAL_FAMILIES[0])

    def test_behavior_score_for_clarify_accepts_question_like_response(self):
        score, notes = _behavior_score(
            {"kind": "clarify"},
            "Which part do you want me to go deeper on?",
            {"action": "answer"},
        )
        assert score == 1.0
        assert notes == []

    def test_uncertainty_score_for_ood_accepts_grounded_unknown(self):
        score, notes = _uncertainty_score(
            {"kind": "ood"},
            "I don't know based on available documents. The retrieved papers are about AI, not sports.",
        )
        assert score == 1.0
        assert notes == []

    def test_uncertainty_score_is_none_for_non_ood(self):
        score, notes = _uncertainty_score({"kind": "knowledge"}, "normal answer")
        assert score is None
        assert notes == []

    def test_phrase_families_cover_refusal_clarify_and_uncertainty(self):
        assert any("can't help" in phrase for phrase in REFUSAL_FAMILIES[0])
        assert any("clarify" in phrase for phrase in CLARIFY_FAMILIES[0])
        assert any("don't know" in phrase for phrase in UNCERTAINTY_FAMILIES[0])


class TestDataset:
    def test_dataset_has_at_least_10_cases(self):
        assert len(DATASET) >= 10

    def test_each_case_has_id_question_and_kind(self):
        for row in DATASET:
            assert "id" in row, f"Missing 'id' in {row}"
            assert "question" in row, f"Missing 'question' in {row}"
            assert "kind" in row, f"Missing 'kind' in {row}"

    def test_dataset_has_refusal_cases(self):
        refusals = [r for r in DATASET if r.get("kind") == "refuse"]
        assert len(refusals) >= 2, "Assignment requires at least 2 refusal/clarification cases"

    def test_dataset_has_clarify_cases(self):
        clarifies = [r for r in DATASET if r.get("kind") == "clarify"]
        assert len(clarifies) >= 2

    def test_dataset_has_realistic_mix_of_categories(self):
        kinds = {row["kind"] for row in DATASET}
        assert {"knowledge", "tool", "math", "clarify", "refuse", "ood", "smalltalk"} <= kinds

    def test_knowledge_rows_use_phrase_families(self):
        knowledge = [r for r in DATASET if r["kind"] == "knowledge"]
        assert all(r.get("concept_groups") for r in knowledge)
        assert all(isinstance(group, list) for r in knowledge for group in r["concept_groups"])

    def test_questions_feel_human_not_template_like(self):
        questions = [row["question"] for row in DATASET]
        assert any("btw" in q or "kinda" in q or "abt" in q for q in questions)


class TestAggregate:
    def test_empty_results_returns_zeros(self):
        result = aggregate([])
        assert result["n"] == 0
        assert result["avg_score"] == 0.0
        assert result["per_kind"] == {}
        assert result["component_averages"] == {}
        assert result["failed_cases"] == []

    def test_all_perfect_scores(self):
        results = [
            EvalResult(
                id="q1",
                question="q",
                answer="a",
                kind="knowledge",
                score=1.0,
                expected_action="retrieve",
                actual_action="retrieve",
                action_match=True,
                breakdown={"action": 1.0, "behavior": 1.0, "content": 1.0, "safety": 1.0},
            ),
            EvalResult(
                id="q2",
                question="q",
                answer="a",
                kind="clarify",
                score=1.0,
                expected_action="clarify",
                actual_action="clarify",
                action_match=True,
                breakdown={"action": 1.0, "behavior": 1.0, "content": None, "safety": 1.0},
            ),
        ]
        agg = aggregate(results)
        assert agg["avg_score"] == 1.0
        assert agg["action_accuracy"] == 1.0
        assert agg["per_kind"]["knowledge"]["avg_score"] == 1.0
        assert agg["per_kind"]["clarify"]["avg_score"] == 1.0
        assert agg["component_averages"]["action"] == 1.0

    def test_mixed_scores_include_failed_case_summary(self):
        results = [
            EvalResult(
                id="q1",
                question="q",
                answer="a",
                kind="knowledge",
                score=1.0,
                expected_action="retrieve",
                actual_action="retrieve",
                action_match=True,
                breakdown={"action": 1.0, "behavior": 1.0, "content": 1.0, "safety": 1.0},
            ),
            EvalResult(
                id="q2",
                question="q",
                answer="a",
                kind="ood",
                score=0.5,
                expected_action="retrieve",
                actual_action="retrieve",
                action_match=True,
                breakdown={
                    "action": 1.0,
                    "behavior": 0.0,
                    "content": None,
                    "safety": 1.0,
                    "uncertainty": 0.0,
                },
                notes=["uncertainty handling missing"],
            ),
        ]
        agg = aggregate(results)
        assert agg["avg_score"] == 0.75
        assert agg["action_accuracy"] == 1.0
        assert agg["per_kind"]["ood"]["avg_score"] == 0.5
        assert agg["component_averages"]["uncertainty"] == 0.0
        assert len(agg["failed_cases"]) == 1
        assert agg["failed_cases"][0]["id"] == "q2"
