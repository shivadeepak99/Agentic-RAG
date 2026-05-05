"""Ablation: hybrid (vector + BM25 rerank) vs. vector-only retrieval.

Run:
    python scripts/run_ablation.py

Prints a side-by-side comparison table and exits with code 0.
The assignment requires showing eval scores with and without hybrid reranking.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.eval.dataset import DATASET
from app.eval.evaluator import aggregate, run_eval

import app.agent.nodes.retrieve as _retrieve_mod
import app.retrieval.hybrid as _hybrid_mod


def _vector_only_search(query: str) -> list[dict]:
    """Return top-k vector hits with no BM25 reranking."""
    from app.config import settings
    from app.retrieval.vector_store import get_vector_store

    store = get_vector_store()
    hits = store.search(query, top_k=settings.retrieval_top_k)
    return hits


def _run_with_mode(label: str, search_fn) -> dict:
    """Monkey-patch hybrid_search and run the full eval."""
    original_hybrid = _hybrid_mod.hybrid_search
    original_node = _retrieve_mod.hybrid_search

    _hybrid_mod.hybrid_search = search_fn
    _retrieve_mod.hybrid_search = search_fn
    try:
        results = run_eval(DATASET)
    finally:
        _hybrid_mod.hybrid_search = original_hybrid
        _retrieve_mod.hybrid_search = original_node

    summary = aggregate(results)
    return {"label": label, "results": results, "summary": summary}


def main() -> int:
    print("Running ablation study: hybrid vs. vector-only retrieval\n")

    hybrid = _run_with_mode("hybrid (vector + BM25 rerank)", _hybrid_mod.hybrid_search)
    vector = _run_with_mode("vector-only (no BM25)", _vector_only_search)

    for mode in (hybrid, vector):
        s = mode["summary"]
        print(f"=== {mode['label']} ===")
        print(f"  avg_score:       {s['avg_score']:.3f}")
        print(f"  action_accuracy: {s['action_accuracy']:.3f}  "
              f"({s['n_with_expected_action']} cases with expected_action)")
        print()

    # Per-case diff
    print(f"{'ID':<20} {'hybrid':>8} {'vec-only':>10}  {'delta':>8}")
    print("-" * 52)
    for hr, vr in zip(hybrid["results"], vector["results"]):
        delta = hr.score - vr.score
        sign = "+" if delta >= 0 else ""
        print(f"{hr.id:<20} {hr.score:>8.3f} {vr.score:>10.3f}  {sign}{delta:>7.3f}")

    hs = hybrid["summary"]["avg_score"]
    vs = vector["summary"]["avg_score"]
    lift = hs - vs
    sign = "+" if lift >= 0 else ""
    print("-" * 52)
    print(f"{'TOTAL':<20} {hs:>8.3f} {vs:>10.3f}  {sign}{lift:>7.3f}")
    print()
    print(f"Hybrid {'outperforms' if lift > 0 else 'underperforms'} vector-only by {abs(lift):.3f} avg score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
