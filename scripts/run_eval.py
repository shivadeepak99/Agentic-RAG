from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.eval.dataset import DATASET
from app.eval.evaluator import aggregate, run_eval
from app.eval.results import write_results


def main() -> int:
    results = run_eval(DATASET)
    summary = aggregate(results)

    out_path = Path("data/eval/results.json")
    write_results(results, out_path)

    print(f"Wrote {len(results)} eval results to {out_path}")
    print(f"Avg score:        {summary['avg_score']:.3f}")
    print(f"Action accuracy:  {summary['action_accuracy']:.3f} "
          f"({summary['n_with_expected_action']} cases with expected_action)")

    print("\nPer-case:")
    for r in results:
        flag = "OK " if r.score >= 0.99 else "..."
        print(f"  [{flag}] {r.id} score={r.score:.2f} action={r.actual_action} "
              f"expected={r.expected_action}")
        for note in r.notes:
            print(f"        - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
