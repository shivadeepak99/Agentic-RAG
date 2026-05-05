from __future__ import annotations

from app.agent.graph import build_graph, run_agent


def main() -> int:
    graph = build_graph()

    for q in [
        "What is RAG?",
        "arxiv: diffusion models",
        "hi",
    ]:
        out = run_agent(graph, q)
        print("\nQ:", q)
        print("A:", out.get("answer"))
        print("Trace:", out.get("trace"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
