from __future__ import annotations

from app.ingestion.fetch_arxiv import fetch_arxiv


def arxiv_search(query: str, max_results: int = 3) -> list[dict]:
    papers = fetch_arxiv(query=query, max_results=max_results)
    return [
        {"id": p.arxiv_id, "title": p.title, "summary": p.summary, "pdf_url": p.pdf_url}
        for p in papers
    ]
