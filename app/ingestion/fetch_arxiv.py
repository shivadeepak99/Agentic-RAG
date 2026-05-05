from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    summary: str
    pdf_url: str


def fetch_arxiv(query: str, max_results: int = 5) -> list[ArxivPaper]:
    """Fetch basic metadata from arXiv API.

    Uses the public Atom feed endpoint. This is intentionally lightweight.
    """

    url = "http://export.arxiv.org/api/query"
    params = {"search_query": query, "start": 0, "max_results": max_results}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    # Minimal Atom parsing without extra deps.
    # We only pick the fields we need via very small string scans.
    text = resp.text
    entries = text.split("<entry>")[1:]

    papers: list[ArxivPaper] = []
    for ent in entries:
        def _extract(tag: str) -> str:
            start = ent.find(f"<{tag}>")
            end = ent.find(f"</{tag}>")
            if start == -1 or end == -1:
                return ""
            return ent[start + len(tag) + 2 : end].strip()

        arxiv_id = _extract("id").split("/")[-1]
        title = _extract("title").replace("\n", " ").strip()
        summary = _extract("summary").replace("\n", " ").strip()

        pdf_url = ""
        for part in ent.split("<link "):
            if 'title="pdf"' in part and "href=" in part:
                href = part.split("href=")[1].split("\"")[1]
                pdf_url = href
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        if arxiv_id:
            papers.append(ArxivPaper(arxiv_id=arxiv_id, title=title, summary=summary, pdf_url=pdf_url))

    return papers
