from __future__ import annotations

from pathlib import Path


def pdf_to_text(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF (fitz)."""

    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    try:
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text("text"))
        return "\n".join(parts).strip()
    finally:
        doc.close()


def write_parsed_text(pdf_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (pdf_path.stem + ".txt")

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    text = pdf_to_text(pdf_path)
    out_path.write_text(text, encoding="utf-8")
    return out_path
