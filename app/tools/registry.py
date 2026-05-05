from __future__ import annotations

from typing import Callable, Any

from app.tools.arxiv_tool import arxiv_search
from app.tools.calculator import calculator


TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "calculator": calculator,
    "arxiv_search": arxiv_search,
}
