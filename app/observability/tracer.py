from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Trace:
    steps: list[str] = field(default_factory=list)

    def add(self, step: str) -> None:
        self.steps.append(step)

    def as_list(self) -> list[str]:
        return list(self.steps)
