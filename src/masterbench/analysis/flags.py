"""Plain-language findings attached to an analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["info", "warn", "problem"]


@dataclass(frozen=True)
class Flag:
    code: str
    severity: Severity
    message: str
    value: float | None = None
    start_s: float | None = None
    end_s: float | None = None
