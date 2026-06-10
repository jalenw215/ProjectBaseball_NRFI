from __future__ import annotations

from datetime import date, datetime
from pathlib import Path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def today_iso() -> str:
    return date.today().isoformat()


def american_to_implied_probability(odds: float | int) -> float:
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def confidence_tier(probability: float) -> str:
    if probability >= 0.62:
        return "A"
    if probability >= 0.56:
        return "B"
    if probability >= 0.50:
        return "C"
    return "YRFI Lean"
