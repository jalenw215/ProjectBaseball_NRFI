from __future__ import annotations

from pathlib import Path

import pandas as pd


def format_daily_report(predictions_path: Path, limit: int = 10) -> str:
    df = pd.read_csv(predictions_path)
    if df.empty:
        return "No NRFI/YRFI predictions available."
    top = df.sort_values("rank").head(limit)
    lines = ["# MLB NRFI/YRFI Predictor", "", "Daily probabilities are research estimates, not guarantees.", ""]
    for _, row in top.iterrows():
        nrfi_odds = _odds_piece(row, "nrfi")
        yrfi_odds = _odds_piece(row, "yrfi")
        lines.append(
            f"{int(row['rank'])}. {row['away_team']} at {row['home_team']} "
            f"NRFI {row['nrfi_probability']:.1%} / YRFI {row['yrfi_probability']:.1%} "
            f"[{row['confidence_tier']}]{nrfi_odds}{yrfi_odds}"
        )
        lines.append(f"   {row['matchup_note']}")
    return "\n".join(lines)


def _odds_piece(row: pd.Series, market: str) -> str:
    odds_col = f"{market}_american_odds"
    implied_col = f"{market}_implied_probability"
    value_col = f"{market}_value_flag"
    if odds_col not in row or pd.isna(row[odds_col]):
        return ""
    value = " VALUE" if bool(row.get(value_col, False)) else ""
    implied = row.get(implied_col)
    implied_text = f" ({float(implied):.1%})" if pd.notna(implied) else ""
    return f" | {market.upper()} odds {row[odds_col]}{implied_text}{value}"
