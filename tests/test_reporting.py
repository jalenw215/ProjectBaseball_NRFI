from pathlib import Path

import pandas as pd

from nrfi_predictor.reporting import format_daily_report


def test_format_daily_report(tmp_path: Path):
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {
                "rank": 1,
                "away_team": "NYY",
                "home_team": "BOS",
                "nrfi_probability": 0.61,
                "yrfi_probability": 0.39,
                "confidence_tier": "A",
                "matchup_note": "NYY at BOS | SP: TBD vs Home Starter",
                "nrfi_american_odds": 110,
                "nrfi_implied_probability": 0.476,
                "nrfi_value_flag": True,
            }
        ]
    ).to_csv(predictions, index=False)

    report = format_daily_report(predictions)

    assert "MLB NRFI/YRFI Predictor" in report
    assert "NYY at BOS" in report
    assert "NRFI 61.0%" in report
    assert "VALUE" in report
