from pathlib import Path

import pandas as pd

from nrfi_predictor.features import build_training_rows
from nrfi_predictor.model import train_model
from nrfi_predictor.predict import attach_odds, build_daily_candidates, predict_for_date
from tests.test_features import _statcast_games


def test_build_daily_candidates_with_missing_pitcher_defaults(tmp_path: Path, monkeypatch):
    statcast = tmp_path / "statcast.csv"
    training_path = tmp_path / "training.csv"
    pd.DataFrame(_statcast_games(days=40)).to_csv(statcast, index=False)
    build_training_rows(statcast, training_path)
    training = pd.read_csv(training_path, parse_dates=["game_date"])
    monkeypatch.setattr("nrfi_predictor.predict.fetch_weather_for_game", lambda game: {"temperature_2m": 72.0, "wind_speed_10m": 4.0})

    candidates = build_daily_candidates(training, [_schedule_game()], ["away_team_pa_30", "away_sp_bf_30"])

    assert len(candidates) == 1
    assert candidates["away_starter"].iloc[0] == "TBD"
    assert candidates["away_sp_bf_30"].iloc[0] == 0.0
    assert "matchup_note" in candidates.columns


def test_attach_odds_for_nrfi_and_yrfi(tmp_path: Path):
    odds = tmp_path / "manual_nrfi_odds.csv"
    odds.write_text(
        "date,game_pk,market,american_odds,book\n"
        "2025-06-01,123,NRFI,+120,BookA\n"
        "2025-06-01,123,YRFI,-105,BookA\n",
        encoding="utf-8",
    )
    candidates = pd.DataFrame(
        [{"game_pk": 123, "nrfi_probability": 0.60, "yrfi_probability": 0.40, "away_team": "NYY", "home_team": "BOS"}]
    )

    merged = attach_odds(candidates, "2025-06-01", odds)

    assert merged["nrfi_american_odds"].iloc[0] == 120
    assert bool(merged["nrfi_value_flag"].iloc[0]) is True
    assert merged["yrfi_american_odds"].iloc[0] == -105
    assert bool(merged["yrfi_value_flag"].iloc[0]) is False


def test_predict_for_date_writes_nrfi_and_yrfi(tmp_path: Path, monkeypatch):
    statcast = tmp_path / "statcast.csv"
    training = tmp_path / "training.csv"
    model = tmp_path / "model.joblib"
    output = tmp_path / "predictions.csv"
    pd.DataFrame(_statcast_games(days=50)).to_csv(statcast, index=False)
    build_training_rows(statcast, training)
    train_model(training, model)
    monkeypatch.setattr("nrfi_predictor.predict.fetch_schedule", lambda game_date: [_schedule_game()])
    monkeypatch.setattr("nrfi_predictor.predict.fetch_weather_for_game", lambda game: {"temperature_2m": 72.0, "wind_speed_10m": 4.0})

    predict_for_date("2025-06-01", training, model, output_path=output)
    predictions = pd.read_csv(output)

    assert len(predictions) == 1
    assert {"nrfi_probability", "yrfi_probability", "rank"}.issubset(predictions.columns)
    assert abs((predictions["nrfi_probability"] + predictions["yrfi_probability"]).iloc[0] - 1.0) < 0.00001


def _schedule_game():
    return {
        "game_pk": 123,
        "game_date": "2025-06-01",
        "game_time_utc": "2025-06-01T17:00:00Z",
        "venue_name": "Fenway Park",
        "away_team": "NYY",
        "home_team": "BOS",
        "away_probable_pitcher_id": None,
        "away_probable_pitcher": None,
        "home_probable_pitcher_id": 200,
        "home_probable_pitcher": "Home Starter",
        "status": "Scheduled",
    }
