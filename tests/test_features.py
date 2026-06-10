from pathlib import Path

import pandas as pd

from nrfi_predictor.features import build_training_rows, resolve_feature_columns


def test_build_training_rows_labels_first_inning_nrfi(tmp_path: Path):
    source = tmp_path / "statcast.csv"
    output = tmp_path / "training.csv"
    pd.DataFrame(_statcast_games(days=4)).to_csv(source, index=False)

    build_training_rows(source, output)
    training = pd.read_csv(output).sort_values("game_date")

    assert len(training) == 4
    assert "target_nrfi" in training.columns
    assert set(training["target_nrfi"]) == {0, 1}
    assert training.loc[training["game_pk"] == 900001, "target_nrfi"].iloc[0] == 1
    assert training.loc[training["game_pk"] == 900002, "target_nrfi"].iloc[0] == 0


def test_rolling_features_exclude_same_day(tmp_path: Path):
    source = tmp_path / "statcast.csv"
    output = tmp_path / "training.csv"
    pd.DataFrame(_statcast_games(days=3)).to_csv(source, index=False)

    build_training_rows(source, output, feature_set="all_free_statcast")
    training = pd.read_csv(output).sort_values("game_date")

    for col in resolve_feature_columns("all_free_statcast"):
        assert col in training.columns
    first = training.iloc[0]
    second = training.iloc[1]
    assert first["away_team_pa_30"] == 0
    assert first["home_sp_bf_30"] == 0
    assert second["away_team_pa_30"] > 0
    assert second["away_team_yrfi_rate_30"] == 0


def test_build_training_rows_tolerates_missing_venue_name(tmp_path: Path):
    source = tmp_path / "statcast.csv"
    output = tmp_path / "training.csv"
    data = pd.DataFrame(_statcast_games(days=2)).drop(columns=["venue_name"])
    data.to_csv(source, index=False)

    build_training_rows(source, output)
    training = pd.read_csv(output)

    assert not training.empty
    assert set(training["venue_name"]) == {"Unknown"}
    assert set(training["park_run_factor"]) == {1.0}


def test_build_training_rows_filters_non_regular_game_type(tmp_path: Path):
    source = tmp_path / "statcast.csv"
    output = tmp_path / "training.csv"
    data = pd.DataFrame(_statcast_games(days=2))
    data.loc[data["game_pk"] == 900002, "game_type"] = "S"
    data.loc[data["game_pk"] == 900001, "game_type"] = "R"
    data.to_csv(source, index=False)

    build_training_rows(source, output)
    training = pd.read_csv(output)

    assert len(training) == 1
    assert training["game_pk"].iloc[0] == 900001


def _statcast_games(days: int = 60) -> list[dict]:
    rows = []
    for i, day in enumerate(pd.date_range("2025-04-01", periods=days)):
        game_pk = 900001 + i
        yrfi = i % 2 == 1
        away_runs = 1 if yrfi else 0
        rows.extend(
            [
                _pitch(
                    day,
                    game_pk,
                    "Top",
                    "NYY",
                    "BOS",
                    batter=100 + i,
                    pitcher=200,
                    event="single" if yrfi else "field_out",
                    post_away_score=away_runs,
                    post_home_score=0,
                    at_bat=1,
                ),
                _pitch(
                    day,
                    game_pk,
                    "Top",
                    "NYY",
                    "BOS",
                    batter=101 + i,
                    pitcher=200,
                    event="strikeout",
                    post_away_score=away_runs,
                    post_home_score=0,
                    at_bat=2,
                ),
                _pitch(
                    day,
                    game_pk,
                    "Bot",
                    "NYY",
                    "BOS",
                    batter=300 + i,
                    pitcher=400,
                    event="walk" if yrfi else "field_out",
                    post_away_score=away_runs,
                    post_home_score=0,
                    at_bat=3,
                ),
                _pitch(
                    day,
                    game_pk,
                    "Bot",
                    "NYY",
                    "BOS",
                    batter=301 + i,
                    pitcher=400,
                    event="field_out",
                    post_away_score=away_runs,
                    post_home_score=0,
                    at_bat=4,
                ),
            ]
        )
    return rows


def _pitch(day, game_pk, topbot, away, home, batter, pitcher, event, post_away_score, post_home_score, at_bat):
    return {
        "game_date": day.date().isoformat(),
        "game_pk": game_pk,
        "inning": 1,
        "inning_topbot": topbot,
        "away_team": away,
        "home_team": home,
        "venue_name": "Fenway Park",
        "batter": batter,
        "pitcher": pitcher,
        "player_name": f"Player {batter}",
        "events": event,
        "description": "hit_into_play" if event != "strikeout" else "swinging_strike",
        "launch_speed": 101 if event == "single" else 84,
        "launch_speed_angle": 6 if event == "single" else 2,
        "post_away_score": post_away_score,
        "post_home_score": post_home_score,
        "at_bat_number": at_bat,
        "pitch_number": 1,
    }
