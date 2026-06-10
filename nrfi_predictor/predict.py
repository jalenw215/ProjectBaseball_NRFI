from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .config import DEFAULT_MODEL_FILE, DEFAULT_ODDS_FILE, DEFAULT_PREDICTIONS_FILE, DEFAULT_TRAINING_FILE
from .data_sources import fetch_schedule, fetch_weather_for_game, read_manual_odds
from .features import latest_pitcher_profiles, latest_team_profiles
from .model import load_model, predict_probabilities
from .parks import park_info
from .utils import american_to_implied_probability, confidence_tier, ensure_parent


def predict_for_date(
    game_date: str | None = None,
    training_path: Path = DEFAULT_TRAINING_FILE,
    model_path: Path = DEFAULT_MODEL_FILE,
    odds_path: Path = DEFAULT_ODDS_FILE,
    output_path: Path = DEFAULT_PREDICTIONS_FILE,
) -> Path:
    game_date = game_date or date.today().isoformat()
    training = pd.read_csv(training_path, parse_dates=["game_date"])
    model, feature_columns = load_model(model_path)
    schedule = fetch_schedule(game_date)
    candidates = build_daily_candidates(training, schedule, feature_columns)
    if candidates.empty:
        raise ValueError("No daily candidates could be built. Check schedule and training data.")
    candidates["nrfi_probability"] = predict_probabilities(model, candidates, feature_columns)
    candidates["yrfi_probability"] = 1.0 - candidates["nrfi_probability"]
    candidates["feature_set"] = _feature_set_name(training)
    candidates["confidence_tier"] = candidates["nrfi_probability"].map(confidence_tier)
    candidates = attach_odds(candidates, game_date, odds_path)
    candidates["rank"] = candidates["nrfi_probability"].rank(ascending=False, method="first").astype(int)
    candidates = candidates.sort_values(["rank", "away_team", "home_team"])
    candidates.to_csv(ensure_parent(output_path), index=False)
    latest = output_path.parent / "latest_predictions.csv"
    if latest != output_path:
        candidates.to_csv(ensure_parent(latest), index=False)
    return output_path


def build_daily_candidates(training: pd.DataFrame, schedule: list[dict], feature_columns: list[str]) -> pd.DataFrame:
    team_profiles = latest_team_profiles(training)
    pitcher_profiles = latest_pitcher_profiles(training)
    rows = []
    for game in schedule:
        weather = fetch_weather_for_game(game)
        park_factor = park_info(game.get("venue_name"))["run_factor"]
        away_team = game.get("away_team")
        home_team = game.get("home_team")
        row = {
            "game_date": game.get("game_date"),
            "game_pk": game.get("game_pk"),
            "away_team": away_team,
            "home_team": home_team,
            "away_starter_id": game.get("away_probable_pitcher_id"),
            "away_starter": game.get("away_probable_pitcher") or "TBD",
            "home_starter_id": game.get("home_probable_pitcher_id"),
            "home_starter": game.get("home_probable_pitcher") or "TBD",
            "venue_name": game.get("venue_name"),
            "status": game.get("status"),
            "park_run_factor": park_factor,
            "temperature_2m": weather["temperature_2m"],
            "wind_speed_10m": weather["wind_speed_10m"],
        }
        row.update(_prefixed_team_features(team_profiles, away_team, "away"))
        row.update(_prefixed_team_features(team_profiles, home_team, "home"))
        row.update(_prefixed_pitcher_features(pitcher_profiles, game.get("away_probable_pitcher_id"), "away_sp"))
        row.update(_prefixed_pitcher_features(pitcher_profiles, game.get("home_probable_pitcher_id"), "home_sp"))
        for col in feature_columns:
            row.setdefault(col, 0.0)
        row["matchup_note"] = _matchup_note(row)
        rows.append(row)
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    for col in feature_columns:
        if col not in candidates.columns:
            candidates[col] = 0.0
    candidates[feature_columns] = candidates[feature_columns].fillna(0.0)
    return candidates


def attach_odds(candidates: pd.DataFrame, game_date: str, odds_path: Path) -> pd.DataFrame:
    out = candidates.copy()
    for market in ["nrfi", "yrfi"]:
        out[f"{market}_american_odds"] = pd.NA
        out[f"{market}_book"] = pd.NA
        out[f"{market}_implied_probability"] = pd.NA
        out[f"{market}_value_flag"] = False

    odds = read_manual_odds(odds_path)
    odds = odds[odds["date"].astype(str) == game_date].copy()
    if odds.empty:
        return out
    odds["game_pk_key"] = odds["game_pk"].astype(str)
    out["game_pk_key"] = out["game_pk"].astype(str)
    for market in ["NRFI", "YRFI"]:
        subset = odds[odds["market"] == market].drop_duplicates("game_pk_key", keep="last")
        if subset.empty:
            continue
        prefix = market.lower()
        merged = out[["game_pk_key"]].merge(
            subset[["game_pk_key", "american_odds", "book"]],
            on="game_pk_key",
            how="left",
        )
        out[f"{prefix}_american_odds"] = merged["american_odds"].values
        out[f"{prefix}_book"] = merged["book"].values
        out[f"{prefix}_implied_probability"] = out[f"{prefix}_american_odds"].map(
            lambda x: american_to_implied_probability(x) if pd.notna(x) else pd.NA
        )
        probability_col = f"{prefix}_probability"
        out[f"{prefix}_value_flag"] = out.apply(
            lambda row: bool(
                pd.notna(row[f"{prefix}_implied_probability"])
                and row[probability_col] > row[f"{prefix}_implied_probability"]
            ),
            axis=1,
        )
    return out.drop(columns=["game_pk_key"])


def _prefixed_team_features(profiles: pd.DataFrame, team: str | None, prefix: str) -> dict[str, float]:
    profile = profiles[profiles["team"] == team] if not profiles.empty and team is not None else pd.DataFrame()
    source = _neutral_team_features() if profile.empty else profile.iloc[0].to_dict()
    return {f"{prefix}_{key}": value for key, value in source.items() if key.startswith("team_")}


def _prefixed_pitcher_features(profiles: pd.DataFrame, pitcher_id, prefix: str) -> dict[str, float]:
    profile = profiles[profiles["pitcher"].astype(str) == str(pitcher_id)] if not profiles.empty and pitcher_id is not None else pd.DataFrame()
    source = _neutral_pitcher_features() if profile.empty else profile.iloc[0].to_dict()
    return {f"{prefix}_{key}": value for key, value in source.items() if key not in {"game_date", "pitcher"}}


def _neutral_team_features() -> dict[str, float]:
    return {
        "team_pa_30": 0.0,
        "team_yrfi_rate_30": 0.27,
        "team_runs_per_fi_30": 0.45,
        "team_hits_per_fi_30": 1.0,
        "team_obp_30": 0.315,
        "team_walk_rate_30": 0.08,
        "team_hardhit_rate_30": 0.36,
        "team_barrel_rate_30": 0.07,
    }


def _neutral_pitcher_features() -> dict[str, float]:
    return {
        "bf_30": 0.0,
        "yrfi_allowed_rate_30": 0.27,
        "runs_allowed_per_fi_30": 0.45,
        "hits_allowed_per_fi_30": 1.0,
        "obp_allowed_30": 0.315,
        "k_rate_30": 0.22,
        "bb_rate_30": 0.08,
        "hardhit_allowed_rate_30": 0.36,
        "barrel_allowed_rate_30": 0.07,
    }


def _matchup_note(row: dict) -> str:
    return (
        f"{row.get('away_team')} at {row.get('home_team')} | "
        f"SP: {row.get('away_starter', 'TBD')} vs {row.get('home_starter', 'TBD')} | "
        f"park {float(row.get('park_run_factor', 1.0)):.2f} | "
        f"{float(row.get('temperature_2m', 70.0)):.0f}F"
    )


def _feature_set_name(training: pd.DataFrame) -> str:
    if "feature_set" in training.columns and training["feature_set"].notna().any():
        return str(training["feature_set"].dropna().iloc[0])
    return "baseline"
