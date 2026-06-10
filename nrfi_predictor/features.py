from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_FEATURE_SET, FEATURE_GROUPS, FEATURE_SET_ALIASES, PROCESSED_DIR
from .parks import park_info
from .utils import ensure_parent


HIT_EVENTS = {"single", "double", "triple", "home_run"}
WALK_EVENTS = {"walk", "intent_walk"}
HBP_EVENTS = {"hit_by_pitch"}


def load_statcast(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if "game_date" not in df.columns:
        raise ValueError("Statcast file must contain game_date")
    if "game_pk" not in df.columns:
        raise ValueError("Statcast file must contain game_pk")
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["inning"] = pd.to_numeric(_series(out, "inning"), errors="coerce")
    out["post_away_score"] = pd.to_numeric(_series(out, "post_away_score"), errors="coerce")
    out["post_home_score"] = pd.to_numeric(_series(out, "post_home_score"), errors="coerce")
    events = _series(out, "events")
    descriptions = _series(out, "description")
    out["is_hit"] = events.isin(HIT_EVENTS).astype(int)
    out["is_walk"] = events.isin(WALK_EVENTS).astype(int)
    out["is_hbp"] = events.isin(HBP_EVENTS).astype(int)
    out["is_onbase"] = (out["is_hit"] | out["is_walk"] | out["is_hbp"]).astype(int)
    out["is_strikeout"] = (events == "strikeout").astype(int)
    out["is_bbe"] = _series(out, "launch_speed").notna().astype(int)
    out["is_hardhit"] = (pd.to_numeric(_series(out, "launch_speed"), errors="coerce") >= 95).astype(int)
    out["is_barrel"] = (pd.to_numeric(_series(out, "launch_speed_angle"), errors="coerce") == 6).astype(int)
    out["is_pa"] = events.notna().astype(int)
    out["is_contact_pitch"] = descriptions.isin(
        ["foul", "foul_bunt", "foul_tip", "hit_into_play", "hit_into_play_no_out", "hit_into_play_score"]
    ).astype(int)
    topbot = _series(out, "inning_topbot")
    out["batter_team"] = np.where(topbot == "Top", _series(out, "away_team"), _series(out, "home_team"))
    out["pitcher_team"] = np.where(topbot == "Top", _series(out, "home_team"), _series(out, "away_team"))
    return out


def resolve_feature_columns(feature_set: str = DEFAULT_FEATURE_SET) -> list[str]:
    feature_set = FEATURE_SET_ALIASES.get(feature_set, feature_set)
    groups = list(FEATURE_GROUPS) if feature_set == "all_free_statcast" else feature_set.split("+")
    unknown = [group for group in groups if group not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"Unknown feature group(s): {unknown}")
    columns: list[str] = []
    for group in groups:
        for col in FEATURE_GROUPS[group]:
            if col not in columns:
                columns.append(col)
    return columns


def feature_set_slug(feature_set: str = DEFAULT_FEATURE_SET) -> str:
    return FEATURE_SET_ALIASES.get(feature_set, feature_set).replace("+", "__")


def build_training_rows(statcast_path: Path, output_path: Path | None = None, feature_set: str = DEFAULT_FEATURE_SET) -> Path:
    output_path = output_path or PROCESSED_DIR / "training_rows.csv"
    df = add_derived_columns(load_statcast(statcast_path))
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"].copy()
    df = df[df["inning"] == 1].copy()
    if df.empty:
        raise ValueError("No first-inning Statcast rows were available.")

    feature_columns = resolve_feature_columns(feature_set)
    game_rows = _game_rows(df)
    team_day = _team_first_inning_day(df)
    pitcher_day = _starter_first_inning_day(df)
    team_roll = _rolling_team_features(team_day)
    pitcher_roll = _rolling_pitcher_features(pitcher_day)

    rows = game_rows.merge(team_roll.add_prefix("away_"), left_on=["game_date", "away_team"], right_on=["away_game_date", "away_team"], how="left")
    rows = rows.drop(columns=[c for c in ["away_game_date"] if c in rows.columns])
    rows = rows.merge(team_roll.add_prefix("home_"), left_on=["game_date", "home_team"], right_on=["home_game_date", "home_team"], how="left")
    rows = rows.drop(columns=[c for c in ["home_game_date"] if c in rows.columns])
    rows = rows.merge(
        pitcher_roll.add_prefix("away_sp_"),
        left_on=["game_date", "away_starter_id"],
        right_on=["away_sp_game_date", "away_sp_pitcher"],
        how="left",
    )
    rows = rows.drop(columns=[c for c in ["away_sp_game_date", "away_sp_pitcher"] if c in rows.columns])
    rows = rows.merge(
        pitcher_roll.add_prefix("home_sp_"),
        left_on=["game_date", "home_starter_id"],
        right_on=["home_sp_game_date", "home_sp_pitcher"],
        how="left",
    )
    rows = rows.drop(columns=[c for c in ["home_sp_game_date", "home_sp_pitcher"] if c in rows.columns])

    rows["park_run_factor"] = rows["venue_name"].map(lambda name: park_info(name).get("run_factor", 1.0)).fillna(1.0)
    rows["temperature_2m"] = 70.0
    rows["wind_speed_10m"] = 5.0
    for col in feature_columns:
        if col not in rows.columns:
            rows[col] = 0.0
    rows[feature_columns] = rows[feature_columns].fillna(0.0)
    rows["feature_set"] = feature_set
    rows = rows.sort_values(["game_date", "game_pk"])
    rows.to_csv(ensure_parent(output_path), index=False)
    return output_path


def _game_rows(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"], na_position="last")
    if "venue_name" not in ordered.columns:
        ordered = ordered.assign(venue_name="Unknown")
    meta = (
        ordered.groupby("game_pk", dropna=False)
        .agg(
            game_date=("game_date", "first"),
            away_team=("away_team", "first"),
            home_team=("home_team", "first"),
            venue_name=("venue_name", _first_or_unknown),
            away_first_score=("post_away_score", "max"),
            home_first_score=("post_home_score", "max"),
        )
        .reset_index()
    )
    starters = _game_starters(ordered)
    meta = meta.merge(starters, on="game_pk", how="left")
    meta["first_inning_runs"] = meta[["away_first_score", "home_first_score"]].fillna(0).sum(axis=1)
    meta["target_nrfi"] = (meta["first_inning_runs"] <= 0).astype(int)
    return meta


def _game_starters(df: pd.DataFrame) -> pd.DataFrame:
    top = (
        df[df["inning_topbot"] == "Top"]
        .sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"], na_position="last")
        .groupby("game_pk", dropna=False)
        .agg(home_starter_id=("pitcher", "first"), home_starter=("player_name", _pitcher_name_fallback))
        .reset_index()
    )
    bot = (
        df[df["inning_topbot"] == "Bot"]
        .sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"], na_position="last")
        .groupby("game_pk", dropna=False)
        .agg(away_starter_id=("pitcher", "first"), away_starter=("player_name", _pitcher_name_fallback))
        .reset_index()
    )
    return top.merge(bot, on="game_pk", how="outer")


def _team_first_inning_day(df: pd.DataFrame) -> pd.DataFrame:
    half = (
        df.groupby(["game_date", "game_pk", "batter_team", "inning_topbot"], dropna=False)
        .agg(
            team=("batter_team", "first"),
            runs=("post_away_score", "max"),
            home_runs=("post_home_score", "max"),
            pa=("is_pa", "sum"),
            hits=("is_hit", "sum"),
            walks=("is_walk", "sum"),
            hbp=("is_hbp", "sum"),
            onbase=("is_onbase", "sum"),
            bbe=("is_bbe", "sum"),
            hardhits=("is_hardhit", "sum"),
            barrels=("is_barrel", "sum"),
        )
        .reset_index(drop=False)
    )
    half["runs"] = np.where(half["inning_topbot"] == "Top", half["runs"], half["home_runs"])
    return half.drop(columns=["home_runs"]).sort_values(["team", "game_date"])


def _starter_first_inning_day(df: pd.DataFrame) -> pd.DataFrame:
    first_pitchers = (
        df.sort_values(["game_date", "game_pk", "inning_topbot", "at_bat_number", "pitch_number"], na_position="last")
        .groupby(["game_pk", "inning_topbot"], dropna=False)["pitcher"]
        .first()
        .reset_index()
        .rename(columns={"pitcher": "starter"})
    )
    first_inning = df.merge(first_pitchers, on=["game_pk", "inning_topbot"], how="left")
    first_inning = first_inning[first_inning["pitcher"] == first_inning["starter"]].copy()
    half = (
        first_inning.groupby(["game_date", "game_pk", "pitcher", "pitcher_team", "inning_topbot"], dropna=False)
        .agg(
            runs_allowed_away=("post_away_score", "max"),
            runs_allowed_home=("post_home_score", "max"),
            batters_faced=("is_pa", "sum"),
            hits_allowed=("is_hit", "sum"),
            walks_allowed=("is_walk", "sum"),
            hbp_allowed=("is_hbp", "sum"),
            onbase_allowed=("is_onbase", "sum"),
            strikeouts=("is_strikeout", "sum"),
            bbe_allowed=("is_bbe", "sum"),
            hardhits_allowed=("is_hardhit", "sum"),
            barrels_allowed=("is_barrel", "sum"),
        )
        .reset_index()
    )
    half["runs_allowed"] = np.where(half["inning_topbot"] == "Top", half["runs_allowed_away"], half["runs_allowed_home"])
    return half.drop(columns=["runs_allowed_away", "runs_allowed_home"]).sort_values(["pitcher", "game_date"])


def _rolling_team_features(team_day: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in team_day.groupby("team", dropna=False):
        group = group.sort_values("game_date").copy()
        group["yrfi"] = (group["runs"] > 0).astype(int)
        shifted = group[["yrfi", "runs", "pa", "hits", "walks", "hbp", "onbase", "bbe", "hardhits", "barrels"]].shift(1)
        roll = shifted.rolling(30, min_periods=1).sum()
        games = shifted["pa"].rolling(30, min_periods=1).count().replace(0, np.nan)
        parts.append(
            pd.DataFrame(
                {
                    "game_date": group["game_date"].values,
                    "team": group["team"].values,
                    "team_pa_30": roll["pa"].fillna(0.0).values,
                    "team_yrfi_rate_30": (roll["yrfi"] / games).fillna(0.0).values,
                    "team_runs_per_fi_30": (roll["runs"] / games).fillna(0.0).values,
                    "team_hits_per_fi_30": (roll["hits"] / games).fillna(0.0).values,
                    "team_obp_30": _safe_rate(roll["onbase"], roll["pa"]).values,
                    "team_walk_rate_30": _safe_rate(roll["walks"] + roll["hbp"], roll["pa"]).values,
                    "team_hardhit_rate_30": _safe_rate(roll["hardhits"], roll["bbe"]).values,
                    "team_barrel_rate_30": _safe_rate(roll["barrels"], roll["bbe"]).values,
                }
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _rolling_pitcher_features(pitcher_day: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in pitcher_day.groupby("pitcher", dropna=False):
        group = group.sort_values("game_date").copy()
        group["yrfi_allowed"] = (group["runs_allowed"] > 0).astype(int)
        shifted = group[
            [
                "yrfi_allowed",
                "runs_allowed",
                "batters_faced",
                "hits_allowed",
                "walks_allowed",
                "hbp_allowed",
                "onbase_allowed",
                "strikeouts",
                "bbe_allowed",
                "hardhits_allowed",
                "barrels_allowed",
            ]
        ].shift(1)
        roll = shifted.rolling(30, min_periods=1).sum()
        games = shifted["batters_faced"].rolling(30, min_periods=1).count().replace(0, np.nan)
        parts.append(
            pd.DataFrame(
                {
                    "game_date": group["game_date"].values,
                    "pitcher": group["pitcher"].values,
                    "bf_30": roll["batters_faced"].fillna(0.0).values,
                    "yrfi_allowed_rate_30": (roll["yrfi_allowed"] / games).fillna(0.0).values,
                    "runs_allowed_per_fi_30": (roll["runs_allowed"] / games).fillna(0.0).values,
                    "hits_allowed_per_fi_30": (roll["hits_allowed"] / games).fillna(0.0).values,
                    "obp_allowed_30": _safe_rate(roll["onbase_allowed"], roll["batters_faced"]).values,
                    "k_rate_30": _safe_rate(roll["strikeouts"], roll["batters_faced"]).values,
                    "bb_rate_30": _safe_rate(roll["walks_allowed"] + roll["hbp_allowed"], roll["batters_faced"]).values,
                    "hardhit_allowed_rate_30": _safe_rate(roll["hardhits_allowed"], roll["bbe_allowed"]).values,
                    "barrel_allowed_rate_30": _safe_rate(roll["barrels_allowed"], roll["bbe_allowed"]).values,
                }
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def latest_team_profiles(training_rows: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for side in ["away", "home"]:
        cols = [c for c in training_rows.columns if c.startswith(f"{side}_team_")]
        renamed = training_rows[["game_date", f"{side}_team", *cols]].rename(
            columns={f"{side}_team": "team", **{col: col.replace(f"{side}_", "") for col in cols}}
        )
        frames.append(renamed)
    rows = pd.concat(frames, ignore_index=True).dropna(subset=["team"])
    rows = rows.sort_values("game_date")
    idx = rows.groupby("team")["game_date"].idxmax()
    return rows.loc[idx].copy()


def latest_pitcher_profiles(training_rows: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for side in ["away_sp", "home_sp"]:
        id_col = f"{side.replace('_sp', '')}_starter_id"
        cols = [c for c in training_rows.columns if c.startswith(f"{side}_") and c not in {f"{side}_name"}]
        cols = [c for c in cols if c not in {f"{side}_game_date", f"{side}_pitcher"}]
        present = [c for c in [id_col, *cols] if c in training_rows.columns]
        if id_col not in present:
            continue
        renamed = training_rows[["game_date", *present]].rename(
            columns={id_col: "pitcher", **{col: col.replace(f"{side}_", "") for col in cols}}
        )
        frames.append(renamed)
    if not frames:
        return pd.DataFrame()
    rows = pd.concat(frames, ignore_index=True).dropna(subset=["pitcher"])
    rows = rows.sort_values("game_date")
    idx = rows.groupby("pitcher")["game_date"].idxmax()
    return rows.loc[idx].copy()


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan)).fillna(0.0)


def _series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(np.nan, index=df.index)


def _first_or_unknown(values: pd.Series) -> str:
    values = values.dropna()
    return str(values.iloc[0]) if not values.empty else "Unknown"


def _pitcher_name_fallback(values: pd.Series) -> str:
    values = values.dropna()
    return str(values.iloc[0]) if not values.empty else "TBD"
