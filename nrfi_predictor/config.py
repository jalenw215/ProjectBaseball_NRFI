from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"
LOGS_DIR = DATA_DIR / "logs"
EXPERIMENTS_DIR = DATA_DIR / "experiments"

STATCAST_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_TRAINING_FILE = PROCESSED_DIR / "training_rows.csv"
DEFAULT_MODEL_FILE = MODELS_DIR / "nrfi_model.joblib"
DEFAULT_PREDICTIONS_FILE = PREDICTIONS_DIR / "latest_predictions.csv"
DEFAULT_ODDS_FILE = RAW_DIR / "manual_nrfi_odds.csv"
DEFAULT_STATCAST_FILE = RAW_DIR / "statcast_history.csv"
DEFAULT_BACKTEST_FILE = PROCESSED_DIR / "backtest_predictions.csv"
DEFAULT_REFRESH_LOG = LOGS_DIR / "refresh.log"

FEATURE_COLUMNS = [
    "away_team_pa_30",
    "away_team_yrfi_rate_30",
    "away_team_obp_30",
    "away_team_hardhit_rate_30",
    "away_team_barrel_rate_30",
    "home_team_pa_30",
    "home_team_yrfi_rate_30",
    "home_team_obp_30",
    "home_team_hardhit_rate_30",
    "home_team_barrel_rate_30",
    "away_sp_bf_30",
    "away_sp_yrfi_allowed_rate_30",
    "away_sp_obp_allowed_30",
    "away_sp_k_rate_30",
    "away_sp_bb_rate_30",
    "home_sp_bf_30",
    "home_sp_yrfi_allowed_rate_30",
    "home_sp_obp_allowed_30",
    "home_sp_k_rate_30",
    "home_sp_bb_rate_30",
    "park_run_factor",
    "temperature_2m",
    "wind_speed_10m",
]

FEATURE_GROUPS = {
    "baseline": FEATURE_COLUMNS,
    "team_form": [
        "away_team_runs_per_fi_30",
        "away_team_hits_per_fi_30",
        "away_team_walk_rate_30",
        "home_team_runs_per_fi_30",
        "home_team_hits_per_fi_30",
        "home_team_walk_rate_30",
    ],
    "starter_form": [
        "away_sp_runs_allowed_per_fi_30",
        "away_sp_hits_allowed_per_fi_30",
        "away_sp_hardhit_allowed_rate_30",
        "away_sp_barrel_allowed_rate_30",
        "home_sp_runs_allowed_per_fi_30",
        "home_sp_hits_allowed_per_fi_30",
        "home_sp_hardhit_allowed_rate_30",
        "home_sp_barrel_allowed_rate_30",
    ],
}

FEATURE_SET_ALIASES = {"all": "all_free_statcast"}
DEFAULT_FEATURE_SET = "baseline"
