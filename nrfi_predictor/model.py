from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_FEATURE_SET, DEFAULT_MODEL_FILE
from .features import resolve_feature_columns
from .utils import ensure_parent


def train_model(training_path: Path, model_path: Path | None = None, feature_set: str = DEFAULT_FEATURE_SET) -> Path:
    model_path = model_path or DEFAULT_MODEL_FILE
    feature_columns = resolve_feature_columns(feature_set)
    rows = pd.read_csv(training_path, parse_dates=["game_date"]).dropna(subset=["target_nrfi"])
    _ensure_feature_columns(rows, feature_columns)
    X = rows[feature_columns]
    y = rows["target_nrfi"].astype(int)
    model = _fit_model(X, y, calibrate=True)
    joblib.dump({"model": model, "features": feature_columns, "feature_set": feature_set}, ensure_parent(model_path))
    return model_path


def load_model(model_path: Path = DEFAULT_MODEL_FILE):
    payload = joblib.load(model_path)
    return payload["model"], payload["features"]


def predict_probabilities(model, rows: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    _ensure_feature_columns(rows, feature_columns)
    return model.predict_proba(rows[feature_columns])[:, 1]


def walk_forward_backtest(
    training_path: Path, min_train_days: int = 45, feature_set: str = DEFAULT_FEATURE_SET
) -> pd.DataFrame:
    feature_columns = resolve_feature_columns(feature_set)
    rows = pd.read_csv(training_path, parse_dates=["game_date"]).sort_values("game_date").dropna(subset=["target_nrfi"])
    _ensure_feature_columns(rows, feature_columns)
    outputs = []
    for current_date in sorted(rows["game_date"].dt.date.unique()):
        train = rows[rows["game_date"].dt.date < current_date]
        test = rows[rows["game_date"].dt.date == current_date]
        if train["game_date"].dt.date.nunique() < min_train_days or test.empty or train["target_nrfi"].nunique() < 2:
            continue
        model = _fit_model(train[feature_columns], train["target_nrfi"].astype(int), calibrate=False)
        day = test.copy()
        day["nrfi_probability"] = model.predict_proba(day[feature_columns])[:, 1]
        day["yrfi_probability"] = 1.0 - day["nrfi_probability"]
        day["rank"] = day["nrfi_probability"].rank(ascending=False, method="first").astype(int)
        day["feature_set"] = feature_set
        outputs.append(day)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def summarize_backtest(predictions: pd.DataFrame) -> dict[str, float]:
    if predictions.empty:
        return {}
    y = predictions["target_nrfi"].astype(int)
    p = predictions["nrfi_probability"].clip(0.001, 0.999)
    top5 = predictions[predictions["rank"] <= 5]
    summary = {
        "rows": float(len(predictions)),
        "nrfi_events": float(y.sum()),
        "yrfi_events": float((1 - y).sum()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "top5_nrfi_rate": float(top5["target_nrfi"].mean()) if not top5.empty else 0.0,
        "top5_nrfi_hits": float(top5["target_nrfi"].sum()) if not top5.empty else 0.0,
    }
    for label, low, high in [
        ("calibration_under_50", 0.0, 0.50),
        ("calibration_50_55", 0.50, 0.55),
        ("calibration_55_60", 0.55, 0.60),
        ("calibration_60_plus", 0.60, 1.01),
    ]:
        bucket = predictions[(p >= low) & (p < high)]
        summary[f"{label}_rows"] = float(len(bucket))
        summary[f"{label}_actual_rate"] = float(bucket["target_nrfi"].mean()) if not bucket.empty else 0.0
        summary[f"{label}_avg_probability"] = float(bucket["nrfi_probability"].mean()) if not bucket.empty else 0.0
    return summary


def _fit_model(X: pd.DataFrame, y: pd.Series, calibrate: bool):
    base = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    if not calibrate:
        return base.fit(X, y)
    min_class = int(y.value_counts().min()) if y.nunique() == 2 else 0
    if min_class < 2:
        return base.fit(X, y)
    cv = min(5, min_class)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
    model.fit(X, y)
    return model


def _ensure_feature_columns(rows: pd.DataFrame, feature_columns: list[str]) -> None:
    missing = [col for col in feature_columns if col not in rows.columns]
    if missing:
        raise ValueError(f"Rows are missing feature columns: {missing}")
