from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import EXPERIMENTS_DIR
from .features import build_training_rows, feature_set_slug
from .model import summarize_backtest, train_model, walk_forward_backtest
from .utils import ensure_parent

DEFAULT_EXPERIMENT_FEATURE_SETS = ["baseline", "baseline+team_form", "baseline+starter_form", "all_free_statcast"]


@dataclass(frozen=True)
class ExperimentResult:
    feature_set: str
    training_path: Path
    model_path: Path
    backtest_path: Path
    summary_path: Path


def run_feature_set_experiments(
    feature_sets: list[str], statcast_path: Path, min_train_days: int = 45
) -> list[ExperimentResult]:
    results = []
    for feature_set in feature_sets:
        slug = feature_set_slug(feature_set)
        exp_dir = EXPERIMENTS_DIR / slug
        training_path = exp_dir / "training_rows.csv"
        model_path = exp_dir / "nrfi_model.joblib"
        backtest_path = exp_dir / "backtest_predictions.csv"
        summary_path = exp_dir / "summary.json"
        build_training_rows(statcast_path, training_path, feature_set=feature_set)
        train_model(training_path, model_path, feature_set=feature_set)
        predictions = walk_forward_backtest(training_path, min_train_days=min_train_days, feature_set=feature_set)
        predictions.to_csv(ensure_parent(backtest_path), index=False)
        summary = summarize_backtest(predictions)
        summary["feature_set"] = feature_set
        ensure_parent(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        results.append(ExperimentResult(feature_set, training_path, model_path, backtest_path, summary_path))
    return results
