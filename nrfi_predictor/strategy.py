from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_MODEL_FILE, EXPERIMENTS_DIR
from .features import feature_set_slug

BEST_TOP5 = "Best Top-5 NRFI"
BEST_PROBABILITY = "Best Probability Accuracy"
BASELINE = "Baseline"
ALL_FEATURES = "All Features"

MODEL_STRATEGIES = [BEST_TOP5, BEST_PROBABILITY, BASELINE, ALL_FEATURES]


@dataclass(frozen=True)
class ModelStrategySelection:
    strategy: str
    feature_set: str
    model_path: Path
    summary_path: Path | None
    explanation: str


def resolve_model_strategy(
    strategy: str,
    experiments_dir: Path = EXPERIMENTS_DIR,
    default_model_path: Path = DEFAULT_MODEL_FILE,
) -> ModelStrategySelection:
    if strategy == BASELINE:
        return _resolve_named_feature_set(strategy, "baseline", experiments_dir, default_model_path, allow_default_fallback=True)
    if strategy == ALL_FEATURES:
        return _resolve_named_feature_set(strategy, "all_free_statcast", experiments_dir, default_model_path, allow_default_fallback=False)

    summaries = _experiment_summaries(experiments_dir)
    if not summaries:
        raise FileNotFoundError("No feature experiment results found. Run Feature Experiments first.")

    if strategy == BEST_TOP5:
        winner = sorted(
            summaries,
            key=lambda item: (
                -float(item["summary"].get("top5_nrfi_rate", 0.0)),
                -float(item["summary"].get("top5_nrfi_hits", 0.0)),
                float(item["summary"].get("brier", 999.0)),
            ),
        )[0]
        explanation = "Selected the experiment with the best daily top-5 NRFI hit rate."
    elif strategy == BEST_PROBABILITY:
        winner = sorted(
            summaries,
            key=lambda item: (float(item["summary"].get("brier", 999.0)), float(item["summary"].get("log_loss", 999.0))),
        )[0]
        explanation = "Selected the experiment with the best calibrated probability score."
    else:
        raise ValueError(f"Unknown model strategy: {strategy}")

    model_path = winner["summary_path"].parent / "nrfi_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing experiment model for {winner['feature_set']}. Run Feature Experiments first.")
    return ModelStrategySelection(strategy, winner["feature_set"], model_path, winner["summary_path"], explanation)


def _resolve_named_feature_set(
    strategy: str,
    feature_set: str,
    experiments_dir: Path,
    default_model_path: Path,
    allow_default_fallback: bool,
) -> ModelStrategySelection:
    exp_dir = experiments_dir / feature_set_slug(feature_set)
    model_path = exp_dir / "nrfi_model.joblib"
    summary_path = exp_dir / "summary.json"
    if model_path.exists():
        return ModelStrategySelection(strategy, feature_set, model_path, summary_path if summary_path.exists() else None, f"Using the {feature_set} experiment model.")
    if allow_default_fallback and default_model_path.exists():
        return ModelStrategySelection(strategy, feature_set, default_model_path, None, "Using the default baseline model because no baseline experiment model was found.")
    raise FileNotFoundError(f"Missing experiment model for {feature_set}. Run Feature Experiments first.")


def _experiment_summaries(experiments_dir: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(experiments_dir.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        feature_set = str(summary.get("feature_set", summary_path.parent.name))
        rows.append({"feature_set": feature_set, "summary": summary, "summary_path": summary_path})
    return rows
