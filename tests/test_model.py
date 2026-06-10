from pathlib import Path

import pandas as pd

from nrfi_predictor.features import build_training_rows
from nrfi_predictor.model import summarize_backtest, train_model, walk_forward_backtest
from tests.test_features import _statcast_games


def test_train_model_and_backtest(tmp_path: Path):
    statcast = tmp_path / "statcast.csv"
    training = tmp_path / "training.csv"
    model = tmp_path / "nrfi_model.joblib"
    pd.DataFrame(_statcast_games(days=70)).to_csv(statcast, index=False)
    build_training_rows(statcast, training)

    train_model(training, model)
    assert model.exists()

    predictions = walk_forward_backtest(training, min_train_days=20)
    assert not predictions.empty
    assert {"nrfi_probability", "yrfi_probability", "rank"}.issubset(predictions.columns)
    summary = summarize_backtest(predictions)
    assert summary["rows"] > 0
    assert "brier" in summary
