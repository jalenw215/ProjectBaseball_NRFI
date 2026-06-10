from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

from nrfi_predictor.runner import PipelineRunner, RunnerConfig


def test_historical_fetch_skips_within_cooldown(tmp_path: Path, monkeypatch):
    statcast_file = tmp_path / "statcast_history.csv"
    statcast_file.write_text("game_pk\n1\n", encoding="utf-8")
    state_file = tmp_path / "fetch_state.json"
    state_file.write_text(json.dumps({"fetched_at": datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")
    calls = []
    monkeypatch.setattr("nrfi_predictor.runner.fetch_statcast_csv", lambda *args, **kwargs: calls.append(args))

    runner = PipelineRunner(config=_config(tmp_path, statcast_file, state_file))
    result = runner.fetch_historical_data()

    assert result.status == "ok"
    assert "skipped historical fetch" in result.detail
    assert calls == []


def test_historical_fetch_runs_after_cooldown(tmp_path: Path, monkeypatch):
    statcast_file = tmp_path / "statcast_history.csv"
    state_file = tmp_path / "fetch_state.json"
    old_fetch = datetime.now() - timedelta(hours=25)
    state_file.write_text(json.dumps({"fetched_at": old_fetch.isoformat(timespec="seconds")}), encoding="utf-8")

    def fake_fetch(start_date, end_date, output_path):
        output_path.write_text("game_pk,at_bat_number,pitch_number,batter,pitcher\n1,1,1,1,2\n", encoding="utf-8")
        return output_path

    monkeypatch.setattr("nrfi_predictor.runner.fetch_statcast_csv", fake_fetch)
    runner = PipelineRunner(config=_config(tmp_path, statcast_file, state_file))
    result = runner.fetch_historical_data()

    assert result.status == "ok"
    assert statcast_file.exists()
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["statcast_file"] == str(statcast_file)


def test_historical_fetch_uses_remote_cooldown_state(tmp_path: Path, monkeypatch):
    statcast_file = tmp_path / "statcast_history.csv"
    statcast_file.write_text("game_pk\n1\n", encoding="utf-8")
    state_file = tmp_path / "fetch_state.json"
    calls = []
    monkeypatch.setattr("nrfi_predictor.runner.fetch_statcast_csv", lambda *args, **kwargs: calls.append(args))
    runner = PipelineRunner(config=_config(tmp_path, statcast_file, state_file))
    runner.artifacts = FakeArtifacts({"last_historical_fetch_at": datetime.now().isoformat(timespec="seconds")})

    result = runner.fetch_historical_data()

    assert result.status == "ok"
    assert "skipped historical fetch" in result.detail
    assert calls == []


def _config(tmp_path: Path, statcast_file: Path, state_file: Path) -> RunnerConfig:
    return RunnerConfig(
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 1),
        statcast_file=statcast_file,
        training_file=tmp_path / "training.csv",
        model_file=tmp_path / "model.joblib",
        predictions_file=tmp_path / "predictions.csv",
        backtest_file=tmp_path / "backtest.csv",
        log_file=tmp_path / "refresh.log",
        fetch_state_file=state_file,
        min_fetch_interval_hours=24,
    )


class FakeArtifacts:
    def __init__(self, state):
        self.state = state

    def download_if_missing(self, *args, **kwargs):
        return False

    def read_refresh_state(self):
        return self.state

    def log_pipeline_run(self, *args, **kwargs):
        return True
