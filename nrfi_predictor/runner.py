from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .artifacts import SupabaseArtifacts
from .config import (
    DEFAULT_BACKTEST_FILE,
    DEFAULT_FETCH_STATE_FILE,
    DEFAULT_FEATURE_SET,
    DEFAULT_MODEL_FILE,
    DEFAULT_PREDICTIONS_FILE,
    DEFAULT_REFRESH_LOG,
    DEFAULT_STATCAST_FILE,
    DEFAULT_TRAINING_FILE,
    RAW_DIR,
)
from .data_sources import fetch_schedule, fetch_statcast_csv
from .experiments import DEFAULT_EXPERIMENT_FEATURE_SETS, run_feature_set_experiments
from .features import build_training_rows
from .model import summarize_backtest, train_model, walk_forward_backtest
from .predict import predict_for_date
from .utils import ensure_parent, parse_date, today_iso

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class RunnerConfig:
    start_date: date = date(2024, 1, 1)
    end_date: date = date.today()
    statcast_file: Path = DEFAULT_STATCAST_FILE
    training_file: Path = DEFAULT_TRAINING_FILE
    model_file: Path = DEFAULT_MODEL_FILE
    predictions_file: Path = DEFAULT_PREDICTIONS_FILE
    backtest_file: Path = DEFAULT_BACKTEST_FILE
    log_file: Path = DEFAULT_REFRESH_LOG
    fetch_state_file: Path = DEFAULT_FETCH_STATE_FILE
    min_fetch_interval_hours: int = 24
    min_train_days: int = 45
    feature_set: str = DEFAULT_FEATURE_SET


@dataclass(frozen=True)
class StepResult:
    name: str
    status: str
    detail: str


def two_season_start(today: date | None = None) -> date:
    today = today or date.today()
    return date(today.year - 2, 1, 1)


def default_config(end_date: date | None = None) -> RunnerConfig:
    end_date = end_date or date.today()
    return RunnerConfig(start_date=two_season_start(end_date), end_date=end_date)


class PipelineRunner:
    def __init__(self, config: RunnerConfig | None = None, progress: ProgressCallback | None = None):
        self.config = config or default_config()
        self.progress = progress
        self.artifacts = SupabaseArtifacts()

    def fetch_historical_data(self) -> StepResult:
        return self._run_step("Fetch historical data", self._fetch_historical_data)

    def build_training_set(self) -> StepResult:
        return self._run_step("Build training set", self._build_training_set)

    def train_model(self) -> StepResult:
        return self._run_step("Train model", self._train_model)

    def run_backtest(self) -> StepResult:
        return self._run_step("Run backtest", self._run_backtest)

    def run_experiments(self, feature_sets: list[str] | None = None) -> StepResult:
        feature_sets = feature_sets or DEFAULT_EXPERIMENT_FEATURE_SETS
        return self._run_step("Run feature experiments", lambda: self._run_experiments(feature_sets))

    def predict_today(self, prediction_date: str | None = None, model_path: Path | None = None) -> StepResult:
        prediction_date = prediction_date or today_iso()
        return self._run_step("Predict today", lambda: self._predict_today(prediction_date, model_path=model_path))

    def morning_refresh(self, prediction_date: str | None = None, model_path: Path | None = None) -> list[StepResult]:
        prediction_date = prediction_date or today_iso()
        return self._run_sequence(
            [
                self.fetch_historical_data,
                self.build_training_set,
                self.train_model,
                self.run_backtest,
                lambda: self.predict_today(prediction_date, model_path=model_path),
            ]
        )

    def lineup_refresh(self, prediction_date: str | None = None) -> list[StepResult]:
        prediction_date = prediction_date or today_iso()
        results: list[StepResult] = []
        if not self.config.training_file.exists() or not self.config.model_file.exists():
            results.extend([self.build_training_set(), self.train_model()])
        results.append(self.predict_today(prediction_date))
        return results

    def full_refresh(self, prediction_date: str | None = None, model_path: Path | None = None) -> list[StepResult]:
        return self.morning_refresh(prediction_date, model_path=model_path)

    def _run_sequence(self, steps: list[Callable[[], StepResult]]) -> list[StepResult]:
        results = []
        for step in steps:
            result = step()
            results.append(result)
            if result.status != "ok":
                break
        return results

    def _run_step(self, name: str, action: Callable[[], str]) -> StepResult:
        started_at = datetime.now(timezone.utc)
        self._log(f"START {name}")
        self._emit(f"Starting: {name}")
        try:
            detail = action()
        except Exception as exc:
            self._log(f"ERROR {name}: {exc}\n{traceback.format_exc()}")
            self._emit(f"Failed: {name} - {exc}")
            self.artifacts.log_pipeline_run(
                name,
                "error",
                detail=str(exc),
                error=str(exc),
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
            return StepResult(name=name, status="error", detail=str(exc))
        self._log(f"DONE {name}: {detail}")
        self._emit(f"Finished: {name}")
        self.artifacts.log_pipeline_run(
            name,
            "ok",
            detail=detail,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        return StepResult(name=name, status="ok", detail=detail)

    def _fetch_historical_data(self) -> str:
        self.artifacts.download_if_missing("statcast", self.config.statcast_file)
        self.artifacts.download_if_missing("fetch_state", self.config.fetch_state_file)
        cooldown = self._historical_fetch_cooldown()
        if cooldown is not None:
            return cooldown
        chunk_paths = []
        for start, end in month_chunks(self.config.start_date, self.config.end_date):
            chunk_path = RAW_DIR / f"statcast_{start}_{end}.csv"
            if chunk_path.exists() and chunk_path.stat().st_size > 0:
                self._log(f"SKIP existing chunk {chunk_path.name}")
            else:
                self._emit(f"Fetching first-inning Statcast {start} to {end}")
                fetch_statcast_csv(start.isoformat(), end.isoformat(), chunk_path)
            chunk_paths.append(chunk_path)
        combined = combine_statcast_chunks(chunk_paths, self.config.statcast_file)
        self._write_fetch_state(combined)
        self.artifacts.upload("statcast", combined, metadata={"start_date": self.config.start_date.isoformat(), "end_date": self.config.end_date.isoformat()})
        return f"wrote {combined}"

    def _historical_fetch_cooldown(self) -> str | None:
        if self.config.min_fetch_interval_hours <= 0 or not self.config.statcast_file.exists():
            return None
        remote_state = self.artifacts.read_refresh_state()
        state = read_fetch_state(self.config.fetch_state_file)
        fetched_at_raw = remote_state.get("last_historical_fetch_at") or state.get("fetched_at")
        if not fetched_at_raw:
            return None
        try:
            fetched_at = datetime.fromisoformat(str(fetched_at_raw))
        except ValueError:
            return None
        elapsed = datetime.now() - fetched_at
        cooldown = timedelta(hours=self.config.min_fetch_interval_hours)
        if elapsed >= cooldown:
            return None
        next_allowed = fetched_at + cooldown
        detail = (
            f"skipped historical fetch; last successful fetch was {fetched_at.isoformat(timespec='seconds')} "
            f"and next fetch is allowed after {next_allowed.isoformat(timespec='seconds')}"
        )
        self._emit(detail)
        self._log(f"SKIP Fetch historical data: {detail}")
        return detail

    def _write_fetch_state(self, path: Path) -> None:
        state = {
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "statcast_file": str(path),
            "start_date": self.config.start_date.isoformat(),
            "end_date": self.config.end_date.isoformat(),
        }
        ensure_parent(self.config.fetch_state_file).write_text(json.dumps(state, indent=2), encoding="utf-8")
        self.artifacts.upload("fetch_state", self.config.fetch_state_file, metadata={"state": "historical_fetch"})
        self.artifacts.upsert_refresh_state(
            {
                "last_historical_fetch_at": state["fetched_at"],
                "cooldown_hours": self.config.min_fetch_interval_hours,
                "latest_statcast_path": self.artifacts.storage_path("statcast"),
                "metadata": {"start_date": state["start_date"], "end_date": state["end_date"]},
            }
        )

    def _build_training_set(self) -> str:
        self.artifacts.download_if_missing("statcast", self.config.statcast_file)
        if not self.config.statcast_file.exists():
            chunk_paths = sorted(RAW_DIR.glob("statcast_????-??-??_????-??-??.csv"))
            if not chunk_paths:
                raise FileNotFoundError(f"Missing Statcast file: {self.config.statcast_file}")
            self._emit("Combining monthly Statcast files before training")
            combine_statcast_chunks(chunk_paths, self.config.statcast_file)
        output = build_training_rows(self.config.statcast_file, self.config.training_file, feature_set=self.config.feature_set)
        self.artifacts.upload("training", output, metadata={"feature_set": self.config.feature_set})
        self.artifacts.upsert_refresh_state({"latest_training_path": self.artifacts.storage_path("training")})
        return f"wrote {output}"

    def _train_model(self) -> str:
        self.artifacts.download_if_missing("training", self.config.training_file)
        if not self.config.training_file.exists():
            raise FileNotFoundError(f"Missing training file: {self.config.training_file}")
        output = train_model(self.config.training_file, self.config.model_file, feature_set=self.config.feature_set)
        self.artifacts.upload("model", output, metadata={"feature_set": self.config.feature_set})
        self.artifacts.upsert_refresh_state({"latest_model_path": self.artifacts.storage_path("model")})
        return f"wrote {output}"

    def _run_backtest(self) -> str:
        self.artifacts.download_if_missing("training", self.config.training_file)
        if not self.config.training_file.exists():
            raise FileNotFoundError(f"Missing training file: {self.config.training_file}")
        predictions = walk_forward_backtest(
            self.config.training_file,
            min_train_days=self.config.min_train_days,
            feature_set=self.config.feature_set,
        )
        predictions.to_csv(ensure_parent(self.config.backtest_file), index=False)
        summary_path = self.config.backtest_file.with_suffix(".summary.json")
        ensure_parent(summary_path).write_text(json.dumps(summarize_backtest(predictions), indent=2), encoding="utf-8")
        self.artifacts.upload("backtest", self.config.backtest_file, metadata={"feature_set": self.config.feature_set})
        return f"wrote {self.config.backtest_file}"

    def _run_experiments(self, feature_sets: list[str]) -> str:
        if not self.config.statcast_file.exists():
            raise FileNotFoundError(f"Missing Statcast file: {self.config.statcast_file}")
        results = run_feature_set_experiments(
            feature_sets,
            statcast_path=self.config.statcast_file,
            min_train_days=self.config.min_train_days,
        )
        return "wrote " + ", ".join(str(result.summary_path) for result in results)

    def _predict_today(self, prediction_date: str, model_path: Path | None = None) -> str:
        model_path = model_path or self.config.model_file
        self.artifacts.download_if_missing("training", self.config.training_file)
        if model_path == self.config.model_file:
            self.artifacts.download_if_missing("model", model_path)
        if not self.config.training_file.exists():
            raise FileNotFoundError(f"Missing training file: {self.config.training_file}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")
        output = predict_for_date(
            prediction_date,
            training_path=self.config.training_file,
            model_path=model_path,
            output_path=self.config.predictions_file,
        )
        schedule_path = self.config.statcast_file.parent / f"schedule_{prediction_date}.json"
        ensure_parent(schedule_path).write_text(json.dumps(fetch_schedule(prediction_date), indent=2), encoding="utf-8")
        self.artifacts.upload("predictions", output, metadata={"prediction_date": prediction_date})
        self.artifacts.upsert_refresh_state({"latest_predictions_path": self.artifacts.storage_path("predictions")})
        return f"wrote {output}"

    def _emit(self, message: str) -> None:
        if self.progress:
            self.progress(message)

    def _log(self, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        ensure_parent(self.config.log_file)
        with self.config.log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")


def month_chunks(start_date: date | str, end_date: date | str) -> Iterable[tuple[date, date]]:
    start = parse_date(start_date)
    end = parse_date(end_date)
    current = start
    while current <= end:
        next_month = date(current.year + 1, 1, 1) if current.month == 12 else date(current.year, current.month + 1, 1)
        chunk_end = min(next_month - timedelta(days=1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def combine_statcast_chunks(paths: Iterable[Path], output_path: Path) -> Path:
    frames = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("No Statcast chunk data was available to combine.")
    combined = pd.concat(frames, ignore_index=True)
    key_cols = [col for col in ["game_pk", "at_bat_number", "pitch_number", "batter", "pitcher"] if col in combined.columns]
    if key_cols:
        combined = combined.drop_duplicates(subset=key_cols)
    combined.to_csv(ensure_parent(output_path), index=False)
    return output_path


def read_recent_log(log_file: Path = DEFAULT_REFRESH_LOG, max_lines: int = 80) -> str:
    if not log_file.exists():
        return "No refresh log yet."
    lines = log_file.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-max_lines:])


def read_fetch_state(fetch_state_file: Path = DEFAULT_FETCH_STATE_FILE) -> dict:
    if not fetch_state_file.exists():
        return {}
    try:
        return json.loads(fetch_state_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
