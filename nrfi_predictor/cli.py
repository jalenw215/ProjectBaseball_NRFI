from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_FEATURE_SET, DEFAULT_MODEL_FILE, DEFAULT_PREDICTIONS_FILE, DEFAULT_TRAINING_FILE, PROCESSED_DIR, RAW_DIR
from .data_sources import fetch_schedule, fetch_statcast_csv
from .experiments import DEFAULT_EXPERIMENT_FEATURE_SETS, run_feature_set_experiments
from .features import build_training_rows
from .model import summarize_backtest, train_model, walk_forward_backtest
from .predict import predict_for_date
from .reporting import format_daily_report
from .runner import PipelineRunner
from .utils import ensure_parent, today_iso


def main() -> None:
    parser = argparse.ArgumentParser(description="MLB daily NRFI/YRFI predictor")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_statcast = sub.add_parser("fetch-statcast", help="Download first-inning Baseball Savant Statcast CSV")
    fetch_statcast.add_argument("--start-date", required=True)
    fetch_statcast.add_argument("--end-date", required=True)
    fetch_statcast.add_argument("--output", type=Path)

    schedule = sub.add_parser("fetch-schedule", help="Fetch MLB schedule/probable pitchers")
    schedule.add_argument("--date", default=today_iso())
    schedule.add_argument("--output", type=Path)

    training = sub.add_parser("build-training", help="Build leakage-safe game-level NRFI rows")
    training.add_argument("--statcast", required=True, type=Path)
    training.add_argument("--output", type=Path, default=DEFAULT_TRAINING_FILE)
    training.add_argument("--feature-set", default=DEFAULT_FEATURE_SET)

    train = sub.add_parser("train", help="Train calibrated NRFI probability model")
    train.add_argument("--training", type=Path, default=DEFAULT_TRAINING_FILE)
    train.add_argument("--model", type=Path, default=DEFAULT_MODEL_FILE)
    train.add_argument("--feature-set", default=DEFAULT_FEATURE_SET)

    backtest = sub.add_parser("backtest", help="Run walk-forward backtest")
    backtest.add_argument("--training", type=Path, default=DEFAULT_TRAINING_FILE)
    backtest.add_argument("--output", type=Path, default=PROCESSED_DIR / "backtest_predictions.csv")
    backtest.add_argument("--min-train-days", type=int, default=45)
    backtest.add_argument("--feature-set", default=DEFAULT_FEATURE_SET)

    experiments = sub.add_parser("experiments", help="Build, train, and backtest feature-set experiments")
    experiments.add_argument("--statcast", required=True, type=Path)
    experiments.add_argument("--feature-set", action="append", dest="feature_sets")
    experiments.add_argument("--min-train-days", type=int, default=45)

    predict = sub.add_parser("predict-today", help="Generate daily NRFI/YRFI rankings")
    predict.add_argument("--date", default=today_iso())
    predict.add_argument("--training", type=Path, default=DEFAULT_TRAINING_FILE)
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL_FILE)
    predict.add_argument("--output", type=Path, default=DEFAULT_PREDICTIONS_FILE)

    report = sub.add_parser("report", help="Print a text report from a predictions CSV")
    report.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_FILE)
    report.add_argument("--limit", type=int, default=10)

    runner = sub.add_parser("run", help="Run no-terminal pipeline jobs")
    runner.add_argument("job", choices=["morning-refresh", "lineup-refresh", "full-refresh", "predict-only"])
    runner.add_argument("--date", default=today_iso())

    args = parser.parse_args()

    if args.command == "fetch-statcast":
        path = fetch_statcast_csv(args.start_date, args.end_date, args.output)
        print(f"Wrote {path}")
    elif args.command == "fetch-schedule":
        import json

        games = fetch_schedule(args.date)
        output = args.output or RAW_DIR / f"schedule_{args.date}.json"
        ensure_parent(output).write_text(json.dumps(games, indent=2), encoding="utf-8")
        print(f"Wrote {output}")
    elif args.command == "build-training":
        path = build_training_rows(args.statcast, args.output, feature_set=args.feature_set)
        print(f"Wrote {path}")
    elif args.command == "train":
        path = train_model(args.training, args.model, feature_set=args.feature_set)
        print(f"Wrote {path}")
    elif args.command == "backtest":
        predictions = walk_forward_backtest(args.training, min_train_days=args.min_train_days, feature_set=args.feature_set)
        predictions.to_csv(ensure_parent(args.output), index=False)
        print(f"Wrote {args.output}")
        print(summarize_backtest(predictions))
    elif args.command == "experiments":
        feature_sets = args.feature_sets or DEFAULT_EXPERIMENT_FEATURE_SETS
        results = run_feature_set_experiments(feature_sets, statcast_path=args.statcast, min_train_days=args.min_train_days)
        for result in results:
            print(f"{result.feature_set}: wrote {result.summary_path}")
    elif args.command == "predict-today":
        path = predict_for_date(args.date, args.training, args.model, output_path=args.output)
        print(f"Wrote {path}")
    elif args.command == "report":
        print(format_daily_report(args.predictions, args.limit))
    elif args.command == "run":
        pipeline = PipelineRunner(progress=print)
        if args.job == "morning-refresh":
            results = pipeline.morning_refresh(args.date)
        elif args.job == "lineup-refresh":
            results = pipeline.lineup_refresh(args.date)
        elif args.job == "full-refresh":
            results = pipeline.full_refresh(args.date)
        else:
            results = [pipeline.predict_today(args.date)]
        for result in results:
            print(f"{result.status.upper()} {result.name}: {result.detail}")


if __name__ == "__main__":
    main()
