from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nrfi_predictor.config import (
    DEFAULT_BACKTEST_FILE,
    DEFAULT_MODEL_FILE,
    DEFAULT_PREDICTIONS_FILE,
    DEFAULT_REFRESH_LOG,
    DEFAULT_STATCAST_FILE,
    DEFAULT_TRAINING_FILE,
    EXPERIMENTS_DIR,
)
from nrfi_predictor.runner import PipelineRunner, read_recent_log
from nrfi_predictor.strategy import BEST_TOP5, MODEL_STRATEGIES, resolve_model_strategy


st.set_page_config(page_title="MLB NRFI/YRFI Predictor", layout="wide")
st.title("MLB NRFI/YRFI Predictor")
st.caption("Research probabilities only. Use backtests and calibration before treating anything as betting signal.")


def run_job(label: str, action):
    status = st.status(label, expanded=True)

    def progress(message: str) -> None:
        status.write(message)

    runner = PipelineRunner(progress=progress)
    results = action(runner)
    errors = [result for result in results if result.status != "ok"]
    if errors:
        status.update(label=f"{label} finished with errors", state="error")
        for result in errors:
            st.error(f"{result.name}: {result.detail}")
    else:
        status.update(label=f"{label} complete", state="complete")
        st.success("Done. Refreshing dashboard data.")
        st.rerun()


st.subheader("No-Terminal Controls")
st.write("Use these buttons instead of Terminal. Long data pulls can take a while, especially the first historical fetch.")

strategy = st.selectbox("Model Strategy", MODEL_STRATEGIES, index=MODEL_STRATEGIES.index(BEST_TOP5))
strategy_selection = None
try:
    strategy_selection = resolve_model_strategy(strategy)
    st.caption(
        f"Using `{strategy_selection.feature_set}` from `{strategy_selection.model_path}`. "
        f"{strategy_selection.explanation}"
    )
except Exception as exc:
    st.warning(f"{strategy}: {exc}")

control_cols = st.columns(3)
with control_cols[0]:
    if st.button("Fetch Historical Data", use_container_width=True):
        run_job("Fetching two seasons of first-inning Statcast data", lambda r: [r.fetch_historical_data()])
    if st.button("Build Training Set", use_container_width=True):
        run_job("Building NRFI training rows", lambda r: [r.build_training_set()])
with control_cols[1]:
    if st.button("Train Model", use_container_width=True):
        run_job("Training NRFI model", lambda r: [r.train_model()])
    if st.button("Run Backtest", use_container_width=True):
        run_job("Running walk-forward backtest", lambda r: [r.run_backtest()])
with control_cols[2]:
    if st.button("Predict Today", use_container_width=True):
        if strategy_selection is None:
            st.error("No model is available for the selected strategy. Run Feature Experiments first or choose Baseline.")
        else:
            run_job("Generating today's predictions", lambda r: [r.predict_today(model_path=strategy_selection.model_path)])
    if st.button("Full Refresh", type="primary", use_container_width=True):
        if strategy_selection is None:
            st.error("No model is available for the selected strategy. Run Feature Experiments first or choose Baseline.")
        else:
            run_job("Running full refresh", lambda r: r.full_refresh(model_path=strategy_selection.model_path))

if st.button("Run Feature Experiments", use_container_width=True):
    run_job("Running feature group experiments", lambda r: [r.run_experiments()])

status_cols = st.columns(4)
status_cols[0].metric("Historical Data", "Ready" if DEFAULT_STATCAST_FILE.exists() else "Missing")
status_cols[1].metric("Training Set", "Ready" if DEFAULT_TRAINING_FILE.exists() else "Missing")
status_cols[2].metric("Model", "Ready" if DEFAULT_MODEL_FILE.exists() else "Missing")
status_cols[3].metric("Predictions", "Ready" if DEFAULT_PREDICTIONS_FILE.exists() else "Missing")

with st.expander("Refresh Log", expanded=not DEFAULT_PREDICTIONS_FILE.exists()):
    st.code(read_recent_log(DEFAULT_REFRESH_LOG), language="text")

if not DEFAULT_PREDICTIONS_FILE.exists():
    st.info("No predictions yet. Click Full Refresh to fetch data, train the model, backtest, and generate today's rankings.")
    st.stop()

df = pd.read_csv(DEFAULT_PREDICTIONS_FILE)
if df.empty:
    st.warning("The prediction file exists but is empty. Run Predict Today or Full Refresh.")
    st.stop()

df = df.sort_values("rank")

st.subheader("Today's NRFI/YRFI Rankings")
top_n = st.slider("Show top games", 5, 30, 15)
tiers = sorted(df["confidence_tier"].dropna().unique())
selected_tiers = st.multiselect("Confidence tiers", tiers, default=tiers)
filtered = df[df["confidence_tier"].isin(selected_tiers)].head(top_n)

metric_cols = st.columns(4)
metric_cols[0].metric("Games", f"{len(df):,}")
metric_cols[1].metric("Top NRFI", f"{df['nrfi_probability'].max():.1%}")
metric_cols[2].metric("Top YRFI", f"{df['yrfi_probability'].max():.1%}")
value_total = int(df.get("nrfi_value_flag", pd.Series(dtype=bool)).fillna(False).sum()) + int(
    df.get("yrfi_value_flag", pd.Series(dtype=bool)).fillna(False).sum()
)
metric_cols[3].metric("Value Flags", f"{value_total}")

feature_set = df.get("feature_set", pd.Series(["baseline"])).dropna()
st.caption(f"Active feature set: {feature_set.iloc[0] if not feature_set.empty else 'baseline'}")

display_cols = [
    "rank",
    "away_team",
    "home_team",
    "nrfi_probability",
    "yrfi_probability",
    "confidence_tier",
    "away_starter",
    "home_starter",
    "venue_name",
    "matchup_note",
]
optional_cols = [
    "nrfi_american_odds",
    "nrfi_implied_probability",
    "nrfi_book",
    "nrfi_value_flag",
    "yrfi_american_odds",
    "yrfi_implied_probability",
    "yrfi_book",
    "yrfi_value_flag",
]
display_cols.extend([c for c in optional_cols if c in df.columns])

st.dataframe(
    filtered[display_cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "nrfi_probability": st.column_config.ProgressColumn("NRFI probability", format="%.1f%%", min_value=0, max_value=1),
        "yrfi_probability": st.column_config.ProgressColumn("YRFI probability", format="%.1f%%", min_value=0, max_value=1),
        "nrfi_implied_probability": st.column_config.ProgressColumn("NRFI implied", format="%.1f%%", min_value=0, max_value=1),
        "yrfi_implied_probability": st.column_config.ProgressColumn("YRFI implied", format="%.1f%%", min_value=0, max_value=1),
    },
)

st.subheader("Model Signals")
signal_cols = [
    "away_team",
    "home_team",
    "away_team_yrfi_rate_30",
    "home_team_yrfi_rate_30",
    "away_sp_yrfi_allowed_rate_30",
    "home_sp_yrfi_allowed_rate_30",
    "park_run_factor",
    "temperature_2m",
    "wind_speed_10m",
]
st.dataframe(filtered[[c for c in signal_cols if c in filtered.columns]], use_container_width=True, hide_index=True)

if DEFAULT_BACKTEST_FILE.exists():
    st.subheader("Backtest Snapshot")
    bt = pd.read_csv(DEFAULT_BACKTEST_FILE)
    if not bt.empty:
        bt["game_date"] = pd.to_datetime(bt["game_date"])
        daily = bt.groupby(bt["game_date"].dt.date).agg(
            games=("game_pk", "count"),
            actual_nrfi=("target_nrfi", "sum"),
            top5_nrfi=("target_nrfi", lambda s: s[bt.loc[s.index, "rank"] <= 5].sum()),
        )
        st.line_chart(daily[["actual_nrfi", "top5_nrfi"]])

experiment_rows = []
if EXPERIMENTS_DIR.exists():
    for summary_path in sorted(EXPERIMENTS_DIR.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        experiment_rows.append(
            {
                "feature_set": summary.get("feature_set", summary_path.parent.name),
                "rows": summary.get("rows", 0),
                "nrfi_events": summary.get("nrfi_events", 0),
                "brier": summary.get("brier"),
                "log_loss": summary.get("log_loss"),
                "top5_nrfi_rate": summary.get("top5_nrfi_rate"),
                "top5_nrfi_hits": summary.get("top5_nrfi_hits"),
            }
        )
if experiment_rows:
    st.subheader("Feature Experiment Results")
    experiments = pd.DataFrame(experiment_rows).sort_values(["brier", "log_loss"], na_position="last")
    st.dataframe(experiments, use_container_width=True, hide_index=True)
