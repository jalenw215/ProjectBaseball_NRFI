# MLB NRFI/YRFI Predictor

Free-data-first MLB NRFI/YRFI workflow modeled after the Home Run Predictor project.

It can:

- fetch historical Statcast data from Baseball Savant;
- build leakage-safe game-level first-inning training rows;
- train a calibrated NRFI probability model;
- backtest daily game rankings;
- generate today's NRFI and YRFI probabilities;
- show the workflow in a Streamlit dashboard;
- optionally merge manual NRFI/YRFI odds and post a Discord report.

The probabilities are research estimates, not guaranteed picks.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,discord]"
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

## Streamlit Community Cloud

Deploy settings:

- Repository: this GitHub repo
- Branch: `main`
- Main file path: `dashboard/app.py`

The app starts without generated data. In the dashboard, click `Full Refresh` to fetch regular-season first-inning Statcast data, build training rows, train the model, backtest, and generate the latest predictions. Generated CSV/model/log files are intentionally ignored by Git.

## No-Terminal Workflow

Use the dashboard buttons:

- `Fetch Historical Data`
- `Build Training Set`
- `Train Model`
- `Run Backtest`
- `Predict Today`
- `Full Refresh`

`Full Refresh` fetches the current two-season Statcast window, builds training rows, trains the model, runs a backtest, and generates today's rankings.

To be gentle with public data sources, the historical Statcast fetch has a 24-hour cooldown after a successful pull. During that cooldown, `Full Refresh` skips the API-heavy historical fetch and continues with the local data already on disk.

## CLI Workflow

```bash
nrfi-predictor fetch-statcast --start-date 2025-04-01 --end-date 2025-04-14
nrfi-predictor build-training --statcast data/raw/statcast_2025-04-01_2025-04-14.csv
nrfi-predictor train
nrfi-predictor backtest
nrfi-predictor predict-today
nrfi-predictor report
```

## Manual Odds

Add optional market odds to:

```text
data/raw/manual_nrfi_odds.csv
```

Expected columns:

```csv
date,game_pk,market,american_odds,book
2026-06-08,777001,NRFI,-120,DraftKings
2026-06-08,777001,YRFI,+100,DraftKings
```

Supported markets are `NRFI` and `YRFI`.

## Discord

Create `.env`:

```text
DISCORD_BOT_TOKEN=your_token_here
DISCORD_CHANNEL_ID=123456789012345678
```

Then run:

```bash
python scripts/post_discord.py --predictions data/predictions/latest_predictions.csv
```
