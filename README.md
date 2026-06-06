# MLB Prediction App

## What it does

MLB Prediction App is an experimental baseball prediction and paper-trading research system. It generates pregame MLB predictions, stores timestamped prediction snapshots, compares outputs against market baselines, and produces model-governance reports.

It is designed to answer one question:

> Can the system produce predictions that are trackable, auditable, calibrated, and eventually better than market baselines?

## Current status: experimental paper-trading only

The current system is not production-ready. Live betting is disabled by governance.

Current operating mode:

- Experimental research
- Paper-trading only
- No real-money betting
- No guaranteed profit
- Not financial advice
- Not gambling advice

## Live betting disabled by governance

The system includes a model governance gate. Live betting remains locked unless strict evidence thresholds are met.

Current minimum thresholds:

- Minimum clean training samples: 300
- Minimum live consideration samples: 500
- Production-level sample target: 1000
- Positive average CLV required
- Positive CLV rate must exceed 55%
- Model must beat market baseline on Brier and logloss
- Data quality must be A or B
- Lineup and starter confirmation must be available

Until those conditions are met, the system remains paper-trading only.

## Data sources

The project may use:

- MLB Stats API
- Baseball Savant / Statcast
- OpenWeatherMap
- Market odds history
- Local prediction snapshots
- Local finalized game results
- Generated context CSV files

See [`DATA_SOURCES.md`](DATA_SOURCES.md).

## Model pipeline

The pipeline has these stages:

1. Fetch schedules, odds, weather, pitcher context, team form, and Statcast proxy context.
2. Generate pregame predictions.
3. Save first-seen pregame snapshots.
4. Set model governance and data quality status.
5. Evaluate settled predictions.
6. Generate baseline, CLV, calibration, walk-forward, and feature diagnostics.
7. Build a static HTML dashboard.

## Feature governance

Features are governed through `scripts/feature_schema.py`.

The schema separates:

- `MODEL_FEATURES`: eligible for the main training pipeline
- `TRACKING_ONLY_FEATURES`: tracked for research, excluded from the main model
- `EXPECTED_FEATURES`: full runtime and snapshot feature contract

Missing source information is represented with availability flags rather than silently treating all missing values as true zero.

## No-leakage snapshot policy

Prediction snapshots must be created before the game start time. Snapshot rows include `snapshot_created_at` and `start_time`. Automated tests verify that snapshots are not created after the scheduled start.

## Evaluation metrics

The system does not rely on accuracy alone. Reports include:

- Brier score
- Logloss
- Accuracy
- Market no-vig baseline
- Constant 50% baseline
- Home historical rate baseline
- CLV by bucket
- Positive CLV rate
- Calibration bins
- Walk-forward scaffold

See [`EVALUATION.md`](EVALUATION.md).

## How to run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run core scripts:

```bash
python prediction.py
python scripts/feature_availability_diagnostic.py
python scripts/feature_zero_root_cause_diagnostic.py
python scripts/feature_grade_report.py
python scripts/baseline_comparison_report.py
python scripts/clv_slice_report.py
python scripts/calibration_report.py
python scripts/walkforward_evaluation.py
python scripts/html_report_builder.py
python scripts/report_health_gate.py
pytest -q
```

Open the dashboard:

```bash
open report/index.html
```

## GitHub Actions pipeline

The GitHub Actions workflow runs the data pipeline, diagnostics, report builders, tests, and health gates. Generated reports are uploaded as artifacts and selected outputs may be committed back to the repository.

## Reports generated

Common report outputs:

- `report/prediction.json`
- `report/evaluation_clv_diagnostic.json`
- `report/feature_availability_diagnostic.json`
- `report/feature_zero_root_cause_diagnostic.json`
- `report/feature_grade_report.json`
- `report/baseline_comparison_report.json`
- `report/clv_by_edge_bucket.json`
- `report/clv_by_side.json`
- `report/clv_by_odds_range.json`
- `report/clv_by_lineup_status.json`
- `report/calibration_report.json`
- `report/walkforward_evaluation.json`
- `report/index.html`

## Risk disclaimer

This project is for research and educational use only. It is not financial advice, gambling advice, or a betting recommendation service. Sports betting is risky, and model predictions can be wrong. No profit is guaranteed.
