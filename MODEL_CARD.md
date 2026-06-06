# Model Card

## Model name

MLB Prediction App main pregame model.

## Model type

Current production behavior is conservative:

- If clean samples are below the training threshold, the system does not load the ML artifact.
- Predictions fall back to manual / baseline logic.
- Main model training uses governed feature schema and an imputation pipeline when enough samples exist.

Expected training model type after sample threshold is met:

```text
calibrated_logistic_regression_with_imputer
```

## Current production readiness

Not ready.

The project is currently an experimental paper-trading research tool.

## Training sample threshold

```text
minimum_clean_train_samples = 300
```

If the clean settled sample count is below 300, training is skipped and the production ML artifact should not be used.

## Live betting threshold

Live betting is disabled by governance. It should not be considered until at least:

- Clean settled samples >= 500
- Walk-forward predictions >= 300
- Model Brier < market Brier
- Model logloss < market logloss
- Average CLV > 0
- Positive CLV rate > 55%
- Lineup, starter, odds, and core data quality >= B

## Production threshold

A production-level system should require at least:

```text
production_samples >= 1000
```

and stable performance over multiple rolling windows.

## Current behavior when samples < 300

When samples are below 300:

- ML training is skipped.
- The ML model artifact is not loaded for production predictions.
- `model_source` should remain manual or fallback.
- `live_betting_allowed` remains false.
- `stake_multiplier` remains 0.0.

## Features

Feature groups include:

- Market odds and no-vig probability context
- Team strength
- Pitcher advanced context
- Bullpen context
- Weather context
- Team form
- Feature availability flags

## Tracking-only features

Tracking-only features are logged but excluded from the main training pipeline until they prove availability, stability, walk-forward contribution, and CLV contribution.

Examples include:

- Catcher context
- Umpire context
- Injury context
- Pitch movement context
- Pitch type matchup context
- Sprint speed
- Some Statcast-derived research features
- Team form extras such as rest pressure

## Data limitations

Known limitations:

- Sample size is currently small.
- Market odds may be incomplete or unavailable for some games.
- Lineup confirmation often arrives late.
- Probable starters can change.
- Weather may be neutral or unavailable for dome games.
- Statcast data can lag or require proxy fallback.
- Historical sample coverage is still expanding.

## Calibration status

Calibration is not ready until at least 500 settled predictions are available. Calibration reports are generated for monitoring but should not be used for promotion yet.

## CLV status

CLV is a core metric. Negative average CLV is a red flag. Live betting must remain blocked unless CLV improves and remains positive across sufficient samples and buckets.

## Known limitations

- The system does not guarantee edge over the betting market.
- Short-term ROI can be misleading.
- Closing line value can remain negative even if occasional picks win.
- Feature importance is unstable at low sample sizes.
- Some features are currently tracking-only by design.

## Do-not-use-for-live-betting warning

Do not use this model for real-money betting. It is for research and educational use only. It is not financial advice or gambling advice.
