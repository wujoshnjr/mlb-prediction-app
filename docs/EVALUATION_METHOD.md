# Evaluation Method

## Purpose

The system evaluates prediction quality through auditable research evidence, not claims of guaranteed profit.

## Core Metrics

### Closing-Line Value

CLV compares entry odds against closing odds.

Positive CLV suggests price capture, but it does not guarantee profit.

### Brier Score

Brier score evaluates probability calibration for binary outcomes.

Lower is better.

### LogLoss

LogLoss penalizes confident wrong predictions.

Lower is better.

### Model vs Market

The model must be compared against market no-vig probability.

Important comparisons:

- model Brier vs market Brier
- model LogLoss vs market LogLoss
- model calibration vs market baseline
- model CLV by segment

## Rolling OOS

Rolling out-of-sample evaluation should track future predictions only.

Each prediction should be timestamped before game start and evaluated only after the finalized result exists.

Planned rolling OOS fields:

- game_id
- game_date
- snapshot_time
- model_version
- pipeline_version
- model_prob
- market_prob
- no_vig_market_prob
- closing_market_prob
- prediction_side
- actual_result
- brier_model
- brier_market
- logloss_model
- logloss_market
- clv
- edge
- data_quality_grade
- risk_status
- paper_status

## Calibration

Calibration checks whether predicted probabilities match observed frequencies.

Calibration should not be considered ready until enough clean settled samples exist.

## No-Leakage Rule

Pregame prediction snapshots must not contain postgame outcome fields for model training or evaluation.

Trusted outcomes must come from `data/finalized_games.csv`.

## Sample State

`data/sample_state.json` is the canonical source for sample counts.

Reports should avoid independently redefining sample count.

## Promotion Gate

Promotion requires:

- enough clean training samples
- enough rolling OOS predictions
- model Brier better than market Brier
- model LogLoss better than market LogLoss
- acceptable calibration
- CLV not materially negative
- no-leakage tests pass
- live-lock tests pass

## Interpretation

A model is not proven by one good run.

Evidence must accumulate over time across OOS windows, market regimes, data quality slices, and model versions.
