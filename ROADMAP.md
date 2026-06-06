# Roadmap

## Phase 1: Governance and diagnostics

Status: mostly complete.

- Paper-only governance
- Training sample gate
- Data quality gate
- Feature governance
- Report health gate
- Zero-root-cause diagnostics
- Feature grade diagnostics

## Phase 2: Baseline, CLV, and calibration

Status: in progress.

- Baseline comparison report
- CLV slice reports
- Calibration report
- Static HTML report

## Phase 3: Walk-forward validation

Status: scaffolded.

- Walk-forward report structure
- Empty OOS predictions CSV
- Future true rolling folds
- Future promotion metrics

## Phase 4: Documentation and dashboard

Status: in progress.

- README
- Model card
- Data sources
- Evaluation methodology
- Risk disclosure
- Roadmap
- Changelog
- HTML dashboard

## Phase 5: Live consideration

Live betting should not be considered until all thresholds are met:

```text
clean settled samples >= 500
walk-forward predictions >= 300
model_brier < market_brier
model_logloss < market_logloss
avg_clv > 0
positive_clv_rate > 55%
large-edge bucket CLV not negative
lineup/starter/odds data quality >= B
