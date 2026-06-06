# Risk Disclosure

## For research and educational use only

This project is an experimental analytics system. It is not a betting service.

## No guaranteed profit

There is no guarantee that the model will make accurate predictions or produce profitable results.

## Not financial advice

Nothing in this repository is financial advice.

## Not gambling advice

Nothing in this repository is gambling advice, betting advice, or a recommendation to place wagers.

## Sports betting is risky

Sports betting involves risk of loss. You should never wager money you cannot afford to lose.

## Model error risk

The model can be wrong because of:

- Noisy data
- Missing data
- Bad assumptions
- Small sample size
- Overfitting
- Late lineup changes
- Starting pitcher changes
- Market movement

## Data source risk

Data sources may be delayed, incomplete, stale, or incorrect. Even with data quality checks, undetected data issues can occur.

## Market efficiency risk

Sports betting markets are often efficient. A model can appear promising over a short period and still fail to beat the market long term.

## Responsible use

Use this project only for research, education, and paper-trading analysis. Do not present outputs as guaranteed picks or betting tips.

## Paper trading default

Paper trading is the default. Live betting is disabled by governance.
```

---

FILE: `ROADMAP.md`

````markdown
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
```

Even after these thresholds, independent review and extended paper-trading should be required.
