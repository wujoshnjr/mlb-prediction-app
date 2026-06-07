# MLB Prediction App

## Current Position

MLB Prediction App is a **paper-trading research dashboard and market comparison evidence pipeline**.

It is designed to track MLB pregame predictions, market odds, closing-line value, data quality, model governance, sample state, and market-baseline comparisons.

This project is **not** a real-money betting execution system.

## What this project is

This repository currently operates as:

- MLB paper-trading research dashboard
- Market comparison evidence pipeline
- CLV tracking system
- Data quality and model-governance monitor
- Prediction snapshot and settlement research pipeline
- Sports intelligence infrastructure prototype

The long-term direction is to evolve into a **Sports Intelligence SaaS / B2B API platform**, but commercial API, billing, multi-tenant access, and enterprise modules are currently planned only.

## What this project is not

This project does not:

- Place real-money bets
- Connect to sportsbook execution APIs
- Handle user funds
- Guarantee profit
- Provide financial advice
- Provide gambling advice
- Automatically wager
- Sell commercial API access before data licensing and governance review

## Safety and governance

Live betting is permanently locked in the current system.

Expected safety fields:

```text
live_betting_allowed = false
shadow_live_allowed = false
production_allowed = false
automated_wagering_allowed = false
user_funds_handled = false
```

The project is research-only unless future governance, legal, data-licensing, and enterprise-readiness requirements are satisfied.

## Core question

The system is designed to answer:

> Can the model produce predictions that are timestamped, auditable, calibrated, settled from trusted outcomes, and eventually better than market baselines?

The answer must be proven through evidence, not assumed.

## Main pipeline concepts

### Prediction snapshots

Pregame predictions are saved with timestamps and market context.

### Finalized outcomes

Trusted outcomes must come from `data/finalized_games.csv`.

Prediction snapshots must not be treated as outcome sources.

### Sample state

`data/sample_state.json` is the canonical source for sample counts:

- raw snapshots
- valid snapshots
- settled snapshots
- clean settled snapshots
- train-eligible samples
- model artifact training samples
- walk-forward predictions

### Market comparison

The system compares model probabilities against market no-vig baselines.

### CLV

Closing-line value is tracked from entry odds versus closing odds. Positive CLV is evidence of price capture, not proof of profitability.

### Promotion gate

`report/promotion_gate_report.json` blocks promotion until sample size, OOS evidence, calibration, CLV, data quality, and live-lock governance requirements pass.

### World-class control tower

`report/world_class_trading_system_report.json` summarizes five layers:

1. Data trust
2. Research quality
3. Risk controls
4. Model upgrade path
5. Product readiness

### SaaS readiness

`report/saas_readiness_report.json` tracks readiness for a future SaaS / B2B API platform.

This does not mean the SaaS is live. It tracks what is implemented, planned, disabled, and blocked.

## Important reports

Common outputs:

```text
report/prediction.json
data/sample_state.json
report/sample_state_report.json
report/settled_prediction_link_report.json
report/baseline_comparison_report.json
report/calibration_report.json
report/rolling_walkforward_evaluation.json
report/market_close_report.json
report/evaluation_clv_diagnostic.json
report/promotion_gate_report.json
report/world_class_trading_system_report.json
report/saas_readiness_report.json
report/data_contract_report.json
report/pipeline_manifest.json
report/index.html
```

## Current model status

The model must not be considered production-ready until:

- clean training samples are sufficient
- rolling OOS predictions are sufficient
- model Brier beats market Brier
- model logloss beats market logloss
- calibration is acceptable
- CLV is not negative
- no-leakage checks pass
- data quality gates pass
- paper-only / live-lock tests pass

Current thresholds may include:

```text
minimum clean training samples: 300
minimum promotion samples: 500
minimum rolling OOS predictions: 300
```

## Render deployment

Render serves the FastAPI dashboard from `main.py`.

Required environment variable:

```text
ADMIN_API_TOKEN
```

Optional data-source secrets may include:

```text
ODDS_API_KEY
OPENWEATHER_API_KEY
APISPORTS_BASEBALL_KEY
APISPORTS_BASEBALL_LEAGUE_ID
BALLDONTLIE_API_KEY
```

Never commit secrets to the repository.

## GitHub Actions workflow

The scheduled workflow:

1. Checks out the repo
2. Installs dependencies
3. Compiles critical Python files
4. Updates finalized game results
5. Builds sample state
6. Trains only if enough finalized samples exist
7. Collects context data
8. Generates prediction report
9. Sanitizes prediction outputs
10. Builds evaluation reports
11. Builds promotion / governance reports
12. Builds world-class and SaaS readiness reports
13. Builds dashboard HTML
14. Runs data contract validation
15. Runs tests
16. Validates health gates
17. Uploads artifacts
18. Commits selected generated outputs

## Documentation

Current and planned documentation:

```text
docs/SAAS_ROADMAP.md
docs/API_DESIGN.md
docs/RISK_POLICY.md
docs/DATA_SOURCES.md
docs/EVALUATION_METHOD.md
docs/DEPLOYMENT_RENDER.md
docs/GITHUB_ACTIONS_WORKFLOW.md
docs/NO_AUTOMATED_WAGERING_POLICY.md
docs/B2B_PRODUCT_SPEC.md
```

## Local usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run key scripts:

```bash
python scripts/update_results.py
python scripts/sample_state_builder.py
python prediction.py
python scripts/sanitize_prediction_report.py
python scripts/baseline_comparison_report.py
python scripts/calibration_report.py
python scripts/market_close_report.py
python scripts/promotion_gate.py
python scripts/world_class_trading_system_report.py
python scripts/saas_readiness_report.py
python scripts/html_report_builder.py
python scripts/data_contract_validator.py
pytest -q
```

Open static report:

```bash
open report/index.html
```

## Risk disclosure

This project is for research, analytics, and evidence tracking only.

It does not provide financial advice, gambling advice, betting instructions, guaranteed profit, or automated wagering. Sports predictions can be wrong, markets can be efficient, and betting involves risk.

No real-money execution is supported.
