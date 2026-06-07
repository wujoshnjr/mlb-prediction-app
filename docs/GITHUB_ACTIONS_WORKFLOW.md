# GitHub Actions Workflow

## Purpose

The GitHub Actions workflow runs the research data pipeline, generates reports, validates contracts, runs tests, and commits selected outputs.

It must not place real-money bets or execute sportsbook actions.

## Workflow Overview

The scheduled workflow generally performs:

1. Checkout repository.
2. Install dependencies.
3. Compile critical Python files.
4. Update finalized game results.
5. Build sample state.
6. Train model only if enough finalized samples exist.
7. Collect data context.
8. Generate prediction report.
9. Sanitize prediction output.
10. Build market and evaluation reports.
11. Build promotion and governance reports.
12. Build world-class trading system report.
13. Build SaaS readiness report.
14. Build HTML report.
15. Run data contract validation.
16. Run pytest.
17. Validate report health gates.
18. Upload artifacts.
19. Commit selected generated outputs.

## Important Reports

Generated reports include:

- `prediction.json`
- `sample_state.json`
- `sample_state_report.json`
- `market_close_report.json`
- `evaluation_clv_diagnostic.json`
- `baseline_comparison_report.json`
- `calibration_report.json`
- `promotion_gate_report.json`
- `world_class_trading_system_report.json`
- `saas_readiness_report.json`
- `data_contract_report.json`
- `pipeline_manifest.json`

## Live Lock

The workflow must preserve:

```text
live_betting_allowed = false
automated_wagering_allowed = false
user_funds_handled = false
production_allowed = false
```

## Failure Policy

The workflow should fail for:

- syntax errors
- invalid JSON
- data contract failure
- report health gate failure
- no-leakage test failure
- live-lock governance violation

The workflow should not fail merely because future SaaS modules are not yet implemented.

Future SaaS modules should appear as planned or partial, not failed.
