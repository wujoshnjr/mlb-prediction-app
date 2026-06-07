# Risk Policy

## Position

This project is a paper-trading research dashboard and sports intelligence evidence pipeline.

It is not a real-money betting system.

## Permanent Safety Defaults

The following flags must remain false:

```text
live_betting_allowed = false
shadow_live_allowed = false
production_allowed = false
automated_wagering_allowed = false
user_funds_handled = false
```

## Prohibited Activities

The project must not:

- Place real-money bets.
- Connect to sportsbook execution APIs.
- Automatically wager.
- Handle user funds.
- Custody balances.
- Guarantee profit.
- Claim that predictions are certain.
- Commercially redistribute unlicensed data.

## Allowed Activities

The project may:

- Generate research predictions.
- Track paper entries.
- Track CLV.
- Compare model probabilities with market baselines.
- Run OOS evaluation.
- Build data quality reports.
- Build model governance reports.
- Display paper-only dashboard results.

## Promotion Requirements

A model cannot be promoted unless all of the following are satisfied:

- Clean training samples meet the threshold.
- Rolling OOS predictions meet the threshold.
- Model Brier beats market Brier.
- Model logloss beats market logloss.
- Calibration is acceptable.
- CLV is not materially negative.
- No-leakage tests pass.
- Data quality gates pass.
- Paper-only / live-lock tests pass.
- Promotion gate is explicit and auditable.

## Kill Switch

Any of the following must immediately block product readiness:

- `live_betting_allowed` is true.
- `automated_wagering_allowed` is true.
- Sportsbook execution code is introduced.
- User fund handling is introduced.
- Production live mode is enabled.

## Responsible Use

Users must understand that:

- Sports predictions can be wrong.
- Market prices can be efficient.
- Positive CLV does not guarantee future profit.
- Historical performance does not guarantee future performance.
- This project is for research and analytics only.

## Jurisdiction and Legal Review

Before any commercial release, the project requires:

- Data licensing review.
- Legal review.
- Terms of Service.
- Privacy Policy.
- Risk Disclosure.
- Responsible Use Policy.
- No Automated Wagering Policy.
