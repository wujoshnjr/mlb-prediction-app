# No Automated Wagering Policy

## Policy

This repository and all derivative products strictly prohibit automated wagering.

## Prohibited

The system must not:

- Place bets automatically.
- Send wager instructions to sportsbooks.
- Connect to sportsbook execution APIs.
- Move user funds.
- Custody user balances.
- Trigger real-money transactions.
- Present paper entries as executable bets.
- Guarantee profit.

## Required Defaults

The following must remain false:

```text
live_betting_allowed = false
automated_wagering_allowed = false
user_funds_handled = false
production_allowed = false
```

## Enforcement

The project should enforce this through:

- code review
- pytest
- data contract validation
- promotion gate
- world-class control tower
- SaaS readiness report
- no-leakage and live-lock tests

## Violation

If automated wagering or live betting is enabled, readiness status must become failed.

## Allowed

The system may support:

- research predictions
- paper-trading ledgers
- CLV tracking
- OOS evaluation
- model governance
- market intelligence dashboards

## Summary

This project is analytics infrastructure, not wagering infrastructure.
