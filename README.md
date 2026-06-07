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
