# Data Sources

## Purpose

This document tracks data-source governance for the MLB paper-trading research dashboard and future Sports Intelligence SaaS / B2B platform.

## Current Data Categories

The system may use:

- MLB game schedule and results
- Team and player context
- Probable pitcher context
- Lineup context
- Weather context
- Market odds snapshots
- Opening and closing odds
- Prediction snapshots
- Finalized outcomes

## Data Source Registry Design

Future `data_sources_registry.json` should include:

```json
{
  "source_name": "string",
  "source_type": "schedule | odds | weather | lineup | player_stats | results",
  "license_status": "unknown | public | trial | paid | restricted",
  "redistribution_allowed": false,
  "refresh_interval": "string",
  "sla": "string",
  "fallback_source": "string",
  "last_success_at": "ISO-8601 timestamp",
  "last_error": "string"
}
```

## License Policy

Data must not be redistributed commercially unless licensing explicitly allows it.

Unknown or trial license sources must be treated as internal research-only sources.

## Redistribution Rule

The future B2B API must not expose third-party data commercially until:

- License terms are reviewed.
- Redistribution rights are confirmed.
- Data vendor contracts are documented.
- Commercial usage scope is approved.

## Data Lineage

Every prediction should be traceable to:

- source file
- ingestion time
- snapshot time
- pipeline version
- model version
- game_id
- data quality state

## Finalized Outcomes

`data/finalized_games.csv` must be the only trusted source for settled outcomes used in:

- training
- calibration
- model vs market evaluation
- ROI
- win rate
- promotion gate

Prediction snapshots must not be used as outcome sources.

## Future Data Governance Modules

Planned modules:

- `data_sources_registry.json`
- `data_license_registry.json`
- data lineage log
- data ingestion status
- data vendor contracts
- redistribution policy checks
