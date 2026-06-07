# API Design

## Status

The public SaaS / B2B API is **planned**, not currently commercialized.

The current Render app exposes internal dashboard endpoints only. These endpoints support the research dashboard and are not a commercial API product.

## Product Position

Future APIs should provide read-only sports intelligence:

- Games
- Predictions
- Market odds
- Market movement
- CLV reports
- Model status
- Data quality
- OOS evaluation
- Health checks

The API must not provide sportsbook execution, automated wagering, or user-fund movement.

## Planned API Endpoints

Planned future endpoints:

```text
GET /v1/games
GET /v1/predictions
GET /v1/predictions/{game_id}
GET /v1/markets/odds
GET /v1/markets/movement
GET /v1/clv/report
GET /v1/models/status
GET /v1/models/{model_version}/card
GET /v1/data-quality/games/{game_id}
GET /v1/backtests/summary
GET /v1/oos/report
GET /v1/health
