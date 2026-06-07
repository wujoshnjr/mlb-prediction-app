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

## Current Internal Dashboard Endpoints

The current FastAPI service may expose internal dashboard endpoints such as:

```text
GET /api/predictions
GET /api/performance
GET /api/health
POST /run
