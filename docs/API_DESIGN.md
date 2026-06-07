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
```

These are not commercial API endpoints.

The `/run` endpoint must remain protected by `ADMIN_API_TOKEN`.

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
```

## Standard Response Envelope

Future B2B API responses should include:

```json
{
  "request_id": "string",
  "generated_at": "ISO-8601 timestamp",
  "pipeline_version": "string",
  "model_version": "string",
  "data_snapshot_id": "string",
  "game_id": "string",
  "prediction": {},
  "market": {},
  "data_quality": {},
  "risk_flags": {},
  "audit": {}
}
```

## Prediction Response Schema

A future prediction response should include:

```json
{
  "game_id": "string",
  "game_date": "YYYY-MM-DD",
  "home_team": "string",
  "away_team": "string",
  "model_probability": 0.0,
  "market_probability": 0.0,
  "edge": 0.0,
  "recommendation_state": "TRACKING_ONLY",
  "paper_status": "PAPER_ENTRY_BLOCKED_BY_RISK",
  "live_status": "LIVE_LOCKED",
  "data_quality_grade": "B",
  "blocking_reasons": [],
  "audit": {
    "snapshot_time": "ISO-8601 timestamp",
    "model_version": "string",
    "feature_version": "string",
    "market_source": "string"
  }
}
```

## Authentication

Planned authentication:

- API key authentication
- Hashed API keys
- Organization-level API keys
- Per-plan access control
- Optional JWT/session auth for dashboard users

Current status:

- Not implemented for commercial use.
- Do not expose commercial endpoints until authentication and usage tracking are implemented.

## Rate Limiting

Planned controls:

- Per-organization quota
- Per-key rate limits
- Burst limits
- Usage metering
- Abuse detection

Current status:

- Planned, not implemented.

## Usage Tracking

Future usage events should record:

- organization_id
- api_key_id
- endpoint
- request_id
- timestamp
- response status
- latency
- quota bucket

## Data Licensing

API commercialization is not ready until:

- data-source licensing is reviewed
- redistribution rights are confirmed
- commercial usage scope is documented
- unlicensed third-party data is excluded or transformed into allowed derived metrics

## Prohibited API Types

The API must never include:

```text
POST /bets
POST /wagers
POST /orders
POST /sportsbook/execute
POST /wallet/deposit
POST /wallet/withdraw
POST /funds/transfer
```

The API must not include:

- sportsbook execution endpoints
- fund transfer endpoints
- wallet custody endpoints
- automated wagering endpoints

## OpenAPI

Future B2B API should expose an OpenAPI schema and developer documentation.

## Readiness Rule

The B2B API is not ready until:

- API key auth exists
- rate limiting exists
- usage tracking exists
- data licensing review is complete
- risk policy is published
- no automated wagering policy is enforced
- OOS evidence is sufficient
- sample state is stable
- promotion gate blocks unsafe states
- live betting remains locked
