# Data Sources

## Overview

The project uses public and locally generated data sources to build MLB pregame prediction reports. Data availability is tracked explicitly so the model can distinguish true zero from unavailable information.

## Critical sources

Critical sources can block betting or prediction quality.

### MLB Stats API

Used for:

- Schedule
- Teams
- Probable pitchers
- Game status
- Scores and final results

### Market odds source

Used for:

- Moneyline odds
- Market no-vig probability
- Closing line comparison
- CLV analysis

If odds are missing or suspicious, betting must be blocked.

### Prediction validation / finalized games

Used for:

- Settled outcomes
- Brier score
- Logloss
- Calibration
- Walk-forward evaluation
- Baseline comparison

## Important sources

Important sources may allow predictions but should block betting if missing.

### Baseball Savant / Statcast

Used for:

- xwOBA proxy
- Hard-hit proxy
- Barrel proxy
- Top hitter context

If Savant is unavailable, MLB Stats proxy fallback may be used, and availability flags must reflect the source state.

### OpenWeatherMap

Used for:

- Temperature
- Wind speed
- Precipitation
- Dome/weather neutrality

Weather can be important but often produces neutral values.

### Pitcher advanced context

Used for:

- FIP
- K%
- BB%
- CSW proxy
- Stuff+ proxy

### Bullpen context

Used for:

- Bullpen fatigue
- Recent pitch load
- Availability difference

### Lineup context

Used for:

- Confirmed lineup
- Projected lineup
- Top-3 hitter availability

Unconfirmed lineup should block live betting.

## Optional sources

Optional sources are currently tracking-only unless validated.

Examples:

- Umpire
- Catcher
- Injury
- Sprint speed
- Pitch movement
- Pitch type matchup

Missing optional sources should not trigger production-blocking warnings.

## Local generated CSVs

Common generated files:

- `data/prediction_snapshots.csv`
- `data/finalized_games.csv`
- `data/market_odds_history.csv`
- `data/daily_game_context.csv`
- `data/weather_context.csv`
- `data/savant_top3_context.csv`
- `data/pitcher_advanced_context.csv`
- `data/team_form_context.csv`
- `data/context_feature_bridge.csv`

## Data freshness

All prediction records should include timestamps where possible. Pregame snapshots must be created before game start time.

## Missing data policy

Missing data should not be silently treated as true zero. Important features should include availability flags such as:

- `odds_available`
- `weather_available`
- `pitcher_advanced_available`
- `bullpen_context_available`
- `team_form_available`
- `lineup_context_available`
- `starter_context_available`
- `statcast_woba_available`
- `top3_woba_available`

## Source classes

```text
critical:
- schedule
- odds
- finalized game results
- prediction validation

important:
- lineup
- probable starters
- bullpen
- weather
- pitcher advanced
- Statcast / top hitter context

optional:
- catcher
- umpire
- injury
- sprint speed
- pitch movement
