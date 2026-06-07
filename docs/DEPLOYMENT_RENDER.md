# Render Deployment Guide

## Purpose

Render hosts the FastAPI dashboard defined in `main.py`.

The Render deployment is for research dashboard access only.

It must not expose real-money betting execution.

## Required Environment Variables

Required:

```text
ADMIN_API_TOKEN
```

Optional data-source secrets:

```text
ODDS_API_KEY
OPENWEATHER_API_KEY
APISPORTS_BASEBALL_KEY
APISPORTS_BASEBALL_LEAGUE_ID
BALLDONTLIE_API_KEY
```

Do not commit secrets to the repository.

## Render Setup

Recommended setup:

1. Create a Render Web Service.
2. Connect the GitHub repository.
3. Set the build command based on the project requirements.
4. Set the start command to run the FastAPI app.
5. Add required environment variables.
6. Configure health checks.
7. Confirm dashboard loads without exposing secrets.

## Dashboard

The dashboard displays:

- today’s predictions
- market evidence
- CLV
- data quality
- model readiness
- paper-only governance

## Manual Run Endpoint

If a manual run endpoint exists, it must require `ADMIN_API_TOKEN`.

Do not expose unauthenticated pipeline execution.

## Security Notes

- Never place secrets in code.
- Never print secrets in logs.
- Never enable live betting.
- Never connect sportsbook execution APIs.
- Keep research-only disclaimers visible.

## Health Checks

Recommended health endpoint:

```text
/api/health
```

The health check should verify that the web app responds, but it should not trigger prediction generation.
