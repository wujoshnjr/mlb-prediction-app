from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://v1.baseball.api-sports.io"
ENV_API_KEY = "APISPORTS_BASEBALL_KEY"
ENV_LEAGUE_ID = "APISPORTS_BASEBALL_LEAGUE_ID"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_text(value: Any) -> str:
    """Return a clean string. Null-like values become empty string."""
    if value is None:
        return ""

    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""

    return text


def _api_get(
    path: str,
    params: Dict[str, Any],
    api_key: str,
    timeout: int = 20,
    max_sample: int = 3,
) -> Dict[str, Any]:
    """Safely call API-Sports and return a diagnostic result.

    This function never raises. It does not print or expose the API key.
    """
    url = f"{BASE_URL}{path}"

    result: Dict[str, Any] = {
        "path": path,
        "params": params,
        "status_code": None,
        "ok": False,
        "response_count": 0,
        "errors": [],
        "sample": [],
        "raw_error": "",
    }

    headers = {
        "x-apisports-key": api_key,
        "Accept": "application/json",
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=timeout,
        )
        result["status_code"] = response.status_code
    except requests.exceptions.RequestException as exc:
        result["raw_error"] = f"Request failed: {exc}"
        return result

    try:
        payload = response.json()
    except ValueError as exc:
        result["raw_error"] = (
            f"JSON decode error: {exc}; body={response.text[:300]}"
        )
        return result

    if not isinstance(payload, dict):
        result["raw_error"] = "Payload is not a dict"
        return result

    result["ok"] = response.status_code == 200

    api_errors = payload.get("errors")
    if api_errors:
        if isinstance(api_errors, list):
            result["errors"] = api_errors
        elif isinstance(api_errors, dict):
            result["errors"] = [str(api_errors)]
        else:
            result["errors"] = [str(api_errors)]

    response_items = payload.get("response", [])
    if isinstance(response_items, list):
        result["response_count"] = len(response_items)
        result["sample"] = response_items[:max_sample]
    else:
        result["raw_error"] = "Payload response is not a list"

    return result


# ---------------------------------------------------------------------------
# Public diagnostic
# ---------------------------------------------------------------------------

def run_diagnostic(date_str: Optional[str] = None) -> Dict[str, Any]:
    """Run API-Sports Baseball diagnostics for games/leagues endpoints.

    Args:
        date_str: Date in YYYY-MM-DD. Defaults to today's UTC date.

    Returns:
        A JSON-serializable diagnostic summary.
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        season = str(datetime.strptime(date_str, "%Y-%m-%d").year)
    except ValueError:
        season = str(datetime.now(timezone.utc).year)

    api_key = _clean_text(os.environ.get(ENV_API_KEY))
    configured_league_id = _clean_text(os.environ.get(ENV_LEAGUE_ID))

    summary: Dict[str, Any] = {
        "date": date_str,
        "season": season,
        "api_key_present": bool(api_key),
        "configured_league_id": configured_league_id,
        "tests": [],
    }

    if not api_key:
        summary["fatal_error"] = "APISPORTS_BASEBALL_KEY missing"
        return summary

    # -----------------------------------------------------------------------
    # 1. /games without league filter
    # -----------------------------------------------------------------------
    summary["tests"].append(
        _api_get(
            "/games",
            {
                "date": date_str,
                "season": season,
            },
            api_key,
            max_sample=3,
        )
    )

    # -----------------------------------------------------------------------
    # 2. /games with league filters
    #    Always test configured league id first, then 1-10.
    # -----------------------------------------------------------------------
    league_ids_to_test: List[str] = []

    if configured_league_id:
        league_ids_to_test.append(configured_league_id)

    for league_id in [str(i) for i in range(1, 11)]:
        if league_id not in league_ids_to_test:
            league_ids_to_test.append(league_id)

    for league_id in league_ids_to_test:
        summary["tests"].append(
            _api_get(
                "/games",
                {
                    "date": date_str,
                    "league": league_id,
                    "season": season,
                },
                api_key,
                max_sample=3,
            )
        )

    # -----------------------------------------------------------------------
    # 3. /leagues
    # -----------------------------------------------------------------------
    summary["tests"].append(
        _api_get(
            "/leagues",
            {
                "season": season,
            },
            api_key,
            max_sample=20,
        )
    )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    diagnostic = run_diagnostic()
    print(json.dumps(diagnostic, ensure_ascii=True, indent=2, default=str))
