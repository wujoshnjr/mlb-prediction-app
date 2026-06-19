from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


WORLD_CUP_PREFIX = "wc2026"
DRAW_KEY = "draw"

TEAM_KEY_ALIASES: dict[str, set[str]] = {
    "czechia": {"czech-republic"},
    "czech-republic": {"czechia"},
    "dr-congo": {"congo-dr", "d-r-congo", "democratic-republic-of-the-congo"},
    "congo-dr": {"dr-congo"},
    "ivory-coast": {"cote-d-ivoire", "cote-divoire"},
    "cote-d-ivoire": {"ivory-coast"},
    "usa": {"united-states", "united-states-of-america"},
    "united-states": {"usa", "united-states-of-america"},
    "south-korea": {"korea-republic", "republic-of-korea"},
    "saudi-arabia": {"ksa"},
    "cape-verde": {"cabo-verde"},
    "curacao": {"curaçao"},
}


def normalize_tournamental_snapshot(payload: Any) -> dict[str, Any]:
    wrapper = payload if isinstance(payload, dict) else {}
    snapshot = wrapper.get("data") if isinstance(wrapper.get("data"), dict) else wrapper
    probabilities = snapshot.get("probabilities", {}) if isinstance(snapshot.get("probabilities"), dict) else {}

    match_markets: list[dict[str, Any]] = []
    group_markets: list[dict[str, Any]] = []
    winner_markets: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []

    for market_key, outcomes in probabilities.items():
        if not isinstance(market_key, str) or not isinstance(outcomes, dict):
            anomalies.append({"market_key": str(market_key), "reason": "invalid_market_payload"})
            continue

        parts = market_key.split(":")
        if len(parts) != 3 or parts[0] != WORLD_CUP_PREFIX:
            anomalies.append({"market_key": market_key, "reason": "unsupported_market_key"})
            continue

        market_type = parts[1]
        entity = parts[2]
        cleaned_outcomes = clean_outcomes(outcomes)

        if market_type == "match":
            row = normalize_match_market(market_key, entity, cleaned_outcomes)
            match_markets.append(row)
            if row["quality_flags"]:
                anomalies.append({"market_key": market_key, "reason": ",".join(row["quality_flags"])})
        elif market_type == "group":
            group_markets.append({
                "team_code": entity,
                "market_key": market_key,
                "yes_probability": cleaned_outcomes.get("Yes"),
                "no_probability": cleaned_outcomes.get("No"),
                "is_extreme": is_extreme_binary(cleaned_outcomes),
            })
        elif market_type == "winner":
            winner_markets.append({
                "team_code": entity,
                "market_key": market_key,
                "yes_probability": cleaned_outcomes.get("Yes"),
                "no_probability": cleaned_outcomes.get("No"),
                "is_extreme": is_extreme_binary(cleaned_outcomes),
            })
        else:
            anomalies.append({"market_key": market_key, "reason": "unknown_market_type"})

    match_markets.sort(key=lambda item: int(item["match_no"]) if str(item["match_no"]).isdigit() else 9999)
    group_markets.sort(key=lambda item: str(item["team_code"]))
    winner_markets.sort(key=lambda item: str(item["team_code"]))

    return {
        "source_key": "tournamental_odds",
        "timestamp_ms": snapshot.get("ts"),
        "timestamp_utc": timestamp_to_utc(snapshot.get("ts")),
        "reported_market_count": snapshot.get("market_count"),
        "market_count": len(probabilities),
        "match_market_count": len(match_markets),
        "usable_match_market_count": sum(1 for item in match_markets if item["is_usable_for_prediction"]),
        "group_market_count": len(group_markets),
        "winner_market_count": len(winner_markets),
        "match_markets": match_markets,
        "group_markets": group_markets,
        "winner_markets": winner_markets,
        "anomalies": anomalies,
        "usage_note": "Match markets are the only Tournamental odds rows used for pre-match predictions. Group and winner markets are tournament-level signals only.",
    }


def normalize_match_market(market_key: str, match_no: str, outcomes: dict[str, float]) -> dict[str, Any]:
    probability_sum = round(sum(outcomes.values()), 6)
    max_probability = max(outcomes.values()) if outcomes else 0.0
    normalized = normalize_outcomes(outcomes)
    outcome_records = [
        {
            "name": name,
            "team_key": normalize_team_key(name),
            "probability": probability,
            "normalized_probability": normalized.get(name),
        }
        for name, probability in outcomes.items()
    ]
    quality_flags: list[str] = []
    non_draw_count = sum(1 for item in outcome_records if item["team_key"] != DRAW_KEY)
    if non_draw_count != 2:
        quality_flags.append("team_outcome_count_mismatch")
    if len(outcomes) < 2:
        quality_flags.append("too_few_outcomes")
    if probability_sum < 0.95:
        quality_flags.append("probability_sum_too_low")
    if probability_sum > 1.05:
        quality_flags.append("probability_sum_too_high")
    if max_probability >= 0.995:
        quality_flags.append("settled_like")

    return {
        "match_no": str(match_no),
        "market_key": market_key,
        "outcomes": outcomes,
        "normalized_outcomes": normalized,
        "outcome_records": outcome_records,
        "probability_sum": probability_sum,
        "max_probability": round(max_probability, 6),
        "quality_flags": quality_flags,
        "is_settled_like": "settled_like" in quality_flags,
        "is_usable_for_prediction": not quality_flags,
    }


def find_market_signal_for_fixture(fixture: Any, market_snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    if not market_snapshot:
        return None
    home_name = getattr(getattr(fixture, "home_team", None), "name", None)
    away_name = getattr(getattr(fixture, "away_team", None), "name", None)
    if not home_name or not away_name:
        return None

    home_keys = candidate_team_keys(str(home_name))
    away_keys = candidate_team_keys(str(away_name))
    for market in market_snapshot.get("match_markets", []):
        if not market.get("is_usable_for_prediction"):
            continue
        outcome_records = market.get("outcome_records", [])
        home_outcome = find_outcome_record(outcome_records, home_keys)
        away_outcome = find_outcome_record(outcome_records, away_keys)
        draw_outcome = find_outcome_record(outcome_records, {DRAW_KEY})
        if not home_outcome or not away_outcome:
            continue
        return {
            "source_key": "tournamental_odds",
            "market_key": market.get("market_key"),
            "match_no": market.get("match_no"),
            "home_team_probability": home_outcome.get("normalized_probability"),
            "draw_probability": draw_outcome.get("normalized_probability") if draw_outcome else None,
            "away_team_probability": away_outcome.get("normalized_probability"),
            "raw_outcomes": market.get("outcomes"),
            "normalized_outcomes": market.get("normalized_outcomes"),
            "probability_sum": market.get("probability_sum"),
        }
    return None


def find_outcome_record(outcome_records: list[dict[str, Any]], candidates: set[str]) -> dict[str, Any] | None:
    for item in outcome_records:
        team_key = item.get("team_key")
        if isinstance(team_key, str) and team_key in candidates:
            return item
    return None


def candidate_team_keys(name: str) -> set[str]:
    base = normalize_team_key(name)
    return {base, *TEAM_KEY_ALIASES.get(base, set())}


def clean_outcomes(outcomes: dict[str, Any]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for name, value in outcomes.items():
        if not isinstance(name, str):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric < 0:
            continue
        cleaned[name] = round(numeric, 6)
    return cleaned


def normalize_outcomes(outcomes: dict[str, float]) -> dict[str, float]:
    total = sum(outcomes.values())
    if total <= 0:
        return {}
    return {name: round(value / total, 6) for name, value in outcomes.items()}


def is_extreme_binary(outcomes: dict[str, float]) -> bool:
    if not outcomes:
        return False
    return max(outcomes.values()) >= 0.995


def normalize_team_key(value: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized or "unknown"


def timestamp_to_utc(value: Any) -> str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric > 10_000_000_000:
        numeric = numeric / 1000
    return datetime.fromtimestamp(numeric, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
