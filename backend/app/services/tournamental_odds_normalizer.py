from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


WORLD_CUP_PREFIX = "wc2026"


def normalize_tournamental_snapshot(payload: Any) -> dict[str, Any]:
    snapshot = payload if isinstance(payload, dict) else {}
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
    quality_flags: list[str] = []
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
        "probability_sum": probability_sum,
        "max_probability": round(max_probability, 6),
        "quality_flags": quality_flags,
        "is_settled_like": "settled_like" in quality_flags,
        "is_usable_for_prediction": not quality_flags,
    }


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


def timestamp_to_utc(value: Any) -> str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric > 10_000_000_000:
        numeric = numeric / 1000
    return datetime.fromtimestamp(numeric, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
