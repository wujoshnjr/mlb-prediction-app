from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from scripts.model_training_common import safe_float, write_json

PREDICTION_PATH = Path("report/prediction.json")
OUTPUT_PATH = Path("report/prediction_trust_report.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, "file_missing"

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, str(exc)

    if not isinstance(payload, dict):
        return {}, "json_not_object"

    return payload, None


def _extract_games(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("games", "predictions", "recommendations"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _status_text(game: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = game.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    return ""


def _prob(game: Dict[str, Any], keys: List[str]) -> float | None:
    for key in keys:
        value = safe_float(game.get(key))
        if value is not None:
            if value > 1.0 and value <= 100.0:
                value = value / 100.0
            if math.isfinite(value):
                return max(0.0, min(1.0, value))
    return None


def _grade_game(game: Dict[str, Any]) -> Dict[str, Any]:
    model_prob = _prob(
        game,
        [
            "model_probability",
            "displayed_home_win_pct",
            "home_win_probability",
            "home_win_prob",
            "probability",
        ],
    )
    market_prob = _prob(
        game,
        [
            "market_probability",
            "market_no_vig_home_prob",
            "no_vig_market_home_prob",
            "market_prob",
        ],
    )

    probability_edge = None
    if model_prob is not None and market_prob is not None:
        probability_edge = model_prob - market_prob

    lineup_status = _status_text(game, ["lineup_status", "lineup_confirmed", "lineup_context_status"])
    starter_status = _status_text(game, ["starter_status", "starter_confirmed", "probable_pitcher_status"])
    odds_status = _status_text(game, ["odds_quality_status", "odds_status", "market_status"])
    data_quality_grade = str(game.get("data_quality_grade") or game.get("quality_grade") or "").upper()

    blockers: List[str] = []
    warnings: List[str] = []

    if model_prob is None:
        blockers.append("model probability unavailable")

    if market_prob is None:
        warnings.append("market probability unavailable")

    if "missing" in odds_status or "unavailable" in odds_status or odds_status in {"bad", "suspicious"}:
        blockers.append("odds unavailable or suspicious")

    if not starter_status or "uncertain" in starter_status or "missing" in starter_status:
        blockers.append("starter uncertain or missing")

    if not lineup_status or "unconfirmed" in lineup_status or "missing" in lineup_status:
        warnings.append("lineup unconfirmed or missing")

    if data_quality_grade in {"D", "F"}:
        blockers.append(f"low data quality grade: {data_quality_grade}")
    elif data_quality_grade == "C":
        warnings.append("medium-low data quality grade")

    model_disagreement = None
    if probability_edge is not None:
        model_disagreement = abs(probability_edge)

    if model_disagreement is not None and model_disagreement > 0.12:
        warnings.append("high model-market disagreement")

    critical_missing_count = len(blockers)
    warning_count = len(warnings)

    if critical_missing_count >= 2:
        trust_grade = "D"
        uncertainty_level = "high"
    elif critical_missing_count == 1:
        trust_grade = "C"
        uncertainty_level = "medium_high"
    elif warning_count >= 2:
        trust_grade = "C"
        uncertainty_level = "medium"
    elif warning_count == 1:
        trust_grade = "B"
        uncertainty_level = "medium_low"
    else:
        trust_grade = "A"
        uncertainty_level = "low"

    if lineup_status and "unconfirmed" in lineup_status and trust_grade == "A":
        trust_grade = "B"

    explanation = "Trust grade is based on probability availability, market availability, odds quality, starter status, lineup status, data quality, and model-market disagreement."

    return {
        "game_id": game.get("game_id"),
        "home_team": game.get("home_team"),
        "away_team": game.get("away_team"),
        "model_probability": model_prob,
        "market_probability": market_prob,
        "probability_edge": probability_edge,
        "model_disagreement": model_disagreement,
        "data_quality_grade": data_quality_grade or "UNKNOWN",
        "lineup_status": lineup_status or "unknown",
        "starter_status": starter_status or "unknown",
        "bullpen_data_quality": game.get("bullpen_data_quality", "unknown"),
        "weather_data_quality": game.get("weather_data_quality", "unknown"),
        "odds_data_quality": odds_status or "unknown",
        "uncertainty_level": uncertainty_level,
        "trust_grade": trust_grade,
        "trust_blockers": blockers,
        "warnings": warnings,
        "explanation": explanation,
    }


def build_report(
    *,
    prediction_path: Path = PREDICTION_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Dict[str, Any]:
    payload, error = _read_json(prediction_path)

    if error:
        report = {
            "generated_at": _utc_now(),
            "status": "partial",
            "prediction_trust_ready": False,
            "prediction_count": 0,
            "trust_counts": {},
            "predictions": [],
            "blockers": [f"prediction report unavailable: {error}"],
            "warnings": [],
            "shadow_only": True,
            "live_betting_allowed": False,
            "automated_wagering_allowed": False,
        }
        write_json(output_path, report)
        return report

    games = _extract_games(payload)
    predictions = [_grade_game(game) for game in games]
    trust_counts: Dict[str, int] = {}

    for item in predictions:
        trust_counts[item["trust_grade"]] = trust_counts.get(item["trust_grade"], 0) + 1

    report = {
        "generated_at": _utc_now(),
        "status": "ok",
        "prediction_trust_ready": True,
        "prediction_count": len(predictions),
        "trust_counts": trust_counts,
        "predictions": predictions,
        "blockers": [],
        "warnings": [],
        "shadow_only": True,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
    }
    write_json(output_path, report)
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
