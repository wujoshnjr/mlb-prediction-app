from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPORT_DIR = Path("report")

PREDICTION_PATH = REPORT_DIR / "prediction.json"
OUTPUT_JSON = REPORT_DIR / "decision_audit_report.json"
OUTPUT_CSV = REPORT_DIR / "decision_audit.csv"

CSV_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "model_prob",
    "market_prob",
    "edge",
    "recommendation",
    "recommendation_status",
    "data_quality_grade",
    "prediction_allowed",
    "bet_allowed",
    "live_betting_allowed",
    "live_bet_candidate",
    "stake_multiplier",
    "block_reasons",
    "missing_critical_sources",
    "missing_important_sources",
    "top_positive_factors",
    "top_negative_factors",
    "audit_status",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> tuple[Optional[Any], Dict[str, Any]]:
    status = {"path": str(path), "exists": path.exists(), "error": ""}
    if not path.exists():
        status["error"] = "file_missing"
        return None, status

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status["error"] = str(exc)
        return None, status

    return data, status


def _predictions(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        for key in ("predictions", "today_predictions", "games"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

    return []


def _nested_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def _first_existing(item: Dict[str, Any], keys: List[str]) -> Any:
    features = _nested_dict(item.get("features"))
    for key in keys:
        if key in item and item.get(key) is not None:
            return item.get(key)
        if key in features and features.get(key) is not None:
            return features.get(key)
    return None


def _audit_row(item: Dict[str, Any]) -> Dict[str, Any]:
    data_quality = _nested_dict(item.get("data_quality_status"))
    governance = _nested_dict(item.get("model_governance_status"))

    block_reasons = []
    for source in (
        item.get("block_reasons"),
        governance.get("block_reasons"),
        data_quality.get("block_reasons"),
    ):
        if isinstance(source, list):
            block_reasons.extend(str(reason) for reason in source)
        elif source:
            block_reasons.append(str(source))

    missing_critical = data_quality.get("missing_critical_sources") or item.get("missing_critical_sources") or []
    missing_important = data_quality.get("missing_important_sources") or item.get("missing_important_sources") or []

    data_quality_grade = (
        item.get("data_quality_grade")
        or data_quality.get("data_quality_grade")
        or data_quality.get("grade")
        or ""
    )

    return {
        "game_id": item.get("game_id"),
        "game_date": item.get("game_date"),
        "home_team": item.get("home_team"),
        "away_team": item.get("away_team"),
        "model_prob": _first_existing(
            item,
            ["model_prob", "predicted_home_win_pct", "premarket_model_home_prob", "home_win_probability"],
        ),
        "market_prob": _first_existing(
            item,
            ["market_no_vig_home_prob", "market_prob", "market_home_prob"],
        ),
        "edge": _first_existing(item, ["model_edge_home", "edge", "moneyline_selected_edge"]),
        "recommendation": item.get("recommendation") or item.get("moneyline_recommendation"),
        "recommendation_status": item.get("recommendation_status"),
        "data_quality_grade": data_quality_grade,
        "prediction_allowed": data_quality.get("prediction_allowed", item.get("prediction_allowed")),
        "bet_allowed": data_quality.get("bet_allowed", item.get("bet_allowed", False)),
        "live_betting_allowed": bool(item.get("live_betting_allowed", False)),
        "live_bet_candidate": bool(item.get("live_bet_candidate", False)),
        "stake_multiplier": item.get("stake_multiplier", 0.0),
        "block_reasons": _list_to_text(sorted(set(block_reasons))),
        "missing_critical_sources": _list_to_text(missing_critical),
        "missing_important_sources": _list_to_text(missing_important),
        "top_positive_factors": _list_to_text(item.get("top_positive_factors")),
        "top_negative_factors": _list_to_text(item.get("top_negative_factors")),
        "audit_status": "ok",
    }


def build_report() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data, prediction_status = _read_json(PREDICTION_PATH)
    predictions = _predictions(data)

    warnings: List[str] = []
    errors: List[str] = []

    rows = [_audit_row(item) for item in predictions]

    if data is None:
        warnings.append("prediction.json missing or unreadable")
    if not predictions:
        warnings.append("no predictions found")

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})

    status = "ok" if rows else "partial"

    report = {
        "generated_at": _utc_now(),
        "status": status,
        "input_files": {"prediction": prediction_status},
        "audit_count": len(rows),
        "rows": rows,
        "errors": errors,
        "warnings": warnings,
        "recommendations": [
            "Decision audit is for transparency only; it must not enable live betting."
        ],
    }

    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    report = build_report()
    print(json.dumps({"status": report["status"], "audit_count": report["audit_count"]}, indent=2))


if __name__ == "__main__":
    main()
