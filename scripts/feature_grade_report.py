from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.feature_schema import (
    MODEL_FEATURES,
    TRACKING_ONLY_FEATURES,
    FEATURE_GRADE_RULES,
)


def _current_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _assign_grade(availability_rate: float, non_zero_rate: float) -> str:
    if (
        availability_rate >= FEATURE_GRADE_RULES["A"]["availability_rate_min"]
        and non_zero_rate >= FEATURE_GRADE_RULES["A"]["non_zero_rate_min"]
    ):
        return "A"

    if (
        availability_rate >= FEATURE_GRADE_RULES["B"]["availability_rate_min"]
        and non_zero_rate >= FEATURE_GRADE_RULES["B"]["non_zero_rate_min"]
    ):
        return "B"

    if (
        availability_rate >= FEATURE_GRADE_RULES["C"]["availability_rate_min"]
        and non_zero_rate >= FEATURE_GRADE_RULES["C"]["non_zero_rate_min"]
    ):
        return "C"

    return "D"


def build_feature_grade_report(
    availability_path: str = "report/feature_availability_diagnostic.json",
    output_path: str = "report/feature_grade_report.json",
) -> dict[str, Any]:
    availability_report = _load_json(Path(availability_path))
    feature_rows = availability_report.get("feature_availability") or []

    model_features = set(MODEL_FEATURES)
    tracking_features = set(TRACKING_ONLY_FEATURES)

    rows: list[dict[str, Any]] = []
    grade_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}

    for row in feature_rows:
        if not isinstance(row, dict):
            continue

        feature = str(row.get("feature", ""))
        missing_rate = _safe_float(row.get("missing_rate"), 1.0)
        non_zero_rate = _safe_float(row.get("non_zero_rate"), 0.0)
        availability_rate = max(0.0, 1.0 - missing_rate)

        grade = _assign_grade(availability_rate, non_zero_rate)
        grade_counts[grade] = grade_counts.get(grade, 0) + 1

        training_allowed_by_schema = feature in model_features
        tracking_only_by_schema = feature in tracking_features

        if grade in {"A", "B"} and training_allowed_by_schema:
            recommendation = "allowed_in_main_model"
        elif tracking_only_by_schema:
            recommendation = "tracking_only_by_schema"
        elif grade in {"C", "D"}:
            recommendation = "exclude_until_stable"
        else:
            recommendation = "review_schema"

        rows.append(
            {
                "feature": feature,
                "group": row.get("group", "unknown"),
                "grade": grade,
                "availability_rate": round(availability_rate, 4),
                "non_zero_rate": round(non_zero_rate, 4),
                "missing_rate": round(missing_rate, 4),
                "zero_rate": row.get("zero_rate"),
                "latest_importance": row.get("latest_importance", 0.0),
                "training_allowed_by_schema": training_allowed_by_schema,
                "tracking_only_by_schema": tracking_only_by_schema,
                "recommendation": recommendation,
            }
        )

    report = {
        "generated_at": _current_utc_iso(),
        "source": availability_path,
        "model_feature_count": len(MODEL_FEATURES),
        "tracking_only_feature_count": len(TRACKING_ONLY_FEATURES),
        "grade_counts": grade_counts,
        "model_features": MODEL_FEATURES,
        "tracking_only_features": TRACKING_ONLY_FEATURES,
        "features": sorted(rows, key=lambda item: (item["grade"], item["feature"])),
        "recommendations": [
            "A/B features may be considered for the main model only if schema allows them.",
            "C/D features should remain tracking-only until walk-forward and CLV contribution are proven.",
            "Do not promote a feature based on coefficient size alone.",
        ],
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    report = build_feature_grade_report()
    print(
        json.dumps(
            {
                "model_feature_count": report["model_feature_count"],
                "tracking_only_feature_count": report["tracking_only_feature_count"],
                "grade_counts": report["grade_counts"],
                "output_path": "report/feature_grade_report.json",
            },
            indent=2,
            ensure_ascii=True,
        )
    )
