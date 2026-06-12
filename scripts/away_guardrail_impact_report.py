from **future** import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PREDICTION_REPORT_PATH = Path("report/prediction.json")
AWAY_PICK_DIAGNOSTIC_PATH = Path("report/away_pick_diagnostic_report.json")
OUTPUT_PATH = Path("report/away_guardrail_impact_report.json")

REPORT_TYPE = "away_guardrail_impact_v1"
PIPELINE_VERSION = "baseline_v2_clean"

def _utc_now() -> str:
return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _json_safe(value: Any) -> Any:
if value is None:
return None

```
if isinstance(value, bool):
    return value

if isinstance(value, int) and not isinstance(value, bool):
    return value

if isinstance(value, float):
    if math.isnan(value) or math.isinf(value):
        return None
    return value

if isinstance(value, str):
    return value

if isinstance(value, dict):
    return {str(k): _json_safe(v) for k, v in value.items()}

if isinstance(value, (list, tuple, set)):
    return [_json_safe(item) for item in value]

try:
    return str(value)
except Exception:
    return "unserializable_object"
```

def safe_json_dump(data: dict[str, Any], path: Path) -> None:
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
json.dumps(
_json_safe(data),
indent=2,
ensure_ascii=False,
allow_nan=False,
),
encoding="utf-8",
)

def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
try:
if not path.is_file():
return None, "file_missing"

```
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    if not isinstance(data, dict):
        return None, "json_not_object"

    return data, None

except json.JSONDecodeError:
    return None, "json_decode_error"
except Exception as exc:
    return None, str(exc)
```

def _as_list(value: Any) -> list[Any]:
if value is None:
return []

```
if isinstance(value, list):
    return value

if isinstance(value, (tuple, set)):
    return list(value)

return [value]
```

def _reason_list(value: Any) -> list[str]:
reasons: list[str] = []

```
for item in _as_list(value):
    reason = str(item or "").strip()
    if reason:
        reasons.append(reason)

return reasons
```

def _predictions_from_report(report: dict[str, Any] | None) -> list[dict[str, Any]]:
if report is None:
return []

```
for key in ("predictions", "today_predictions", "games"):
    value = report.get(key)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]

return []
```

def _safe_float(value: Any) -> float | None:
try:
if value is None:
return None

```
    parsed = float(value)

    if math.isnan(parsed) or math.isinf(parsed):
        return None

    return parsed

except (TypeError, ValueError):
    return None
```

def _safe_bool(value: Any) -> bool:
if isinstance(value, bool):
return value

```
if isinstance(value, str):
    return value.strip().lower() in {"true", "1", "yes", "y"}

if isinstance(value, (int, float)) and not isinstance(value, bool):
    return value == 1

return False
```

def _status_text(value: Any) -> str:
return str(value or "").strip()

def _base_report(
prediction_report: dict[str, Any] | None,
prediction_error: str | None,
diagnostic_report: dict[str, Any] | None,
diagnostic_error: str | None,
) -> dict[str, Any]:
generated_at = _utc_now()

```
return {
    "generated_at": generated_at,
    "status": "ok",
    "report_type": REPORT_TYPE,
    "pipeline_version": PIPELINE_VERSION,
    "betting_mode": "paper_research",
    "live_betting_allowed": False,
    "automated_wagering_allowed": False,
    "production_model_replacement_allowed": False,
    "input_files": {
        "prediction": {
            "path": str(PREDICTION_REPORT_PATH),
            "required": True,
            "available": prediction_report is not None,
            "error": prediction_error,
        },
        "away_pick_diagnostic": {
            "path": str(AWAY_PICK_DIAGNOSTIC_PATH),
            "required": False,
            "available": diagnostic_report is not None,
            "error": diagnostic_error,
        },
    },
    "summary": {
        "prediction_count": 0,
        "away_candidate_count": 0,
        "away_guardrail_applied_count": 0,
        "away_guardrail_applied_rate": None,
        "retained_away_paper_signal_count": 0,
        "downgraded_away_tracking_only_count": 0,
        "home_candidate_count": 0,
    },
    "reason_counts": {},
    "guardrail_status_counts": {},
    "downgraded_examples": [],
    "diagnostic_snapshot": {},
    "interpretation": {
        "guardrail_note": (
            "Away guardrail impact is measured on current prediction "
            "candidates, not settled outcomes."
        ),
        "paper_only_note": (
            "This report never enables live betting, automated wagering, "
            "or production model replacement."
        ),
        "recommended_use": (
            "Use this report to monitor how many away candidates are "
            "downgraded to tracking-only before evaluating settled outcomes later."
        ),
    },
    "warnings": [],
    "errors": [],
    "recommendations": [],
}
```

def _diagnostic_snapshot(
diagnostic_report: dict[str, Any] | None,
) -> dict[str, Any]:
if not isinstance(diagnostic_report, dict):
return {}

```
official_accuracy = diagnostic_report.get("official_accuracy")
sample_summary = diagnostic_report.get("sample_summary")

away_picks: dict[str, Any] = {}
if isinstance(official_accuracy, dict) and isinstance(
    official_accuracy.get("away_picks"), dict
):
    away_picks = official_accuracy["away_picks"]

recommended_guardrails = diagnostic_report.get("recommended_guardrails")
if not isinstance(recommended_guardrails, list):
    recommended_guardrails = []

return {
    "away_pick_accuracy": away_picks.get("accuracy"),
    "away_pick_sample_count": away_picks.get("sample_count"),
    "away_pick_rate": (
        sample_summary.get("away_pick_rate")
        if isinstance(sample_summary, dict)
        else None
    ),
    "recommended_guardrails": recommended_guardrails,
}
```

def compute_report(
prediction_report: dict[str, Any] | None,
diagnostic_report: dict[str, Any] | None = None,
prediction_error: str | None = None,
diagnostic_error: str | None = None,
) -> dict[str, Any]:
report = _base_report(
prediction_report=prediction_report,
prediction_error=prediction_error,
diagnostic_report=diagnostic_report,
diagnostic_error=diagnostic_error,
)

```
errors = report["errors"]
recommendations = report["recommendations"]

if prediction_report is None:
    report["status"] = "partial"
    errors.append("Missing or invalid report/prediction.json")
    recommendations.append(
        "Run prediction.py before away_guardrail_impact_report.py."
    )
    return report

predictions = _predictions_from_report(prediction_report)

away_candidates: list[dict[str, Any]] = []
home_candidates: list[dict[str, Any]] = []
guardrail_applied: list[dict[str, Any]] = []
retained_paper: list[dict[str, Any]] = []
downgraded_tracking: list[dict[str, Any]] = []

reason_counter: Counter[str] = Counter()
status_counter: Counter[str] = Counter()

for prediction in predictions:
    side = _status_text(prediction.get("moneyline_selected_side")).lower()
    recommendation_status = _status_text(
        prediction.get("recommendation_status")
    ).upper()

    if side == "away":
        away_candidates.append(prediction)

        applied = _safe_bool(prediction.get("away_guardrail_applied"))
        guardrail_status = _status_text(
            prediction.get("away_guardrail_status")
        ).lower()
        status_counter[guardrail_status if guardrail_status else "missing"] += 1

        for reason in _reason_list(prediction.get("away_guardrail_reasons")):
            reason_counter[reason] += 1

        if applied:
            guardrail_applied.append(prediction)

        if recommendation_status == "PAPER_BET" and not applied:
            retained_paper.append(prediction)

        if applied and recommendation_status == "TRACKING_ONLY":
            downgraded_tracking.append(prediction)

    elif side == "home":
        home_candidates.append(prediction)

away_candidate_count = len(away_candidates)
guardrail_applied_count = len(guardrail_applied)

report["summary"].update(
    {
        "prediction_count": len(predictions),
        "away_candidate_count": away_candidate_count,
        "away_guardrail_applied_count": guardrail_applied_count,
        "away_guardrail_applied_rate": (
            guardrail_applied_count / away_candidate_count
            if away_candidate_count
            else None
        ),
        "retained_away_paper_signal_count": len(retained_paper),
        "downgraded_away_tracking_only_count": len(downgraded_tracking),
        "home_candidate_count": len(home_candidates),
    }
)

report["reason_counts"] = dict(reason_counter)
report["guardrail_status_counts"] = dict(status_counter)

downgraded_examples: list[dict[str, Any]] = []
for prediction in downgraded_tracking[:10]:
    downgraded_examples.append(
        {
            "game_id": prediction.get("game_id"),
            "game_date": prediction.get("game_date"),
            "away_team": prediction.get("away_team"),
            "home_team": prediction.get("home_team"),
            "moneyline_selected_edge": _safe_float(
                prediction.get("moneyline_selected_edge")
            ),
            "away_selected_probability": _safe_float(
                prediction.get("away_selected_probability")
            ),
            "market_no_vig_home_prob": _safe_float(
                prediction.get("market_no_vig_home_prob")
            ),
            "away_guardrail_reasons": _reason_list(
                prediction.get("away_guardrail_reasons")
            ),
            "recommendation_status": prediction.get("recommendation_status"),
        }
    )

report["downgraded_examples"] = downgraded_examples
report["diagnostic_snapshot"] = _diagnostic_snapshot(diagnostic_report)

if guardrail_applied_count > 0:
    recommendations.append(
        "Track downgraded away candidates separately to verify whether the "
        "guardrail improves settled away accuracy."
    )

if away_candidate_count == 0:
    recommendations.append(
        "No away candidates in current prediction report; continue monitoring."
    )

return report
```

def generate_report() -> dict[str, Any]:
prediction_report, prediction_error = load_json(PREDICTION_REPORT_PATH)
diagnostic_report, diagnostic_error = load_json(AWAY_PICK_DIAGNOSTIC_PATH)

```
report = compute_report(
    prediction_report=prediction_report,
    diagnostic_report=diagnostic_report,
    prediction_error=prediction_error,
    diagnostic_error=diagnostic_error,
)

safe_json_dump(report, OUTPUT_PATH)

return report
```

def main() -> int:
report = generate_report()
print(
json.dumps(
_json_safe(report),
indent=2,
ensure_ascii=False,
allow_nan=False,
)
)
return 0

if **name** == "**main**":
raise SystemExit(main())
