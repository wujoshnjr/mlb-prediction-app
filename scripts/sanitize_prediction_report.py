from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPORT_DIR = Path("report")

PREDICTION_PATH = REPORT_DIR / "prediction.json"
OUTPUT_REPORT = REPORT_DIR / "prediction_sanitization_report.json"

NAN_LIKE_STRINGS = {
    "nan",
    "+nan",
    "-nan",
    "inf",
    "+inf",
    "-inf",
    "infinity",
    "+infinity",
    "-infinity",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_non_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False

    if isinstance(value, (int, float)):
        try:
            return not math.isfinite(float(value))
        except Exception:
            return True

    return False


def _is_nan_like_string(value: str) -> bool:
    return value.strip().lower() in NAN_LIKE_STRINGS


def _clean_value(value: Any, path: str, changes: List[Dict[str, Any]]) -> Any:
    if _is_non_finite_number(value):
        changes.append(
            {
                "path": path,
                "old_value": str(value),
                "new_value": None,
                "reason": "non_finite_number",
            }
        )
        return None

    if isinstance(value, str):
        if _is_nan_like_string(value):
            changes.append(
                {
                    "path": path,
                    "old_value": value,
                    "new_value": None,
                    "reason": "literal_nan_like_string",
                }
            )
            return None

        return value

    if isinstance(value, list):
        cleaned_list = []

        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"

            if isinstance(item, str) and _is_nan_like_string(item):
                changes.append(
                    {
                        "path": item_path,
                        "old_value": item,
                        "new_value": None,
                        "reason": "dropped_literal_nan_like_string_from_list",
                    }
                )
                continue

            cleaned_list.append(_clean_value(item, item_path, changes))

        return cleaned_list

    if isinstance(value, dict):
        cleaned_dict = {}

        for key, item in value.items():
            key_text = str(key)
            cleaned_dict[key_text] = _clean_value(
                item,
                f"{path}.{key_text}",
                changes,
            )

        return cleaned_dict

    return value


def _load_prediction() -> Tuple[Any, List[str]]:
    if not PREDICTION_PATH.exists():
        return None, ["report/prediction.json not found"]

    try:
        data = json.loads(PREDICTION_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"failed to read prediction.json: {exc}"]

    return data, []


def sanitize_prediction_report() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data, errors = _load_prediction()
    changes: List[Dict[str, Any]] = []

    if errors:
        report = {
            "generated_at": _utc_now(),
            "status": "failed",
            "input_files": {
                "prediction": {
                    "path": str(PREDICTION_PATH),
                    "exists": PREDICTION_PATH.exists(),
                }
            },
            "changed": False,
            "change_count": 0,
            "changes": [],
            "errors": errors,
            "warnings": [],
            "recommendations": [
                "prediction.json must exist before prediction sanitization can run."
            ],
        }

        OUTPUT_REPORT.write_text(
            json.dumps(report, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return report

    cleaned = _clean_value(data, "prediction", changes)

    if changes:
        PREDICTION_PATH.write_text(
            json.dumps(cleaned, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    report = {
        "generated_at": _utc_now(),
        "status": "ok",
        "input_files": {
            "prediction": {
                "path": str(PREDICTION_PATH),
                "exists": PREDICTION_PATH.exists(),
            }
        },
        "changed": bool(changes),
        "change_count": len(changes),
        "changes": changes[:100],
        "errors": [],
        "warnings": [],
        "recommendations": [
            "Sanitizer removes literal nan-like strings and non-finite numbers from prediction.json before health gate validation."
        ],
    }

    OUTPUT_REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    return report


def main() -> None:
    report = sanitize_prediction_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
