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
    "none",
    "null",
    "nat",
    "<na>",
    "n/a",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_bad_float(value: Any) -> bool:
    if not isinstance(value, float):
        return False
    return math.isnan(value) or math.isinf(value)


def _is_nan_like_string(value: str) -> bool:
    return value.strip().lower() in NAN_LIKE_STRINGS


def _clean_value(value: Any, path: str, changes: List[Dict[str, Any]]) -> Any.isnan(value) or math.isinf(value)


def _is_nan_like_string(value: str) -> bool:
    return value.strip().lower() in NAN_L:
    if _is_bad_float(value):
        changes.append(
            {
                "path": path,
                "old_value": str(value),
                "new_value": None,
                "reason": "non_finite_float",
            }
        )
        return None

    if isinstance(value, str):
        if _is_nan_like_string(value):
            changes.append(
                {
                    "path": path,
                    "old_value": value,
                    "new_value": "",
                    "reason": "literal_nan_like_string",
                }
            )
            return ""

        return value

    if isinstance(value, list):
        cleaned_list = []
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            cleaned_item = _clean_value(item, item_path, changes)

            # For text/detail lists, dropping empty nan-like values is cleaner than
            # keeping empty strings that add no information.
            if isinstance(item, str) and _is_nan_like_string(item):
                continue

            cleaned_list.append(cleaned_item)

        return cleaned_list

    if isinstance(value, dict):
        cleaned_dict = {}
        for key, item in value.items():
            key_text = str(key)
            cleaned_dict[key_text] = _clean_value(item, f"{path}.{key_text}", changes)
        return cleaned_dict

    return value


def _load_prediction() -> Tuple[Any, List[str]]:
    errors: List[str] = []

    if not PREDICTION_PATH.exists():
        return None, ["report/prediction.json not found"]

    try:
        data = json.loads(PREDICTION_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"failed to read prediction.json: {exc}"]

    return data, errors


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
            "Sanitizer removes literal nan-like strings and non-finite floats from prediction.json before health gate validation."
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
