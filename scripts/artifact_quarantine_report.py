from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_STATUS_PATH = Path("report/model_status_consistency_report.json")
OUTPUT_PATH = Path("report/artifact_quarantine_report.json")
MINIMUM_PRODUCTION_TRAINING_SAMPLES = 300


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(float(value))
    except Exception:
        return None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_report() -> dict[str, Any]:
    source = read_json(MODEL_STATUS_PATH)
    mismatches = source.get("mismatches") if isinstance(source.get("mismatches"), list) else []

    current_samples = to_int(source.get("training_status_sample_count"))
    artifact_samples = to_int(source.get("artifact_metadata_training_sample_count"))
    artifact_exists = bool(source.get("artifact_exists", False))
    artifact_loadable = bool(source.get("artifact_loadable", False))
    active_model_allowed = bool(source.get("active_model_allowed", False))
    trained = bool(source.get("trained", False))

    allowed_mismatch_fields = {
        "artifact_status_vs_training_status",
        "artifact_metadata_vs_training_status",
        "artifact_metadata_vs_training_samples",
        "model_artifact_status_sample_count_consistent",
    }
    mismatch_fields = {str(item.get("field")) for item in mismatches if isinstance(item, dict)}

    stale_sample_mismatch = (
        artifact_exists
        and artifact_loadable
        and current_samples is not None
        and artifact_samples is not None
        and artifact_samples > 0
        and current_samples > artifact_samples
        and mismatch_fields.issubset(allowed_mismatch_fields)
    )
    safely_locked = not active_model_allowed and not trained
    quarantined = bool(stale_sample_mismatch and safely_locked)

    status = "quarantined" if quarantined else "ok"
    if source and not quarantined and mismatches:
        status = "needs_review"
    if not source:
        status = "missing_source"

    report = {
        "generated_at": utc_now(),
        "report_type": "artifact_quarantine_report",
        "status": status,
        "source_path": str(MODEL_STATUS_PATH),
        "artifact_path": source.get("artifact_selected_path"),
        "artifact_exists": artifact_exists,
        "artifact_loadable": artifact_loadable,
        "artifact_training_sample_count": artifact_samples,
        "current_training_sample_count": current_samples,
        "minimum_production_training_samples": MINIMUM_PRODUCTION_TRAINING_SAMPLES,
        "current_samples_below_minimum": bool(
            current_samples is not None and 0 < current_samples < MINIMUM_PRODUCTION_TRAINING_SAMPLES
        ),
        "stale_sample_mismatch": stale_sample_mismatch,
        "safely_locked_out": safely_locked,
        "quarantined": quarantined,
        "mismatch_count": len(mismatches),
        "mismatch_fields": sorted(mismatch_fields),
        "promotion_allowed": False,
        "production_model_replacement_allowed": False,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "recommendations": [],
    }

    if quarantined:
        report["recommendations"].append(
            "Keep the stale artifact for auditability, but do not use it as the active model."
        )
        report["recommendations"].append(
            "Retrain only after sample-count and model-quality gates pass."
        )
    elif status == "needs_review":
        report["recommendations"].append(
            "Review model status consistency mismatches before trusting the artifact."
        )
    elif status == "missing_source":
        report["recommendations"].append(
            "Run scripts/model_status_consistency_report.py before this report."
        )
    else:
        report["recommendations"].append("No stale artifact quarantine condition detected.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    print(json.dumps(build_report(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
