from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
REPORT_DIR = Path("report")

SAMPLE_STATE_PATH = DATA_DIR / "sample_state.json"
UNDERDOG_REPORT_PATH = REPORT_DIR / "underdog_diagnostic_report.json"
CONFIDENCE_REPORT_PATH = REPORT_DIR / "confidence_bucket_guardrail_report.json"
MODEL_DECISION_GUARDRAIL_PATH = REPORT_DIR / "model_decision_guardrail_report.json"
CALIBRATION_DIAGNOSTICS_PATH = REPORT_DIR / "calibration_diagnostics_report.json"

REPORT_PATH = REPORT_DIR / "slice_promotion_gate_report.json"

MIN_RESEARCH_SAMPLE = 300

ALLOWED_STATES = {
    "NO_SIGNAL",
    "MODEL_SIGNAL_ONLY",
    "TRACKING_ONLY",
    "PAPER_ENTRY_ALLOWED",
    "PAPER_ENTRY_BLOCKED_BY_RISK",
    "LIVE_CANDIDATE_FORBIDDEN",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value) and not isinstance(value, (str, bool)):
            return None
    except Exception:
        pass
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {}, "file_missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, str(exc)
    if not isinstance(payload, dict):
        return {}, "json_not_object"
    return payload, ""


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return parsed
    except Exception:
        return None


def _to_int(value: Any) -> Optional[int]:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _state(value: str) -> str:
    if value not in ALLOWED_STATES:
        return "TRACKING_ONLY"
    return value


def _bucket_unsafe(report: dict[str, Any], bucket_name: str) -> bool:
    buckets = report.get("buckets")
    if not isinstance(buckets, dict):
        return False
    bucket = buckets.get(bucket_name)
    if not isinstance(bucket, dict):
        return False
    return bool(bucket.get("unsafe"))


def _bucket_decision(report: dict[str, Any], bucket_name: str) -> str:
    buckets = report.get("buckets")
    if not isinstance(buckets, dict):
        return "unknown"
    bucket = buckets.get(bucket_name)
    if not isinstance(bucket, dict):
        return "unknown"
    return str(bucket.get("decision") or "unknown")


def build_report() -> dict[str, Any]:
    warnings: list[str] = []
    blockers: list[str] = []
    errors: list[str] = []

    sample_state, sample_error = _read_json(SAMPLE_STATE_PATH)
    underdog, underdog_error = _read_json(UNDERDOG_REPORT_PATH)
    confidence, confidence_error = _read_json(CONFIDENCE_REPORT_PATH)
    model_guardrail, model_guardrail_error = _read_json(MODEL_DECISION_GUARDRAIL_PATH)
    calibration, calibration_error = _read_json(CALIBRATION_DIAGNOSTICS_PATH)

    for label, error in [
        ("sample_state", sample_error),
        ("underdog_diagnostic_report", underdog_error),
        ("confidence_bucket_guardrail_report", confidence_error),
        ("model_decision_guardrail_report", model_guardrail_error),
        ("calibration_diagnostics_report", calibration_error),
    ]:
        if error:
            warnings.append(f"{label} unavailable: {error}")

    sample_count = (
        _to_int(sample_state.get("clean_settled_snapshots"))
        or _to_int(sample_state.get("train_eligible_samples"))
        or 0
    )

    global_decision = "NO_PROMOTION_SHADOW_ONLY"

    if sample_count < MIN_RESEARCH_SAMPLE:
        blockers.append(f"insufficient_samples_for_slice_promotion: {sample_count} < {MIN_RESEARCH_SAMPLE}")

    guardrail_decision = str(model_guardrail.get("decision") or "")
    if guardrail_decision and guardrail_decision != "NO_PROMOTION_SHADOW_ONLY":
        blockers.append(f"unexpected_model_guardrail_decision: {guardrail_decision}")

    if bool(model_guardrail.get("production_model_replacement_allowed")):
        blockers.append("model guardrail unexpectedly allows production replacement")

    confidence_policy = confidence.get("global_policy") if isinstance(confidence.get("global_policy"), dict) else {}
    high_confidence_blocked = bool(confidence_policy.get("block_high_confidence_language"))
    unsafe_confidence_buckets = confidence.get("unsafe_buckets", [])
    if not isinstance(unsafe_confidence_buckets, list):
        unsafe_confidence_buckets = []

    underdog_policy = "TRACKING_ONLY"
    underdog_recommendation = underdog.get("recommendation")
    if isinstance(underdog_recommendation, dict):
        underdog_policy = str(underdog_recommendation.get("underdog_policy") or "TRACKING_ONLY")

    underdog_overall = underdog.get("overall") if isinstance(underdog.get("overall"), dict) else {}
    underdog_accuracy = _to_float(underdog_overall.get("accuracy"))
    underdog_ece = _to_float(underdog_overall.get("ece"))
    underdog_sample = _to_int(underdog_overall.get("sample_count")) or 0

    underdog_unsafe = False
    if underdog_sample < 30:
        warnings.append("underdog sample_count below 30; underdog paper entry remains blocked or tracking only")
        underdog_unsafe = True
    elif underdog_accuracy is not None and underdog_accuracy < 0.50:
        underdog_unsafe = True
    elif underdog_ece is not None and underdog_ece > 0.15:
        underdog_unsafe = True
    elif underdog_policy != "PAPER_ENTRY_ALLOWED":
        underdog_unsafe = True

    sample_gate_open = sample_count >= MIN_RESEARCH_SAMPLE and not blockers

    policy = {
        "home_favorite": "TRACKING_ONLY",
        "away_favorite": "TRACKING_ONLY",
        "home_underdog": "TRACKING_ONLY",
        "away_underdog": "TRACKING_ONLY",
        "high_confidence": "TRACKING_ONLY",
        "lineup_missing": "TRACKING_ONLY",
        "lineup_ready": "MODEL_SIGNAL_ONLY",
        "default": "TRACKING_ONLY",
    }

    if sample_gate_open:
        policy["home_favorite"] = "PAPER_ENTRY_ALLOWED"
        policy["away_favorite"] = "MODEL_SIGNAL_ONLY"

    if underdog_unsafe:
        policy["home_underdog"] = "PAPER_ENTRY_BLOCKED_BY_RISK"
        policy["away_underdog"] = "PAPER_ENTRY_BLOCKED_BY_RISK"
    elif sample_gate_open:
        policy["home_underdog"] = "MODEL_SIGNAL_ONLY"
        policy["away_underdog"] = "MODEL_SIGNAL_ONLY"

    if high_confidence_blocked or unsafe_confidence_buckets:
        policy["high_confidence"] = "PAPER_ENTRY_BLOCKED_BY_RISK"
    elif sample_gate_open:
        policy["high_confidence"] = "MODEL_SIGNAL_ONLY"

    if not sample_gate_open:
        for key, value in list(policy.items()):
            if value == "PAPER_ENTRY_ALLOWED":
                policy[key] = "TRACKING_ONLY"

    policy = {key: _state(value) for key, value in policy.items()}

    if underdog_unsafe:
        blockers.append("underdog slice is not approved for paper entry upgrade")

    if high_confidence_blocked:
        blockers.append("high-confidence language is blocked by confidence guardrail")

    calibration_ready = bool(calibration.get("calibration_ready"))
    calibration_ece = _to_float(calibration.get("ece"))
    if calibration_ece is None:
        calibration_ece = _to_float(calibration.get("overall_ece"))

    if calibration_ece is not None and calibration_ece > 0.05:
        blockers.append(f"calibration_ece_above_threshold: {calibration_ece:.4f} > 0.05")
    elif calibration_error:
        warnings.append("calibration diagnostics missing; no slice can be promoted")

    report = {
        "generated_at": _utc_now(),
        "status": "ok" if not errors else "partial",
        "global_decision": global_decision,
        "sample_count": sample_count,
        "minimum_research_sample": MIN_RESEARCH_SAMPLE,
        "paper_entry_policy": policy,
        "evidence": {
            "underdog_accuracy": underdog_accuracy,
            "underdog_ece": underdog_ece,
            "underdog_sample_count": underdog_sample,
            "underdog_policy": underdog_policy,
            "high_confidence_blocked": high_confidence_blocked,
            "unsafe_confidence_buckets": unsafe_confidence_buckets,
            "calibration_ready": calibration_ready,
            "calibration_ece": calibration_ece,
        },
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "errors": errors,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }

    return report


def main() -> None:
    report = build_report()
    _write_json(REPORT_PATH, report)
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
