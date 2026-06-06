from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPORT_DIR = Path("report")
DATA_DIR = Path("data")

REGISTRY_PATH = DATA_DIR / "model_registry.json"
OUTPUT_PATH = REPORT_DIR / "model_registry_report.json"

TRAINING_STATUS_PATH = DATA_DIR / "training_status.json"
BASELINE_PATH = REPORT_DIR / "baseline_comparison_report.json"
CALIBRATION_PATH = REPORT_DIR / "calibration_report.json"
WALKFORWARD_PATH = REPORT_DIR / "walkforward_evaluation.json"
RESEARCH_QUALITY_PATH = REPORT_DIR / "research_quality_report.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    status = {"path": str(path), "exists": path.exists(), "error": ""}
    if not path.exists():
        status["error"] = "file_missing"
        return None, status

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status["error"] = str(exc)
        return None, status

    if not isinstance(data, dict):
        status["error"] = "json_not_object"
        return None, status

    return data, status


def _load_registry() -> List[Dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return []

    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict) and isinstance(data.get("models"), list):
        return [item for item in data["models"] if isinstance(item, dict)]

    return []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _make_model_id(sample_count: int, trained: bool, created_at: str) -> str:
    raw = f"mlb-model|{sample_count}|{trained}|{created_at[:19]}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"mlb-{created_at[:10].replace('-', '')}-{digest}"


def build_registry() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    training, training_status = _read_json(TRAINING_STATUS_PATH)
    baseline, baseline_status = _read_json(BASELINE_PATH)
    calibration, calibration_status = _read_json(CALIBRATION_PATH)
    walkforward, walkforward_status = _read_json(WALKFORWARD_PATH)
    research, research_status = _read_json(RESEARCH_QUALITY_PATH)

    input_files = {
        "training_status": training_status,
        "baseline_comparison": baseline_status,
        "calibration": calibration_status,
        "walkforward": walkforward_status,
        "research_quality": research_status,
    }

    errors: List[str] = []
    warnings: List[str] = []

    if training is None:
        warnings.append("training_status.json missing; registry record will be partial")

    created_at = _utc_now()
    sample_count = _safe_int(
        (training or {}).get("sample_count")
        or (training or {}).get("clean_model_sample_count")
    )
    trained = bool((training or {}).get("trained", False))

    metrics: Dict[str, Any] = {}
    if baseline:
        comparison = baseline.get("comparison") or {}
        metrics["baseline"] = comparison

    if walkforward:
        metrics["walkforward"] = {
            "model_brier": walkforward.get("model_brier"),
            "market_brier": walkforward.get("market_brier"),
            "model_logloss": walkforward.get("model_logloss"),
            "market_logloss": walkforward.get("market_logloss"),
            "avg_clv": walkforward.get("avg_clv"),
            "positive_clv_rate": walkforward.get("positive_clv_rate"),
        }

    research_grade = (research or {}).get("research_grade")
    promotion_blockers: List[str] = []

    if sample_count < 500:
        promotion_blockers.append("sample_count < 500")
    if not trained:
        promotion_blockers.append("model not trained")
    if research_grade in {None, "D", "F"}:
        promotion_blockers.append("research quality not sufficient")
    promotion_blockers.append("live betting disabled by governance")

    model_id = _make_model_id(sample_count, trained, created_at)

    record = {
        "model_id": model_id,
        "created_at": created_at,
        "model_type": (training or {}).get("model_type", "mlb_prediction_model"),
        "sample_count": sample_count,
        "trained": trained,
        "artifact_paths": [
            "data/calibrator.pkl",
            "data/market_residual_model.pkl",
            "data/training_status.json",
        ],
        "metrics": metrics,
        "calibration_ready": bool((calibration or {}).get("calibration_ready")),
        "walkforward_ready": bool((walkforward or {}).get("walkforward_ready")),
        "promotion_status": "blocked",
        "promotion_blockers": promotion_blockers,
    }

    registry = _load_registry()

    existing_ids = {str(item.get("model_id")) for item in registry}
    existing_created = {str(item.get("created_at")) for item in registry}

    added = False
    if model_id not in existing_ids and created_at not in existing_created:
        registry.append(record)
        added = True

    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=True), encoding="utf-8")

    report = {
        "generated_at": created_at,
        "status": "ok" if not errors else "failed",
        "input_files": input_files,
        "registry_path": str(REGISTRY_PATH),
        "registry_count": len(registry),
        "new_record_added": added,
        "model_id": model_id,
        "errors": errors,
        "warnings": warnings,
        "recommendations": [
            "Registry records are for audit only; promotion remains blocked until promotion_gate_report allows it."
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    report = build_registry()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
