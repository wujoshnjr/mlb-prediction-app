from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.model_training_common import safe_float, write_json

MODEL_LAB_PATH = Path("report/model_lab_report.json")
WALK_FORWARD_PATH = Path("report/walk_forward_validation_report.json")
CALIBRATION_PATH = Path("report/calibration_diagnostics_report.json")
FEATURE_PROMOTION_PATH = Path("report/feature_promotion_report.json")
OUTPUT_PATH = Path("report/model_comparison_report.json")


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


def _ranking(models: List[Dict[str, Any]], metric: str, lower_is_better: bool = True) -> List[Dict[str, Any]]:
    rows = []
    for model in models:
        value = safe_float(model.get(metric))
        if value is None:
            continue
        rows.append({"model_name": model.get("model_name"), metric: value})
    rows.sort(key=lambda item: item[metric], reverse=not lower_is_better)
    return rows


def _stability_score(model: Dict[str, Any]) -> float:
    score = 100.0

    if model.get("skipped"):
        score -= 60.0

    if model.get("promotion_blockers"):
        score -= min(30.0, len(model.get("promotion_blockers") or []) * 8.0)

    if safe_float(model.get("ece")) is not None and safe_float(model.get("ece")) > 0.05:
        score -= 15.0

    if model.get("warnings"):
        score -= min(20.0, len(model.get("warnings") or []) * 4.0)

    return round(max(0.0, min(100.0, score)), 2)


def build_report(
    *,
    model_lab_path: Path = MODEL_LAB_PATH,
    walk_forward_path: Path = WALK_FORWARD_PATH,
    calibration_path: Path = CALIBRATION_PATH,
    feature_promotion_path: Path = FEATURE_PROMOTION_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Dict[str, Any]:
    model_lab, model_lab_error = _read_json(model_lab_path)
    walk_forward, walk_forward_error = _read_json(walk_forward_path)
    calibration, calibration_error = _read_json(calibration_path)
    feature_promotion, feature_promotion_error = _read_json(feature_promotion_path)

    warnings = []
    blockers = []

    for name, error in (
        ("model_lab", model_lab_error),
        ("walk_forward", walk_forward_error),
        ("calibration", calibration_error),
        ("feature_promotion", feature_promotion_error),
    ):
        if error:
            warnings.append(f"{name} unavailable: {error}")

    models = model_lab.get("models") if isinstance(model_lab.get("models"), list) else []

    enriched = []
    for model in models:
        enriched.append(
            {
                **model,
                "model_stability_score": _stability_score(model),
                "overfitting_warning": (
                    "small validation sample"
                    if int(model.get("validation_count") or 0) < 50
                    else ""
                ),
                "small_sample_warning": (
                    f"sample_count below promotion threshold: {model_lab.get('sample_count')} < 300"
                    if int(model_lab.get("sample_count") or 0) < 300
                    else ""
                ),
            }
        )

    recommended_champion = None
    recommended_challenger = None

    eligible = [model for model in enriched if model.get("promotion_eligible")]
    if eligible:
        eligible.sort(key=lambda item: safe_float(item.get("brier")) or 999.0)
        recommended_champion = eligible[0].get("model_name")
    else:
        blockers.append("no model is promotion eligible")

    trained = [model for model in enriched if model.get("trained")]
    if trained:
        trained.sort(key=lambda item: safe_float(item.get("brier")) or 999.0)
        recommended_challenger = trained[0].get("model_name")

    report = {
        "generated_at": _utc_now(),
        "status": "ok" if models else "partial",
        "sample_count": model_lab.get("sample_count", 0),
        "validation_count": model_lab.get("validation_count", 0),
        "walkforward_oos_predictions": walk_forward.get("total_oos_predictions", 0),
        "calibration_ready": calibration.get("calibration_ready", False),
        "feature_ready_for_review_count": feature_promotion.get("ready_for_review_count", 0),
        "model_ranking_by_brier": _ranking(enriched, "brier", lower_is_better=True),
        "model_ranking_by_logloss": _ranking(enriched, "logloss", lower_is_better=True),
        "model_ranking_by_accuracy": _ranking(enriched, "accuracy", lower_is_better=False),
        "model_ranking_by_ece": _ranking(enriched, "ece", lower_is_better=True),
        "model_ranking_by_auc": _ranking(enriched, "auc", lower_is_better=False),
        "models": enriched,
        "model_vs_market_comparison": walk_forward.get("model_vs_market", {}),
        "recommended_champion": recommended_champion,
        "recommended_challenger": recommended_challenger,
        "no_promotion_reason": blockers,
        "warnings": sorted(set(warnings)),
        "shadow_only": True,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }

    write_json(output_path, report)
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
