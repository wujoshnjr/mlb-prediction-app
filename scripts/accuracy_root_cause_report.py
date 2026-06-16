from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_DIR = Path("report")
OUTPUT_PATH = REPORT_DIR / "accuracy_root_cause_report.json"

DAILY_ACCURACY_PATH = REPORT_DIR / "daily_model_accuracy_report.json"
MODEL_EVAL_PATH = REPORT_DIR / "model_eval_report.json"
BASELINE_COMPARISON_PATH = REPORT_DIR / "baseline_comparison_report.json"
WALKFORWARD_PATH = REPORT_DIR / "walkforward_evaluation.json"
FEATURE_SOURCE_COVERAGE_PATH = REPORT_DIR / "feature_source_coverage_report.json"
SHADOW_ABLATION_PATH = REPORT_DIR / "shadow_feature_ablation_report.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(child) for child in value]
    return str(value)


def _num(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def _round(value: Any, digits: int = 4) -> float | None:
    parsed = _num(value)
    return None if parsed is None else round(parsed, digits)


def _non_empty_daily_rows(daily_accuracy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = daily_accuracy.get("daily_accuracy")
    if not isinstance(rows, list):
        return []
    clean = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        sample_count = _num(item.get("sample_count")) or 0
        accuracy = _num(item.get("accuracy"))
        if sample_count <= 0 or accuracy is None:
            continue
        clean.append(item)
    return clean


def _slice_rows(daily_accuracy: dict[str, Any], official_accuracy: float | None) -> list[dict[str, Any]]:
    slices = daily_accuracy.get("slices")
    if not isinstance(slices, dict):
        return []
    results: list[dict[str, Any]] = []
    for name, payload in slices.items():
        if not isinstance(payload, dict):
            continue
        accuracy = _num(payload.get("accuracy"))
        sample_count = _num(payload.get("sample_count")) or 0
        if accuracy is None or sample_count <= 0:
            continue
        delta = accuracy - official_accuracy if official_accuracy is not None else None
        weak = accuracy < 0.5 or (delta is not None and delta <= -0.03)
        results.append(
            {
                "slice": name,
                "sample_count": int(sample_count),
                "correct": payload.get("correct"),
                "accuracy": _round(accuracy, 4),
                "delta_vs_official": _round(delta, 4),
                "weak_slice": bool(weak),
            }
        )
    return sorted(results, key=lambda item: (item["accuracy"] if item["accuracy"] is not None else 9, -item["sample_count"]))


def _degraded_days(rows: list[dict[str, Any]], rolling_30_accuracy: float | None) -> list[dict[str, Any]]:
    recent = rows[-10:]
    results = []
    for item in recent:
        sample_count = _num(item.get("sample_count")) or 0
        accuracy = _num(item.get("accuracy"))
        if accuracy is None or sample_count <= 0:
            continue
        delta = accuracy - rolling_30_accuracy if rolling_30_accuracy is not None else None
        materially_weak = accuracy < 0.45 or (sample_count >= 10 and delta is not None and delta <= -0.07)
        if not materially_weak:
            continue
        results.append(
            {
                "game_date": item.get("game_date"),
                "sample_count": int(sample_count),
                "correct": item.get("correct"),
                "accuracy": _round(accuracy, 4),
                "paper_signal_count": item.get("paper_signal_count"),
                "paper_signal_accuracy": _round(item.get("paper_signal_accuracy"), 4),
                "pending_count": item.get("pending_count"),
                "delta_vs_30d": _round(delta, 4),
            }
        )
    return results


def _inactive_signal_days(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recent = rows[-7:]
    result = []
    for item in recent:
        if (_num(item.get("paper_signal_count")) or 0) <= 0:
            result.append(
                {
                    "game_date": item.get("game_date"),
                    "sample_count": item.get("sample_count"),
                    "accuracy": _round(item.get("accuracy"), 4),
                    "paper_signal_count": item.get("paper_signal_count"),
                    "pending_count": item.get("pending_count"),
                }
            )
    return result


def _feature_gap_examples(feature_source_coverage: dict[str, Any]) -> list[dict[str, Any]]:
    categories = feature_source_coverage.get("categories") if isinstance(feature_source_coverage.get("categories"), dict) else {}
    examples: list[dict[str, Any]] = []
    for category_name in ("source_missing_all_zero", "source_missing_or_not_backfilled", "all_zero_needs_review", "feature_disabled_with_partial_signal"):
        items = categories.get(category_name)
        if not isinstance(items, list):
            continue
        for item in items[:5]:
            if not isinstance(item, dict):
                continue
            examples.append(
                {
                    "feature": item.get("feature"),
                    "feature_group": item.get("feature_group"),
                    "category": item.get("category") or category_name,
                    "zero_rate": _round(item.get("zero_rate"), 4),
                    "non_zero_rate": _round(item.get("non_zero_rate"), 4),
                    "allow_in_main_model": item.get("allow_in_main_model"),
                    "recommended_action": item.get("recommended_action"),
                    "reasons": item.get("reasons") or [],
                }
            )
    return examples[:12]


def _ablation_summary(shadow_ablation: dict[str, Any]) -> dict[str, Any]:
    summary = shadow_ablation.get("summary") if isinstance(shadow_ablation.get("summary"), dict) else {}
    results = shadow_ablation.get("feature_ablation_results") if isinstance(shadow_ablation.get("feature_ablation_results"), list) else []
    best = []
    for item in results[:8]:
        if not isinstance(item, dict):
            continue
        deltas = item.get("deltas_vs_core") if isinstance(item.get("deltas_vs_core"), dict) else {}
        best.append(
            {
                "feature": item.get("feature"),
                "decision": item.get("decision"),
                "brier_delta_vs_core": _round(deltas.get("brier_delta_vs_core"), 6),
                "logloss_delta_vs_core": _round(deltas.get("logloss_delta_vs_core"), 6),
                "accuracy_delta_vs_core": _round(deltas.get("accuracy_delta_vs_core"), 6),
            }
        )
    return {"summary": summary, "top_rejected_features": best}


def _add_cause(causes: list[dict[str, Any]], *, cause_id: str, severity: str, title: str, evidence: list[str], repair: str) -> None:
    causes.append(
        {
            "id": cause_id,
            "severity": severity,
            "title": title,
            "evidence": [str(item) for item in evidence if item is not None and str(item) != ""],
            "repair": repair,
        }
    )


def build_report() -> dict[str, Any]:
    daily_accuracy = _read_json(DAILY_ACCURACY_PATH)
    model_eval = _read_json(MODEL_EVAL_PATH)
    baseline = _read_json(BASELINE_COMPARISON_PATH)
    walkforward = _read_json(WALKFORWARD_PATH)
    feature_source_coverage = _read_json(FEATURE_SOURCE_COVERAGE_PATH)
    shadow_ablation = _read_json(SHADOW_ABLATION_PATH)

    official = daily_accuracy.get("official_accuracy") if isinstance(daily_accuracy.get("official_accuracy"), dict) else {}
    rolling = daily_accuracy.get("rolling_windows") if isinstance(daily_accuracy.get("rolling_windows"), dict) else {}
    rolling_7d = rolling.get("7d") if isinstance(rolling.get("7d"), dict) else {}
    rolling_30d = rolling.get("30d") if isinstance(rolling.get("30d"), dict) else {}

    official_accuracy = _num(official.get("accuracy"))
    rolling_7d_accuracy = _num(rolling_7d.get("accuracy"))
    rolling_30d_accuracy = _num(rolling_30d.get("accuracy"))
    recent_drop_vs_30d = rolling_7d_accuracy - rolling_30d_accuracy if rolling_7d_accuracy is not None and rolling_30d_accuracy is not None else None

    daily_rows = _non_empty_daily_rows(daily_accuracy)
    degraded_days = _degraded_days(daily_rows, rolling_30d_accuracy)
    inactive_signal_days = _inactive_signal_days(daily_rows)
    weak_slices = [item for item in _slice_rows(daily_accuracy, official_accuracy) if item.get("weak_slice")]

    metrics = model_eval.get("metrics") if isinstance(model_eval.get("metrics"), dict) else {}
    collapse = model_eval.get("collapse_guardrail") if isinstance(model_eval.get("collapse_guardrail"), dict) else {}
    quality_gate = baseline.get("quality_gate") if isinstance(baseline.get("quality_gate"), dict) else {}
    feature_summary = feature_source_coverage.get("root_summary") if isinstance(feature_source_coverage.get("root_summary"), dict) else {}
    pending = daily_accuracy.get("pending_predictions") if isinstance(daily_accuracy.get("pending_predictions"), dict) else {}

    causes: list[dict[str, Any]] = []
    if recent_drop_vs_30d is not None and recent_drop_vs_30d <= -0.03:
        _add_cause(
            causes,
            cause_id="recent_rolling_accuracy_drop",
            severity="high",
            title="近 7 日命中率低於 30 日平均",
            evidence=[
                f"7d accuracy={rolling_7d_accuracy:.4f}",
                f"30d accuracy={rolling_30d_accuracy:.4f}",
                f"delta={recent_drop_vs_30d:.4f}",
            ],
            repair="把近期低命中日和 tracking-only 天數獨立顯示；不要用單一總命中率代表模型可用性。",
        )

    if collapse.get("do_not_promote") is True or collapse.get("model_has_no_discrimination_power") is True:
        _add_cause(
            causes,
            cause_id="model_collapse_guardrail_failed",
            severity="critical",
            title="模型評估觸發 collapse guardrail",
            evidence=[
                f"roc_auc={metrics.get('roc_auc')}",
                f"balanced_accuracy={metrics.get('balanced_accuracy')}",
                f"predicted_positive_rate={metrics.get('predicted_positive_rate')}",
                ", ".join(str(item) for item in (collapse.get("collapse_reasons") or [])[:6]),
            ],
            repair="正式站只顯示 tracking-only；模型不可 promotion。先修 feature coverage、再重跑 time-ordered evaluation。",
        )

    if quality_gate.get("promotion_allowed") is False:
        _add_cause(
            causes,
            cause_id="baseline_quality_gate_blocked",
            severity="critical",
            title="模型沒有打贏必要 baseline",
            evidence=[", ".join(str(item) for item in (quality_gate.get("reasons") or [])[:6])],
            repair="新增頁面直接顯示 baseline blocker；模型要先打贏 constant_50、home historical、market no-vig 的 Brier/Logloss 才能升級。",
        )

    if (walkforward.get("collapse_fold_count") or 0) and _num(walkforward.get("collapse_rate")) and (_num(walkforward.get("collapse_rate")) or 0) >= 0.4:
        _add_cause(
            causes,
            cause_id="walkforward_recent_fold_collapse",
            severity="high",
            title="時間序列 walk-forward 多個 fold 退化",
            evidence=[
                f"fold_count={walkforward.get('fold_count')}",
                f"collapse_fold_count={walkforward.get('collapse_fold_count')}",
                f"collapse_rate={walkforward.get('collapse_rate')}",
                f"settled_oos_predictions={walkforward.get('settled_oos_predictions')}",
            ],
            repair="把 walk-forward fold collapse 當作發布 gate；近期 fold 未恢復前，網站只做追蹤，不做正式推薦。",
        )

    if inactive_signal_days and len(inactive_signal_days) >= 3:
        _add_cause(
            causes,
            cause_id="paper_signal_drought",
            severity="medium",
            title="近期 paper signal 幾乎消失，盤面多為 tracking-only",
            evidence=[f"recent inactive signal days={len(inactive_signal_days)}", f"pending_predictions={pending.get('count')}"] ,
            repair="網站新增 paper signal / tracking-only 分離；official accuracy 之外另顯示可用訊號命中率。",
        )

    missing_or_partial_count = sum(int(feature_summary.get(key) or 0) for key in ("source_missing_all_zero_count", "source_missing_or_not_backfilled_count", "feature_disabled_with_partial_signal_count"))
    if missing_or_partial_count > 0:
        _add_cause(
            causes,
            cause_id="feature_source_incomplete",
            severity="high",
            title="重要 MLB 特徵資料源不完整，無法安全進主模型",
            evidence=[
                f"source_missing_all_zero_count={feature_summary.get('source_missing_all_zero_count')}",
                f"source_missing_or_not_backfilled_count={feature_summary.get('source_missing_or_not_backfilled_count')}",
                f"feature_disabled_with_partial_signal_count={feature_summary.get('feature_disabled_with_partial_signal_count')}",
            ],
            repair="先做 starting pitcher、lineup、statcast、bullpen 的 freshness/backfill audit；修完前不得把這些特徵 promotion。",
        )

    status = "blocked" if any(item["severity"] == "critical" for item in causes) else "warning"
    report = {
        "generated_at": _utc_now(),
        "report_type": "accuracy_root_cause_report",
        "status": status,
        "summary": {
            "official_sample_count": official.get("sample_count"),
            "official_accuracy": _round(official_accuracy, 4),
            "rolling_7d_sample_count": rolling_7d.get("sample_count"),
            "rolling_7d_accuracy": _round(rolling_7d_accuracy, 4),
            "rolling_30d_sample_count": rolling_30d.get("sample_count"),
            "rolling_30d_accuracy": _round(rolling_30d_accuracy, 4),
            "recent_drop_vs_30d": _round(recent_drop_vs_30d, 4),
            "degraded_day_count": len(degraded_days),
            "weak_slice_count": len(weak_slices),
            "inactive_signal_day_count": len(inactive_signal_days),
            "root_cause_count": len(causes),
        },
        "root_causes": causes,
        "degraded_days": degraded_days,
        "inactive_signal_days": inactive_signal_days,
        "weak_slices": weak_slices,
        "model_collapse": {
            "status": collapse.get("status"),
            "model_has_no_discrimination_power": collapse.get("model_has_no_discrimination_power"),
            "do_not_promote": collapse.get("do_not_promote"),
            "collapse_reasons": collapse.get("collapse_reasons") or [],
            "metrics": {
                "accuracy": _round(metrics.get("accuracy"), 4),
                "balanced_accuracy": _round(metrics.get("balanced_accuracy"), 4),
                "roc_auc": _round(metrics.get("roc_auc"), 4),
                "brier": _round(metrics.get("brier"), 4),
                "logloss": _round(metrics.get("logloss"), 4),
                "predicted_positive_rate": _round(metrics.get("predicted_positive_rate"), 4),
                "predicted_negative_rate": _round(metrics.get("predicted_negative_rate"), 4),
            },
        },
        "baseline_gate": {
            "status": quality_gate.get("status"),
            "promotion_allowed": quality_gate.get("promotion_allowed"),
            "reasons": quality_gate.get("reasons") or [],
        },
        "walkforward": {
            "status": walkforward.get("status"),
            "settled_oos_predictions": walkforward.get("settled_oos_predictions"),
            "min_required_oos_predictions": walkforward.get("min_required_oos_predictions"),
            "fold_count": walkforward.get("fold_count"),
            "collapse_fold_count": walkforward.get("collapse_fold_count"),
            "collapse_rate": walkforward.get("collapse_rate"),
            "median_model_brier": walkforward.get("median_model_brier"),
            "median_model_logloss": walkforward.get("median_model_logloss"),
        },
        "feature_source_summary": feature_summary,
        "feature_gap_examples": _feature_gap_examples(feature_source_coverage),
        "shadow_ablation": _ablation_summary(shadow_ablation),
        "corrective_actions": [
            "Separate official accuracy, paper-signal accuracy, and tracking-only accuracy on the public site.",
            "Keep promotion_allowed=false until collapse guardrail, baseline gate, and walk-forward folds improve together.",
            "Run feature source freshness/backfill audit for starting pitcher, lineup, statcast batting, and bullpen groups.",
            "Do not promote partially zero-filled shadow features just because they look domain-relevant; require Brier and logloss improvement.",
        ],
        "promotion_allowed": False,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }
    return report


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    OUTPUT_PATH.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report.get("status"), "output_path": str(OUTPUT_PATH), "root_cause_count": report.get("summary", {}).get("root_cause_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
