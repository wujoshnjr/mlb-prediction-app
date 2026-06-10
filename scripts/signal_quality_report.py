from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


REPORT_DIR = Path("report")
DATA_DIR = Path("data")

EDGE_SANITY_PATH = REPORT_DIR / "edge_sanity_guardrail_report.json"
UNDERDOG_PATH = REPORT_DIR / "underdog_diagnostic_report.json"
CONFIDENCE_PATH = REPORT_DIR / "confidence_bucket_guardrail_report.json"
LINEUP_PATH = REPORT_DIR / "lineup_quality_report.json"
FRESHNESS_PATH = REPORT_DIR / "feature_freshness_report.json"
MODEL_CORRECTNESS_PATH = REPORT_DIR / "model_correctness_report.json"
SLICE_GATE_PATH = REPORT_DIR / "slice_promotion_gate_report.json"
SAMPLE_STATE_PATH = DATA_DIR / "sample_state.json"

REPORT_PATH = REPORT_DIR / "signal_quality_report.json"

MIN_PROMOTION_SAMPLE = 300


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


def _nested(report: dict[str, Any], *keys: str) -> Any:
    current: Any = report
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return parsed
    except Exception:
        return None


def _int(value: Any) -> Optional[int]:
    parsed = _float(value)
    if parsed is None:
        return None
    return int(parsed)


def _slice_metric(model_correctness: dict[str, Any], group: str, name: str) -> dict[str, Any]:
    metric = _nested(model_correctness, "slices", group, name)
    return metric if isinstance(metric, dict) else {}


def _accuracy(metric: dict[str, Any]) -> Optional[float]:
    return _float(metric.get("accuracy"))


def _sample_count(metric: dict[str, Any]) -> int:
    return _int(metric.get("sample_count")) or 0


def _case(
    *,
    case_id: str,
    label: str,
    status: str,
    evidence: dict[str, Any],
    reason: list[str],
    ui_priority: int,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "label": label,
        "status": status,
        "evidence": evidence,
        "reason": reason,
        "ui_priority": ui_priority,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }


def build_report() -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []

    reports: dict[str, dict[str, Any]] = {}
    for name, path in {
        "edge_sanity": EDGE_SANITY_PATH,
        "underdog": UNDERDOG_PATH,
        "confidence": CONFIDENCE_PATH,
        "lineup": LINEUP_PATH,
        "freshness": FRESHNESS_PATH,
        "model_correctness": MODEL_CORRECTNESS_PATH,
        "slice_gate": SLICE_GATE_PATH,
        "sample_state": SAMPLE_STATE_PATH,
    }.items():
        payload, error = _read_json(path)
        reports[name] = payload
        if error:
            warnings.append(f"{name} unavailable: {error}")

    sample_state = reports["sample_state"]
    model_correctness = reports["model_correctness"]
    edge_sanity = reports["edge_sanity"]
    underdog = reports["underdog"]
    confidence = reports["confidence"]
    freshness = reports["freshness"]

    sample_count = (
        _int(sample_state.get("clean_settled_snapshots"))
        or _int(model_correctness.get("sample_count"))
        or 0
    )
    gate_open = sample_count >= MIN_PROMOTION_SAMPLE

    favorite = _slice_metric(model_correctness, "market_role", "favorite_pick")
    underdog_metric = _slice_metric(model_correctness, "market_role", "underdog_pick")
    home = _slice_metric(model_correctness, "model_pick_side", "home")
    paper = _slice_metric(model_correctness, "recommendation_status", "PAPER_BET")
    odds_ok = _slice_metric(model_correctness, "odds_quality_status", "OK")
    edge_3_to_5 = (
        _nested(edge_sanity, "buckets", "edge_3_to_5")
        or _slice_metric(model_correctness, "edge_bucket", "edge_3_to_5")
    )
    edge_5_plus = (
        _nested(edge_sanity, "buckets", "edge_5_plus")
        or _slice_metric(model_correctness, "edge_bucket", "edge_5_plus")
    )

    edge_policy = _nested(edge_sanity, "policy") or {}
    block_large_edge = bool(edge_policy.get("block_large_edge"))
    preferred_edge_bucket = edge_policy.get("preferred_edge_bucket")

    underdog_policy = _nested(underdog, "recommendation", "underdog_policy") or "TRACKING_ONLY"
    confidence_policy = _nested(confidence, "global_policy") or {}
    unsafe_confidence_buckets = (
        confidence.get("unsafe_buckets", [])
        if isinstance(confidence.get("unsafe_buckets"), list)
        else []
    )

    cases: list[dict[str, Any]] = []

    strong_evidence = {
        "favorite_accuracy": _accuracy(favorite),
        "favorite_sample_count": _sample_count(favorite),
        "home_accuracy": _accuracy(home),
        "home_sample_count": _sample_count(home),
        "edge_3_to_5_accuracy": _accuracy(edge_3_to_5),
        "edge_3_to_5_sample_count": _sample_count(edge_3_to_5),
        "odds_ok_accuracy": _accuracy(odds_ok),
        "odds_ok_sample_count": _sample_count(odds_ok),
        "paper_bet_accuracy": _accuracy(paper),
        "paper_bet_sample_count": _sample_count(paper),
    }

    strong_signal_ready = (
        _sample_count(favorite) >= 20
        and (_accuracy(favorite) or 0) >= 0.55
        and _sample_count(edge_3_to_5) >= 20
        and (_accuracy(edge_3_to_5) or 0) >= 0.55
        and _sample_count(odds_ok) >= 20
        and (_accuracy(odds_ok) or 0) >= 0.55
    )

    cases.append(
        _case(
            case_id="strong_favorite_signal",
            label="Strong Favorite Signal",
            status="MODEL_SIGNAL_ONLY" if strong_signal_ready else "TRACKING_ONLY",
            evidence=strong_evidence,
            reason=[
                "Favorite picks, clean odds, and moderate edge are the strongest current slices.",
                "Keep shadow-only until sample gate reaches 300.",
            ],
            ui_priority=1,
        )
    )

    cases.append(
        _case(
            case_id="weak_underdog",
            label="Weak Underdog",
            status="PAPER_ENTRY_BLOCKED_BY_RISK",
            evidence={
                "underdog_accuracy": _accuracy(underdog_metric),
                "underdog_sample_count": _sample_count(underdog_metric),
                "underdog_report_accuracy": _nested(underdog, "overall", "accuracy"),
                "underdog_policy": underdog_policy,
            },
            reason=[
                "Underdog slice remains below acceptable paper-entry quality.",
                "Keep underdogs tracking-only or blocked by risk.",
            ],
            ui_priority=2,
        )
    )

    cases.append(
        _case(
            case_id="suspicious_large_edge",
            label="Suspicious Large Edge",
            status="PAPER_ENTRY_BLOCKED_BY_RISK" if block_large_edge else "TRACKING_ONLY",
            evidence={
                "edge_5_plus_accuracy": _accuracy(edge_5_plus),
                "edge_5_plus_sample_count": _sample_count(edge_5_plus),
                "preferred_edge_bucket": preferred_edge_bucket,
                "block_large_edge": block_large_edge,
                "large_edge_reasons": edge_policy.get("large_edge_reasons", []),
            },
            reason=[
                "Large edge is not automatically better.",
                "Current evidence suggests moderate edge is more reliable than extreme edge.",
            ],
            ui_priority=3,
        )
    )

    cases.append(
        _case(
            case_id="clean_paper_signal",
            label="Clean Paper Signal",
            status=(
                "MODEL_SIGNAL_ONLY"
                if (_accuracy(paper) or 0) >= 0.55 and _sample_count(paper) >= 20
                else "TRACKING_ONLY"
            ),
            evidence={
                "paper_bet_accuracy": _accuracy(paper),
                "paper_bet_sample_count": _sample_count(paper),
                "odds_ok_accuracy": _accuracy(odds_ok),
                "odds_ok_sample_count": _sample_count(odds_ok),
            },
            reason=[
                "Paper-bet and OK-odds slices are outperforming tracking-only slices.",
                "Do not expand paper volume until guardrails stay stable.",
            ],
            ui_priority=4,
        )
    )

    global_policy = {
        "global_decision": "NO_PROMOTION_SHADOW_ONLY",
        "sample_count": sample_count,
        "minimum_research_sample": MIN_PROMOTION_SAMPLE,
        "sample_gate_open": gate_open,
        "preferred_signal_recipe": [
            "market_role:favorite_pick",
            "model_pick_side:home",
            "edge_bucket:edge_3_to_5",
            "odds_quality_status:OK",
            "recommendation_status:PAPER_BET",
        ],
        "blocked_signal_recipe": [
            "market_role:underdog_pick",
            "edge_bucket:edge_5_plus_when_block_large_edge",
            "odds_quality_status:SUSPICIOUS",
            "recommendation_status:TRACKING_ONLY",
        ],
        "confidence_policy": confidence_policy,
        "unsafe_confidence_buckets": unsafe_confidence_buckets,
        "freshness_global_grade": freshness.get("global_grade"),
    }

    return {
        "generated_at": _utc_now(),
        "status": "ok" if not errors else "partial",
        "global_policy": global_policy,
        "cases": sorted(cases, key=lambda item: int(item.get("ui_priority", 999))),
        "warnings": sorted(set(warnings)),
        "errors": errors,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }


def main() -> None:
    report = build_report()
    _write_json(REPORT_PATH, report)
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
