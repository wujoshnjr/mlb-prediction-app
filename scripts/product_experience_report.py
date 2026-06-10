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

MODEL_CORRECTNESS_PATH = REPORT_DIR / "model_correctness_report.json"
EDGE_SANITY_PATH = REPORT_DIR / "edge_sanity_guardrail_report.json"
SIGNAL_QUALITY_PATH = REPORT_DIR / "signal_quality_report.json"
UNDERDOG_PATH = REPORT_DIR / "underdog_diagnostic_report.json"
CONFIDENCE_PATH = REPORT_DIR / "confidence_bucket_guardrail_report.json"
SLICE_GATE_PATH = REPORT_DIR / "slice_promotion_gate_report.json"
LINEUP_PATH = REPORT_DIR / "lineup_quality_report.json"
FRESHNESS_PATH = REPORT_DIR / "feature_freshness_report.json"
READINESS_PATH = REPORT_DIR / "research_promotion_readiness_report.json"

REPORT_PATH = REPORT_DIR / "product_experience_report.json"


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


def _pct(value: Any, *, decimals: int = 1) -> str:
    parsed = _float(value)
    if parsed is None:
        return "--"
    return f"{parsed * 100:.{decimals}f}%"


def _metric_card(
    *,
    card_id: str,
    label: str,
    value: Any,
    display_value: str,
    sample_count: Any = None,
    status: str = "neutral",
    caption: str = "",
    priority: int = 100,
) -> dict[str, Any]:
    return {
        "card_id": card_id,
        "label": label,
        "value": value,
        "display_value": display_value,
        "sample_count": sample_count,
        "status": status,
        "caption": caption,
        "priority": priority,
    }


def _slice_metric(model_correctness: dict[str, Any], group: str, name: str) -> dict[str, Any]:
    metric = _nested(model_correctness, "slices", group, name)
    return metric if isinstance(metric, dict) else {}


def _accuracy(metric: dict[str, Any]) -> Optional[float]:
    return _float(metric.get("accuracy"))


def _sample_count(metric: dict[str, Any]) -> int:
    return _int(metric.get("sample_count")) or 0


def _status_from_accuracy(value: Optional[float], *, good: float = 0.55, bad: float = 0.50) -> str:
    if value is None:
        return "waiting"
    if value >= good:
        return "positive"
    if value < bad:
        return "negative"
    return "neutral"


def _build_hero_metrics(
    *,
    sample_state: dict[str, Any],
    model_correctness: dict[str, Any],
    signal_quality: dict[str, Any],
    edge_sanity: dict[str, Any],
) -> list[dict[str, Any]]:
    overall_accuracy = _float(model_correctness.get("overall_accuracy"))
    sample_count = _int(model_correctness.get("sample_count")) or _int(sample_state.get("clean_settled_snapshots")) or 0
    signal_policy = signal_quality.get("global_policy", {}) if isinstance(signal_quality.get("global_policy"), dict) else {}
    sample_gate_open = bool(signal_policy.get("sample_gate_open", False))
    preferred_edge = _nested(edge_sanity, "policy", "preferred_edge_bucket")
    block_large_edge = bool(_nested(edge_sanity, "policy", "block_large_edge"))

    return [
        _metric_card(
            card_id="overall_accuracy",
            label="Overall Accuracy",
            value=overall_accuracy,
            display_value=_pct(overall_accuracy),
            sample_count=sample_count,
            status=_status_from_accuracy(overall_accuracy),
            caption=f"{sample_count} settled samples",
            priority=1,
        ),
        _metric_card(
            card_id="sample_gate",
            label="Research Sample Gate",
            value=sample_count,
            display_value=f"{sample_count} / 300",
            sample_count=sample_count,
            status="positive" if sample_gate_open else "waiting",
            caption="Needed before research promotion opens",
            priority=2,
        ),
        _metric_card(
            card_id="preferred_edge",
            label="Preferred Edge Bucket",
            value=preferred_edge,
            display_value=str(preferred_edge or "--").replace("_", " "),
            sample_count=None,
            status="positive" if preferred_edge else "waiting",
            caption="Moderate edge is preferred over extreme edge",
            priority=3,
        ),
        _metric_card(
            card_id="large_edge_guardrail",
            label="Large Edge Guardrail",
            value=block_large_edge,
            display_value="Blocked" if block_large_edge else "Tracking",
            sample_count=None,
            status="negative" if block_large_edge else "waiting",
            caption="Extreme edge is downgraded when evidence is weak",
            priority=4,
        ),
    ]


def _build_evidence_cards(
    *,
    model_correctness: dict[str, Any],
    edge_sanity: dict[str, Any],
    underdog: dict[str, Any],
) -> list[dict[str, Any]]:
    favorite = _slice_metric(model_correctness, "market_role", "favorite_pick")
    underdog_metric = _slice_metric(model_correctness, "market_role", "underdog_pick")
    home = _slice_metric(model_correctness, "model_pick_side", "home")
    away = _slice_metric(model_correctness, "model_pick_side", "away")
    odds_ok = _slice_metric(model_correctness, "odds_quality_status", "OK")
    paper = _slice_metric(model_correctness, "recommendation_status", "PAPER_BET")
    tracking = _slice_metric(model_correctness, "recommendation_status", "TRACKING_ONLY")
    edge_3_to_5 = _nested(edge_sanity, "buckets", "edge_3_to_5") or _slice_metric(model_correctness, "edge_bucket", "edge_3_to_5")
    edge_5_plus = _nested(edge_sanity, "buckets", "edge_5_plus") or _slice_metric(model_correctness, "edge_bucket", "edge_5_plus")
    confidence_55_60 = _slice_metric(model_correctness, "confidence_bucket", "55_60")
    confidence_60_65 = _slice_metric(model_correctness, "confidence_bucket", "60_65")

    definitions = [
        ("favorite_pick", "Favorite Pick", favorite, 1, "Market favorite model picks"),
        ("underdog_pick", "Underdog Pick", underdog_metric, 2, "Weak slice; risk-blocked"),
        ("home_pick", "Home Pick", home, 3, "Home side model picks"),
        ("away_pick", "Away Pick", away, 4, "Away side model picks"),
        ("odds_ok", "Odds OK", odds_ok, 5, "Verified odds quality"),
        ("paper_bet", "Paper Bet", paper, 6, "Existing paper-entry gate"),
        ("tracking_only", "Tracking Only", tracking, 7, "Non-entry tracking signals"),
        ("edge_3_to_5", "Edge 3–5%", edge_3_to_5, 8, "Best current edge slice"),
        ("edge_5_plus", "Edge 5%+", edge_5_plus, 9, "Suspicious large edge"),
        ("confidence_55_60", "Confidence 55–60%", confidence_55_60, 10, "Stable confidence range"),
        ("confidence_60_65", "Confidence 60–65%", confidence_60_65, 11, "Secondary confidence range"),
    ]

    cards: list[dict[str, Any]] = []
    for card_id, label, metric, priority, caption in definitions:
        accuracy = _accuracy(metric)
        sample_count = _sample_count(metric)
        cards.append(
            _metric_card(
                card_id=card_id,
                label=label,
                value=accuracy,
                display_value=_pct(accuracy),
                sample_count=sample_count,
                status=_status_from_accuracy(accuracy),
                caption=f"{sample_count} samples · {caption}",
                priority=priority,
            )
        )

    underdog_policy = _nested(underdog, "recommendation", "underdog_policy")
    if underdog_policy:
        cards.append(
            _metric_card(
                card_id="underdog_policy",
                label="Underdog Policy",
                value=underdog_policy,
                display_value=str(underdog_policy).replace("_", " "),
                sample_count=_nested(underdog, "underdog_sample_count"),
                status="negative" if str(underdog_policy).upper() != "PAPER_ENTRY_ALLOWED" else "positive",
                caption="Underdog remains blocked or tracking-only unless evidence improves",
                priority=12,
            )
        )

    return sorted(cards, key=lambda card: int(card.get("priority", 999)))


def _build_signal_case_cards(signal_quality: dict[str, Any]) -> list[dict[str, Any]]:
    cases = signal_quality.get("cases", [])
    if not isinstance(cases, list):
        return []

    status_order = {
        "MODEL_SIGNAL_ONLY": "positive",
        "PAPER_ENTRY_BLOCKED_BY_RISK": "negative",
        "TRACKING_ONLY": "waiting",
        "NO_SIGNAL": "neutral",
    }

    cards: list[dict[str, Any]] = []
    for item in cases:
        if not isinstance(item, dict):
            continue

        status = str(item.get("status") or "TRACKING_ONLY")
        evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        reasons = item.get("reason") if isinstance(item.get("reason"), list) else []

        cards.append(
            {
                "case_id": item.get("case_id"),
                "label": item.get("label"),
                "status": status,
                "display_status": status.replace("_", " "),
                "ui_status": status_order.get(status, "neutral"),
                "evidence": evidence,
                "reason": reasons,
                "summary": reasons[0] if reasons else "",
                "ui_priority": int(item.get("ui_priority") or 999),
                "live_betting_allowed": False,
                "automated_wagering_allowed": False,
                "production_model_replacement_allowed": False,
            }
        )

    return sorted(cards, key=lambda card: int(card.get("ui_priority", 999)))


def _build_risk_cards(
    *,
    edge_sanity: dict[str, Any],
    underdog: dict[str, Any],
    confidence: dict[str, Any],
    slice_gate: dict[str, Any],
    freshness: dict[str, Any],
) -> list[dict[str, Any]]:
    edge_policy = edge_sanity.get("policy", {}) if isinstance(edge_sanity.get("policy"), dict) else {}
    confidence_policy = confidence.get("global_policy", {}) if isinstance(confidence.get("global_policy"), dict) else {}
    paper_entry_policy = slice_gate.get("paper_entry_policy", {}) if isinstance(slice_gate.get("paper_entry_policy"), dict) else {}

    cards = [
        {
            "risk_id": "large_edge",
            "label": "Large Edge Risk",
            "status": "blocked" if edge_policy.get("block_large_edge") else "tracking",
            "display_status": edge_policy.get("large_edge_policy", "TRACKING_ONLY"),
            "reasons": edge_policy.get("large_edge_reasons", []),
            "description": "Extreme edge is downgraded when historical evidence underperforms moderate edge.",
            "priority": 1,
        },
        {
            "risk_id": "underdog",
            "label": "Underdog Risk",
            "status": "blocked",
            "display_status": _nested(underdog, "recommendation", "underdog_policy") or "TRACKING_ONLY",
            "reasons": _nested(underdog, "recommendation", "block_when") or [],
            "description": "Underdog picks remain weak and should not be upgraded.",
            "priority": 2,
        },
        {
            "risk_id": "confidence_language",
            "label": "Confidence Language",
            "status": "allowed" if not confidence_policy.get("block_high_confidence_language") else "capped",
            "display_status": "CAPPED" if confidence_policy.get("block_high_confidence_language") else "SHADOW_ONLY",
            "reasons": confidence_policy.get("reason", []),
            "description": "High-confidence wording is governed separately from model probability.",
            "priority": 3,
        },
        {
            "risk_id": "sample_gate",
            "label": "Sample Gate",
            "status": "blocked",
            "display_status": slice_gate.get("global_decision", "NO_PROMOTION_SHADOW_ONLY"),
            "reasons": slice_gate.get("blockers", []),
            "description": "Promotion is blocked until minimum research sample is reached.",
            "priority": 4,
        },
        {
            "risk_id": "freshness",
            "label": "Data Freshness",
            "status": "ok" if freshness.get("global_grade") in {"A", "B"} else "warning",
            "display_status": freshness.get("global_grade", "--"),
            "reasons": freshness.get("stale_sources", []),
            "description": "Fresh data improves interpretability of game-board signals.",
            "priority": 5,
        },
    ]

    if paper_entry_policy:
        cards.append(
            {
                "risk_id": "paper_entry_policy",
                "label": "Paper Entry Policy",
                "status": "tracking",
                "display_status": "SLICE_CONTROLLED",
                "reasons": [f"{key}: {value}" for key, value in sorted(paper_entry_policy.items())],
                "description": "Each signal slice has its own model-safety status.",
                "priority": 6,
            }
        )

    return sorted(cards, key=lambda card: int(card.get("priority", 999)))


def _build_copy_blocks() -> dict[str, str]:
    return {
        "hero_title": "MLB Intelligence Cloud",
        "hero_subtitle": "AI Sports Research Terminal",
        "hero_body": (
            "A paper-only MLB research dashboard for market comparison, model evidence, "
            "signal quality, lineup context, and safety guardrails."
        ),
        "signal_center_intro": (
            "Signals are grouped by evidence quality. Strong Favorite and Clean Paper are "
            "model-signal-only research cases; Weak Underdog and Suspicious Large Edge remain blocked."
        ),
        "evidence_center_intro": (
            "Evidence cards summarize settled-sample accuracy by market role, confidence, edge, odds quality, "
            "and recommendation status."
        ),
        "safety_footer": (
            "Live betting is disabled. No automated wagering. No user funds. "
            "All outputs are for paper tracking and model research."
        ),
    }


def build_report() -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []

    reports: dict[str, dict[str, Any]] = {}
    for name, path in {
        "sample_state": SAMPLE_STATE_PATH,
        "model_correctness": MODEL_CORRECTNESS_PATH,
        "edge_sanity": EDGE_SANITY_PATH,
        "signal_quality": SIGNAL_QUALITY_PATH,
        "underdog": UNDERDOG_PATH,
        "confidence": CONFIDENCE_PATH,
        "slice_gate": SLICE_GATE_PATH,
        "lineup": LINEUP_PATH,
        "freshness": FRESHNESS_PATH,
        "readiness": READINESS_PATH,
    }.items():
        payload, error = _read_json(path)
        reports[name] = payload
        if error:
            warnings.append(f"{name} unavailable: {error}")

    sample_state = reports["sample_state"]
    model_correctness = reports["model_correctness"]
    edge_sanity = reports["edge_sanity"]
    signal_quality = reports["signal_quality"]
    underdog = reports["underdog"]
    confidence = reports["confidence"]
    slice_gate = reports["slice_gate"]
    freshness = reports["freshness"]

    hero_metrics = _build_hero_metrics(
        sample_state=sample_state,
        model_correctness=model_correctness,
        signal_quality=signal_quality,
        edge_sanity=edge_sanity,
    )

    evidence_cards = _build_evidence_cards(
        model_correctness=model_correctness,
        edge_sanity=edge_sanity,
        underdog=underdog,
    )

    signal_case_cards = _build_signal_case_cards(signal_quality)

    risk_cards = _build_risk_cards(
        edge_sanity=edge_sanity,
        underdog=underdog,
        confidence=confidence,
        slice_gate=slice_gate,
        freshness=freshness,
    )

    recommended_ui_sections = [
        {
            "section_id": "today_overview",
            "label": "Today Overview",
            "description": "High-level market and research status for the day.",
            "priority": 1,
        },
        {
            "section_id": "signal_center",
            "label": "Signal Center",
            "description": "Group games by signal quality and risk classification.",
            "priority": 2,
        },
        {
            "section_id": "game_board",
            "label": "Game Board",
            "description": "Single-game cards with model, market, edge, and guardrail context.",
            "priority": 3,
        },
        {
            "section_id": "evidence_center",
            "label": "Evidence Center",
            "description": "Settled-sample slices that explain which signals are currently working.",
            "priority": 4,
        },
        {
            "section_id": "risk_center",
            "label": "Risk Center",
            "description": "Explicitly shows why certain cases remain tracking-only or blocked.",
            "priority": 5,
        },
    ]

    product_summary = {
        "mode": "MLB_ONLY_PAPER_RESEARCH",
        "primary_strength": "favorite/home/odds-ok/paper-bet/moderate-edge signals",
        "primary_risks": [
            "underdog slice remains weak",
            "edge_5_plus is downgraded by edge sanity guardrail",
            "sample gate remains below research promotion threshold",
        ],
        "sample_count": _int(model_correctness.get("sample_count")) or _int(sample_state.get("clean_settled_snapshots")) or 0,
        "overall_accuracy": model_correctness.get("overall_accuracy"),
        "production_allowed": False,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
    }

    return {
        "generated_at": _utc_now(),
        "status": "ok" if not errors else "partial",
        "product_summary": product_summary,
        "hero_metrics": hero_metrics,
        "signal_case_cards": signal_case_cards,
        "evidence_cards": evidence_cards,
        "risk_cards": risk_cards,
        "recommended_ui_sections": recommended_ui_sections,
        "copy_blocks": _build_copy_blocks(),
        "source_reports": {
            "sample_state": str(SAMPLE_STATE_PATH),
            "model_correctness": str(MODEL_CORRECTNESS_PATH),
            "edge_sanity": str(EDGE_SANITY_PATH),
            "signal_quality": str(SIGNAL_QUALITY_PATH),
            "underdog": str(UNDERDOG_PATH),
            "confidence": str(CONFIDENCE_PATH),
            "slice_gate": str(SLICE_GATE_PATH),
            "lineup": str(LINEUP_PATH),
            "freshness": str(FRESHNESS_PATH),
            "readiness": str(READINESS_PATH),
        },
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
