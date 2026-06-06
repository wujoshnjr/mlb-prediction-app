from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPORT_DIR = Path("report")
OUTPUT_PATH = REPORT_DIR / "research_quality_report.json"

INPUT_REPORTS = {
    "baseline": REPORT_DIR / "baseline_comparison_report.json",
    "clv_edge": REPORT_DIR / "clv_by_edge_bucket.json",
    "clv_side": REPORT_DIR / "clv_by_side.json",
    "clv_odds": REPORT_DIR / "clv_by_odds_range.json",
    "clv_lineup": REPORT_DIR / "clv_by_lineup_status.json",
    "calibration": REPORT_DIR / "calibration_report.json",
    "walkforward": REPORT_DIR / "walkforward_evaluation.json",
    "rolling_walkforward": REPORT_DIR / "rolling_walkforward_evaluation.json",
    "lineup_starter_slice": REPORT_DIR / "lineup_starter_slice_report.json",
    "market_close": REPORT_DIR / "market_close_report.json",
}


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


def _bool(value: Any) -> bool:
    return bool(value) is True


def _find_avg_clv_and_positive_rate(*reports: Optional[Dict[str, Any]]) -> tuple[Optional[float], Optional[float]]:
    for report in reports:
        if not report:
            continue

        avg_clv = report.get("avg_clv")
        positive_rate = report.get("positive_clv_rate")

        if avg_clv is not None or positive_rate is not None:
            try:
                avg = float(avg_clv) if avg_clv is not None else None
            except Exception:
                avg = None
            try:
                rate = float(positive_rate) if positive_rate is not None else None
            except Exception:
                rate = None
            return avg, rate

    return None, None


def _clv_report_has_slices(report: Optional[Dict[str, Any]]) -> bool:
    if not report:
        return False
    slices = report.get("slices")
    return isinstance(slices, list) and len(slices) > 0


def build_report() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    reports: Dict[str, Optional[Dict[str, Any]]] = {}
    input_files: Dict[str, Any] = {}
    warnings: List[str] = []

    for name, path in INPUT_REPORTS.items():
        data, status = _read_json(path)
        reports[name] = data
        input_files[name] = status
        if data is None:
            warnings.append(f"Missing or invalid report: {path}")

    baseline = reports.get("baseline")
    calibration = reports.get("calibration")
    walkforward = reports.get("walkforward")
    rolling = reports.get("rolling_walkforward")
    market_close = reports.get("market_close")
    lineup_starter = reports.get("lineup_starter_slice")

    baseline_ready = baseline is not None and baseline.get("status") in {"ok", "partial"}
    clv_ready = any(
        _clv_report_has_slices(reports.get(key))
        for key in ("clv_edge", "clv_side", "clv_odds", "clv_lineup")
    )
    calibration_ready = bool(calibration and calibration.get("calibration_ready") is True)
    walkforward_ready = bool(
        (walkforward and walkforward.get("walkforward_ready") is True)
        or (rolling and rolling.get("status") == "ok")
    )
    market_close_ready = bool(
        market_close
        and market_close.get("status") in {"ok", "partial"}
        and market_close.get("closing_odds_coverage_rate") is not None
    )
    lineup_starter_slice_ready = bool(
        lineup_starter
        and isinstance(lineup_starter.get("slices"), list)
        and len(lineup_starter.get("slices")) > 0
    )

    model_beats_market = False
    if rolling:
        model_beats_market = bool(
            rolling.get("model_beats_market_brier")
            and rolling.get("model_beats_market_logloss")
        )
    elif baseline:
        comparison = baseline.get("comparison") or {}
        model_beats_market = bool(
            comparison.get("model_beats_market_brier")
            and comparison.get("model_beats_market_logloss")
        )

    avg_clv, positive_rate = _find_avg_clv_and_positive_rate(rolling, walkforward)
    avg_clv_positive = avg_clv is not None and avg_clv > 0
    positive_clv_rate_ok = positive_rate is not None and positive_rate > 0.55

    blockers: List[str] = []
    next_actions: List[str] = []

    if not baseline_ready:
        blockers.append("baseline report not ready")
        next_actions.append("Generate baseline_comparison_report.json")

    if not clv_ready:
        blockers.append("CLV slice reports not ready")
        next_actions.append("Populate per-pick CLV and CLV slice reports")

    if not calibration_ready:
        blockers.append("calibration not ready")
        next_actions.append("Accumulate at least 500 settled predictions for calibration")

    if not walkforward_ready:
        blockers.append("walk-forward validation not ready")
        next_actions.append("Accumulate at least 300 rolling OOS predictions")

    if not market_close_ready:
        next_actions.append("Improve closing odds coverage")

    if not lineup_starter_slice_ready:
        next_actions.append("Improve lineup/starter slice coverage")

    ready_count = sum(
        [
            baseline_ready,
            clv_ready,
            calibration_ready,
            walkforward_ready,
            market_close_ready,
            lineup_starter_slice_ready,
        ]
    )

    if ready_count >= 6 and model_beats_market and avg_clv_positive and positive_clv_rate_ok:
        research_grade = "A"
    elif ready_count >= 5:
        research_grade = "B"
    elif ready_count >= 3:
        research_grade = "C"
    elif ready_count >= 1:
        research_grade = "D"
    else:
        research_grade = "F"

    status = "ok" if research_grade in {"A", "B", "C"} else "partial"

    report = {
        "generated_at": _utc_now(),
        "status": status,
        "input_files": input_files,
        "baseline_ready": baseline_ready,
        "clv_ready": clv_ready,
        "calibration_ready": calibration_ready,
        "walkforward_ready": walkforward_ready,
        "market_close_ready": market_close_ready,
        "lineup_starter_slice_ready": lineup_starter_slice_ready,
        "model_beats_market": model_beats_market,
        "avg_clv_positive": avg_clv_positive,
        "positive_clv_rate_ok": positive_clv_rate_ok,
        "research_grade": research_grade,
        "blockers": blockers,
        "warnings": warnings,
        "errors": [],
        "next_actions": sorted(set(next_actions)),
        "recommendations": [
            "Use this report for research readiness only; it must not enable live betting."
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
