from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


EDGE_BUCKETS = {
    "3-5%": (0.03, 0.05),
    "5-8%": (0.05, 0.08),
    "8%+": (0.08, float("inf")),
}


def _current_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    number = _safe_float(value)
    if number is None:
        return None
    return int(number)


def _safe_read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        return pd.read_csv(path), ""
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except Exception as exc:
        return None, f"Error reading {path}: {exc}"


def _clip_probability(probability: Optional[float]) -> Optional[float]:
    if probability is None:
        return None
    return min(max(float(probability), 1e-15), 1.0 - 1e-15)


def _logloss(probability: Optional[float], outcome: Optional[int]) -> Optional[float]:
    probability = _clip_probability(probability)
    if probability is None or outcome not in (0, 1):
        return None
    return float(
        -(outcome * math.log(probability) + (1 - outcome) * math.log(1 - probability))
    )


def _brier(probability: Optional[float], outcome: Optional[int]) -> Optional[float]:
    probability = _clip_probability(probability)
    if probability is None or outcome not in (0, 1):
        return None
    return float((probability - outcome) ** 2)


def _decimal_clv(entry_odds: Optional[float], closing_odds: Optional[float]) -> Optional[float]:
    if entry_odds is None or closing_odds is None:
        return None
    if entry_odds <= 1 or closing_odds <= 1:
        return None
    return float(closing_odds - entry_odds)


def _selected_side(row: pd.Series) -> str:
    edge = _safe_float(row.get("model_edge_home"))
    recommendation = str(row.get("moneyline_recommendation", "")).lower()
    home_team = str(row.get("home_team", "")).lower()
    away_team = str(row.get("away_team", "")).lower()

    if edge is not None:
        if edge >= 0.03:
            return "home"
        if edge <= -0.03:
            return "away"

    if home_team and home_team in recommendation:
        return "home"
    if away_team and away_team in recommendation:
        return "away"
    return ""


def _entry_odds(row: pd.Series, side: str) -> Optional[float]:
    if side == "home":
        return _safe_float(row.get("home_moneyline_odds"))
    if side == "away":
        return _safe_float(row.get("away_moneyline_odds"))
    return None


def _closing_odds(row: pd.Series, side: str) -> Optional[float]:
    if side == "home":
        return _safe_float(row.get("closing_home_odds"))
    if side == "away":
        return _safe_float(row.get("closing_away_odds"))
    return None


def _selected_model_probability(row: pd.Series, side: str) -> Optional[float]:
    home_prob = (
        _safe_float(row.get("premarket_model_home_prob"))
        or _safe_float(row.get("predicted_home_win_pct"))
        or _safe_float(row.get("displayed_home_win_pct"))
    )
    if home_prob is None:
        return None
    if side == "home":
        return home_prob
    if side == "away":
        return 1.0 - home_prob
    return None


def _selected_market_probability(row: pd.Series, side: str) -> Optional[float]:
    home_prob = _safe_float(row.get("market_no_vig_home_prob"))
    if home_prob is None:
        return None
    if side == "home":
        return home_prob
    if side == "away":
        return 1.0 - home_prob
    return None


def _selected_outcome(row: pd.Series, side: str) -> Optional[int]:
    home_win = _safe_int(row.get("home_win"))
    if home_win not in (0, 1):
        return None
    if side == "home":
        return 1 if home_win == 1 else 0
    if side == "away":
        return 1 if home_win == 0 else 0
    return None


def _edge_bucket(edge_abs: Optional[float]) -> str:
    if edge_abs is None:
        return "below_threshold"
    for name, (low, high) in EDGE_BUCKETS.items():
        if low <= edge_abs < high:
            return name
    return "below_threshold"


def _is_health_degraded(row: pd.Series) -> bool:
    statcast_zero = (
        (_safe_float(row.get("statcast_woba_diff")) or 0.0) == 0.0
        and (_safe_float(row.get("statcast_barrel_diff")) or 0.0) == 0.0
        and (_safe_float(row.get("statcast_hard_hit_diff")) or 0.0) == 0.0
    )
    top3_zero = (_safe_float(row.get("top3_woba_diff")) or 0.0) == 0.0
    lineup_missing = str(row.get("recommendation_status", "")).upper() != "PAPER_BET"
    return bool(statcast_zero or top3_zero or lineup_missing)


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "win_rate": None,
            "avg_clv": None,
            "model_brier": None,
            "market_brier": None,
            "model_logloss": None,
            "market_logloss": None,
            "beaten_market_brier": False,
            "beaten_market_logloss": False,
        }

    wins = [r["won"] for r in rows if r.get("won") in (0, 1)]
    clvs = [r["clv"] for r in rows if r.get("clv") is not None]
    model_briers = [r["model_brier"] for r in rows if r.get("model_brier") is not None]
    market_briers = [r["market_brier"] for r in rows if r.get("market_brier") is not None]
    model_ll = [r["model_logloss"] for r in rows if r.get("model_logloss") is not None]
    market_ll = [r["market_logloss"] for r in rows if r.get("market_logloss") is not None]

    def avg(values: List[float]) -> Optional[float]:
        return float(sum(values) / len(values)) if values else None

    mb = avg(model_briers)
    xb = avg(market_briers)
    ml = avg(model_ll)
    xl = avg(market_ll)

    return {
        "count": len(rows),
        "win_rate": avg(wins),
        "avg_clv": avg(clvs),
        "model_brier": mb,
        "market_brier": xb,
        "model_logloss": ml,
        "market_logloss": xl,
        "beaten_market_brier": bool(mb is not None and xb is not None and mb < xb),
        "beaten_market_logloss": bool(ml is not None and xl is not None and ml < xl),
    }


def build_market_edge_research(
    snapshots_path: str = "data/prediction_snapshots.csv",
    output_path: str = "report/market_edge_research.json",
) -> Dict[str, Any]:
    frame, error = _safe_read_csv(Path(snapshots_path))

    report: Dict[str, Any] = {
        "generated_at": _current_utc_iso(),
        "input_file": {
            "path": snapshots_path,
            "exists": frame is not None,
            "error": error,
            "rows": int(len(frame)) if frame is not None else 0,
        },
        "settled_rows": 0,
        "paper_bet_rows": 0,
        "overall": {},
        "edge_buckets": {},
        "data_health_buckets": {},
        "promote_ready": False,
        "promote_reasons": [],
        "recommendations": [],
    }

    if frame is None or frame.empty:
        report["recommendations"].append("No snapshot rows available.")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    rows: List[Dict[str, Any]] = []

    for _, row in frame.iterrows():
        if str(row.get("recommendation_status", "")).upper() != "PAPER_BET":
            continue

        side = _selected_side(row)
        if side not in {"home", "away"}:
            continue

        outcome = _selected_outcome(row, side)
        if outcome is None:
            continue

        model_prob = _selected_model_probability(row, side)
        market_prob = _selected_market_probability(row, side)

        edge_home = _safe_float(row.get("model_edge_home"))
        edge_abs = abs(edge_home) if edge_home is not None else None

        entry = _entry_odds(row, side)
        close = _closing_odds(row, side)

        completed = {
            "game_id": str(row.get("game_id", "")),
            "side": side,
            "won": outcome,
            "edge_abs": edge_abs,
            "edge_bucket": _edge_bucket(edge_abs),
            "clv": _decimal_clv(entry, close),
            "model_brier": _brier(model_prob, outcome),
            "market_brier": _brier(market_prob, outcome),
            "model_logloss": _logloss(model_prob, outcome),
            "market_logloss": _logloss(market_prob, outcome),
            "health_bucket": "degraded" if _is_health_degraded(row) else "healthy",
        }
        rows.append(completed)

    report["settled_rows"] = len(rows)
    report["paper_bet_rows"] = len(rows)
    report["overall"] = _summarize(rows)

    for bucket in ["3-5%", "5-8%", "8%+"]:
        report["edge_buckets"][bucket] = _summarize(
            [row for row in rows if row["edge_bucket"] == bucket]
        )

    for bucket in ["healthy", "degraded"]:
        report["data_health_buckets"][bucket] = _summarize(
            [row for row in rows if row["health_bucket"] == bucket]
        )

    overall = report["overall"]
    reasons: List[str] = []

    if len(rows) < 300:
        reasons.append("sample_count_below_300")
    if overall.get("avg_clv") is None or overall.get("avg_clv") <= 0:
        reasons.append("avg_clv_not_positive")
    if not overall.get("beaten_market_brier"):
        reasons.append("model_brier_not_better_than_market")
    if not overall.get("beaten_market_logloss"):
        reasons.append("model_logloss_not_better_than_market")

    high_edge = report["edge_buckets"].get("8%+", {})
    if high_edge.get("count", 0) > 0 and (
        high_edge.get("avg_clv") is None or high_edge.get("avg_clv") <= 0
    ):
        reasons.append("8pct_plus_edge_bucket_clv_not_positive")

    report["promote_reasons"] = reasons
    report["promote_ready"] = len(reasons) == 0

    if "avg_clv_not_positive" in reasons:
        report["recommendations"].append(
            "Average CLV is not positive; keep all live bet candidates blocked."
        )
    if "8pct_plus_edge_bucket_clv_not_positive" in reasons:
        report["recommendations"].append(
            "8%+ edge bucket has not proven positive CLV; continue large-edge guard."
        )
    if "model_brier_not_better_than_market" in reasons:
        report["recommendations"].append(
            "Model Brier is not better than market baseline; residual modeling should stay in shadow mode."
        )
    if "model_logloss_not_better_than_market" in reasons:
        report["recommendations"].append(
            "Model logloss is not better than market baseline; do not promote production model."
        )
    if len(rows) < 300:
        report["recommendations"].append(
            "Settled sample count below 300; use report for research only."
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    diagnostic = build_market_edge_research()
    print(
        json.dumps(
            {
                "generated_at": diagnostic["generated_at"],
                "settled_rows": diagnostic["settled_rows"],
                "overall": diagnostic["overall"],
                "promote_ready": diagnostic["promote_ready"],
                "promote_reasons": diagnostic["promote_reasons"],
                "report_written_to": "report/market_edge_research.json",
            },
            indent=2,
            ensure_ascii=True,
        )
    )
