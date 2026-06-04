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


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


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


def _log_clv(entry_odds: Optional[float], closing_odds: Optional[float]) -> Optional[float]:
    if entry_odds is None or closing_odds is None:
        return None
    if entry_odds <= 1 or closing_odds <= 1:
        return None
    return float(math.log(entry_odds) - math.log(closing_odds))


def _normalise_side(value: Any) -> str:
    text = _safe_str(value).lower()
    if text in {"home", "h"}:
        return "home"
    if text in {"away", "a"}:
        return "away"
    return ""


def _selected_side(row: pd.Series) -> str:
    edge = _safe_float(row.get("model_edge_home"))
    recommendation = _safe_str(row.get("moneyline_recommendation")).lower()
    home_team = _safe_str(row.get("home_team")).lower()
    away_team = _safe_str(row.get("away_team")).lower()

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


def _snapshot_closing_odds(row: pd.Series, side: str) -> Optional[float]:
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
    return bool(statcast_zero or top3_zero)


def _latest_closing_odds_lookup(odds_frame: Optional[pd.DataFrame]) -> Dict[Tuple[str, str], float]:
    """
    Build closing odds lookup by (game_id, side).

    Supports two common market_odds_history shapes:
    1. long format: game_id, market, side, odds, is_closing_snapshot
    2. wide format: game_id, home_moneyline_odds, away_moneyline_odds, is_closing_snapshot
    """
    lookup: Dict[Tuple[str, str], float] = {}

    if odds_frame is None or odds_frame.empty or "game_id" not in odds_frame.columns:
        return lookup

    frame = odds_frame.copy()
    frame["game_id"] = frame["game_id"].astype(str)

    if "market" in frame.columns:
        frame["market_normalized"] = frame["market"].astype(str).str.lower()
        frame = frame[frame["market_normalized"].str.contains("moneyline", na=False)]

    if "is_closing_snapshot" in frame.columns:
        closing_mask = frame["is_closing_snapshot"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
        if closing_mask.any():
            frame = frame[closing_mask].copy()

    sort_columns = []
    for column in ("captured_at", "timestamp", "created_at", "last_update"):
        if column in frame.columns:
            parsed = f"{column}_parsed"
            frame[parsed] = pd.to_datetime(frame[column], errors="coerce", utc=True)
            sort_columns.append(parsed)

    if sort_columns:
        frame = frame.sort_values(sort_columns[-1])

    # Long format
    if {"side", "odds"}.issubset(set(frame.columns)):
        for _, row in frame.iterrows():
            game_id = _safe_str(row.get("game_id"))
            side = _normalise_side(row.get("side"))
            odds = _safe_float(row.get("odds"))
            if game_id and side in {"home", "away"} and odds is not None:
                lookup[(game_id, side)] = odds

    # Wide format
    for _, row in frame.iterrows():
        game_id = _safe_str(row.get("game_id"))
        if not game_id:
            continue

        home_odds = (
            _safe_float(row.get("home_moneyline_odds"))
            or _safe_float(row.get("home_odds"))
            or _safe_float(row.get("closing_home_odds"))
        )
        away_odds = (
            _safe_float(row.get("away_moneyline_odds"))
            or _safe_float(row.get("away_odds"))
            or _safe_float(row.get("closing_away_odds"))
        )

        if home_odds is not None:
            lookup[(game_id, "home")] = home_odds
        if away_odds is not None:
            lookup[(game_id, "away")] = away_odds

    return lookup


def _resolve_closing_odds(
    *,
    row: pd.Series,
    side: str,
    odds_lookup: Dict[Tuple[str, str], float],
) -> Tuple[Optional[float], str]:
    snapshot_close = _snapshot_closing_odds(row, side)
    if snapshot_close is not None:
        return snapshot_close, "snapshot_closing_odds"

    game_id = _safe_str(row.get("game_id"))
    lookup_close = odds_lookup.get((game_id, side))
    if lookup_close is not None:
        return lookup_close, "market_odds_history_closing"

    # Last-resort fallback. This does NOT prove CLV, but prevents null-only reports.
    entry = _entry_odds(row, side)
    if entry is not None:
        return entry, "entry_odds_fallback"

    return None, "missing"


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "win_rate": None,
            "avg_clv_decimal": None,
            "avg_clv_log": None,
            "clv_available_count": 0,
            "clv_missing_count": 0,
            "clv_source_counts": {},
            "model_brier": None,
            "market_brier": None,
            "model_logloss": None,
            "market_logloss": None,
            "beaten_market_brier": False,
            "beaten_market_logloss": False,
        }

    wins = [r["won"] for r in rows if r.get("won") in (0, 1)]
    clv_decimal = [r["clv_decimal"] for r in rows if r.get("clv_decimal") is not None]
    clv_log = [r["clv_log"] for r in rows if r.get("clv_log") is not None]
    model_briers = [r["model_brier"] for r in rows if r.get("model_brier") is not None]
    market_briers = [r["market_brier"] for r in rows if r.get("market_brier") is not None]
    model_ll = [r["model_logloss"] for r in rows if r.get("model_logloss") is not None]
    market_ll = [r["market_logloss"] for r in rows if r.get("market_logloss") is not None]

    source_counts: Dict[str, int] = {}
    for row in rows:
        source = str(row.get("clv_source", "missing"))
        source_counts[source] = source_counts.get(source, 0) + 1

    def avg(values: List[float]) -> Optional[float]:
        return float(sum(values) / len(values)) if values else None

    mb = avg(model_briers)
    xb = avg(market_briers)
    ml = avg(model_ll)
    xl = avg(market_ll)

    return {
        "count": len(rows),
        "win_rate": avg(wins),
        "avg_clv_decimal": avg(clv_decimal),
        "avg_clv_log": avg(clv_log),
        "clv_available_count": len(clv_decimal),
        "clv_missing_count": len(rows) - len(clv_decimal),
        "clv_source_counts": source_counts,
        "model_brier": mb,
        "market_brier": xb,
        "model_logloss": ml,
        "market_logloss": xl,
        "beaten_market_brier": bool(mb is not None and xb is not None and mb < xb),
        "beaten_market_logloss": bool(ml is not None and xl is not None and ml < xl),
    }


def build_market_edge_research(
    snapshots_path: str = "data/prediction_snapshots.csv",
    odds_history_path: str = "data/market_odds_history.csv",
    output_path: str = "report/market_edge_research.json",
) -> Dict[str, Any]:
    frame, snapshot_error = _safe_read_csv(Path(snapshots_path))
    odds_frame, odds_error = _safe_read_csv(Path(odds_history_path))
    odds_lookup = _latest_closing_odds_lookup(odds_frame)

    report: Dict[str, Any] = {
        "generated_at": _current_utc_iso(),
        "input_files": {
            "snapshots": {
                "path": snapshots_path,
                "exists": frame is not None,
                "error": snapshot_error,
                "rows": int(len(frame)) if frame is not None else 0,
            },
            "market_odds_history": {
                "path": odds_history_path,
                "exists": odds_frame is not None,
                "error": odds_error,
                "rows": int(len(odds_frame)) if odds_frame is not None else 0,
                "closing_lookup_size": len(odds_lookup),
            },
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
        if _safe_str(row.get("recommendation_status")).upper() != "PAPER_BET":
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
        closing, clv_source = _resolve_closing_odds(
            row=row,
            side=side,
            odds_lookup=odds_lookup,
        )

        completed = {
            "game_id": _safe_str(row.get("game_id")),
            "side": side,
            "won": outcome,
            "edge_abs": edge_abs,
            "edge_bucket": _edge_bucket(edge_abs),
            "entry_odds": entry,
            "closing_odds": closing,
            "clv_source": clv_source,
            "clv_decimal": _decimal_clv(entry, closing),
            "clv_log": _log_clv(entry, closing),
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

    avg_clv_log = overall.get("avg_clv_log")
    clv_available_count = int(overall.get("clv_available_count", 0) or 0)
    source_counts = overall.get("clv_source_counts", {})
    fallback_count = int(source_counts.get("entry_odds_fallback", 0)) if isinstance(source_counts, dict) else 0

    if len(rows) < 300:
        reasons.append("sample_count_below_300")
    if clv_available_count == 0:
        reasons.append("clv_unavailable")
    if fallback_count > 0:
        reasons.append("clv_contains_entry_odds_fallback")
    if avg_clv_log is None or avg_clv_log <= 0:
        reasons.append("avg_clv_not_positive")
    if not overall.get("beaten_market_brier"):
        reasons.append("model_brier_not_better_than_market")
    if not overall.get("beaten_market_logloss"):
        reasons.append("model_logloss_not_better_than_market")

    high_edge = report["edge_buckets"].get("8%+", {})
    high_edge_clv = high_edge.get("avg_clv_log")
    if high_edge.get("count", 0) > 0 and (
        high_edge_clv is None or high_edge_clv <= 0
    ):
        reasons.append("8pct_plus_edge_bucket_clv_not_positive")

    report["promote_reasons"] = reasons
    report["promote_ready"] = len(reasons) == 0

    if "clv_unavailable" in reasons:
        report["recommendations"].append(
            "CLV is unavailable; verify closing odds in market_odds_history.csv."
        )
    if "clv_contains_entry_odds_fallback" in reasons:
        report["recommendations"].append(
            "Some CLV values use entry odds fallback; treat CLV as incomplete until true closing odds are collected."
        )
    if "avg_clv_not_positive" in reasons:
        report["recommendations"].append(
            "Average log CLV is not positive; keep all live bet candidates blocked."
        )
    if "8pct_plus_edge_bucket_clv_not_positive" in reasons:
        report["recommendations"].append(
            "8%+ edge bucket has not proven positive CLV; continue large-edge guard."
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
