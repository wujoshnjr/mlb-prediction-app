from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


REPORT_DIR = Path("report")
DATA_DIR = Path("data")

PREDICTION_JSON = REPORT_DIR / "prediction.json"
SNAPSHOTS_CSV = DATA_DIR / "prediction_snapshots.csv"
MARKET_ODDS_CSV = DATA_DIR / "market_odds_history.csv"
FINALIZED_CSV = DATA_DIR / "finalized_games.csv"
CLV_DIAGNOSTIC_JSON = REPORT_DIR / "evaluation_clv_diagnostic.json"

OUTPUT_EDGE = REPORT_DIR / "clv_by_edge_bucket.json"
OUTPUT_SIDE = REPORT_DIR / "clv_by_side.json"
OUTPUT_ODDS = REPORT_DIR / "clv_by_odds_range.json"
OUTPUT_LINEUP = REPORT_DIR / "clv_by_lineup_status.json"

MIN_SLICE_SAMPLE = 10
MIN_POSITIVE_CLV_RATE = 0.55


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        parsed = float(value)
        if not math.isfinite(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    parsed = _as_float(value)
    if parsed is None:
        return default
    return int(parsed)


def _safe_read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    status = {"path": str(path), "exists": path.exists(), "rows": None, "error": ""}
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


def _safe_read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    status = {"path": str(path), "exists": path.exists(), "rows": 0, "error": ""}
    if not path.exists():
        status["error"] = "file_missing"
        return None, status

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        status["error"] = str(exc)
        return None, status

    status["rows"] = int(len(frame))
    return frame, status


def _normalise_game_id(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "game_id" in frame.columns:
        frame["game_id"] = frame["game_id"].astype(str)
    return frame


def _bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        .fillna(False)
        .astype(bool)
    )


def _closing_moneyline_odds_by_game(
    odds: Optional[pd.DataFrame],
) -> pd.DataFrame:
    columns = ["game_id", "closing_home_odds", "closing_away_odds"]
    if odds is None or odds.empty:
        return pd.DataFrame(columns=columns)

    required = {"game_id", "market", "side", "odds"}
    if not required.issubset(odds.columns):
        return pd.DataFrame(columns=columns)

    frame = odds.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["market"] = frame["market"].astype(str).str.strip().str.lower()
    frame["side"] = frame["side"].astype(str).str.strip().str.lower()
    frame["odds"] = pd.to_numeric(frame["odds"], errors="coerce")

    frame = frame[
        (frame["market"] == "moneyline")
        & frame["side"].isin(["home", "away"])
        & frame["odds"].notna()
        & (frame["odds"] > 1.0)
    ].copy()

    if frame.empty:
        return pd.DataFrame(columns=columns)

    if "is_closing_snapshot" not in frame.columns:
        return pd.DataFrame(columns=columns)

    frame = frame[_bool_series(frame["is_closing_snapshot"])].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    aggregated = (
        frame.groupby(["game_id", "side"], as_index=False)["odds"]
        .mean()
        .copy()
    )

    if aggregated.empty:
        return pd.DataFrame(columns=columns)

    pivot = aggregated.pivot_table(
        index="game_id",
        columns="side",
        values="odds",
        aggfunc="mean",
    ).reset_index()

    if "home" not in pivot.columns:
        pivot["home"] = None
    if "away" not in pivot.columns:
        pivot["away"] = None

    pivot = pivot.rename(
        columns={
            "home": "closing_home_odds",
            "away": "closing_away_odds",
        }
    )

    return pivot[columns].copy()


def _merge_closing_odds(
    snapshots: pd.DataFrame,
    market_odds: Optional[pd.DataFrame],
) -> pd.DataFrame:
    closing = _closing_moneyline_odds_by_game(market_odds)
    if snapshots.empty or closing.empty or "game_id" not in snapshots.columns:
        return snapshots

    frame = snapshots.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    merged = frame.merge(
        closing,
        on="game_id",
        how="left",
        suffixes=("", "_from_market"),
    )

    for column in ("closing_home_odds", "closing_away_odds"):
        source = f"{column}_from_market"
        if source not in merged.columns:
            continue

        source_values = pd.to_numeric(merged[source], errors="coerce")

        if column in merged.columns:
            current_values = pd.to_numeric(merged[column], errors="coerce")
            merged[column] = current_values.combine_first(source_values)
        else:
            merged[column] = source_values

        merged.drop(columns=[source], inplace=True)

    return merged


def _selected_side(row: pd.Series) -> str:
    side = str(row.get("side", "") or "").strip().lower()
    if side in {"home", "away"}:
        return side

    edge = _as_float(row.get("model_edge_home"))
    if edge is not None:
        return "home" if edge >= 0 else "away"

    recommendation = str(row.get("moneyline_recommendation", "") or "").strip().lower()
    home_team = str(row.get("home_team", "") or "").strip().lower()
    away_team = str(row.get("away_team", "") or "").strip().lower()

    if home_team and home_team in recommendation:
        return "home"
    if away_team and away_team in recommendation:
        return "away"

    return "unknown"


def _entry_odds(row: pd.Series, side: str) -> Optional[float]:
    if side == "home":
        return _as_float(row.get("home_moneyline_odds"))
    if side == "away":
        return _as_float(row.get("away_moneyline_odds"))
    return _as_float(row.get("odds"))


def _closing_odds(row: pd.Series, side: str) -> Optional[float]:
    if side == "home":
        return _as_float(row.get("closing_home_odds"))
    if side == "away":
        return _as_float(row.get("closing_away_odds"))
    return None


def _compute_clv(row: pd.Series, side: str) -> Optional[float]:
    generic = _as_float(row.get("clv"))
    if generic is not None:
        return generic

    if side == "home":
        direct = _as_float(row.get("clv_home_moneyline"))
        if direct is not None:
            return direct

    if side == "away":
        direct = _as_float(row.get("clv_away_moneyline"))
        if direct is not None:
            return direct

    entry = _entry_odds(row, side)
    closing = _closing_odds(row, side)
    if entry is None or closing is None or entry <= 1.0 or closing <= 1.0:
        return None

    return float(math.log(entry) - math.log(closing))


def _edge_bucket(edge: Optional[float]) -> str:
    if edge is None:
        return "unknown"

    edge = abs(float(edge))
    if edge < 0.03:
        return "below_3pct"
    if edge < 0.05:
        return "3_to_5pct"
    if edge < 0.08:
        return "5_to_8pct"
    return "8pct_plus"


def _odds_range(odds: Optional[float]) -> str:
    if odds is None:
        return "unknown"
    if odds < 1.70:
        return "odds_below_1_70"
    if odds < 2.10:
        return "odds_1_70_to_2_10"
    if odds < 2.80:
        return "odds_2_10_to_2_80"
    return "odds_above_2_80"


def _lineup_status(row: pd.Series) -> str:
    for column in ("lineup_status", "home_projected_lineup_status", "away_projected_lineup_status"):
        value = str(row.get(column, "") or "").strip().lower()
        if value:
            if value == "confirmed" or "confirmed" in value:
                return "confirmed"
            if "projected" in value:
                return "projected"
            if "pending" in value:
                return "pending"

    available = _as_float(row.get("lineup_context_available"))
    if available is not None:
        return "confirmed" if available >= 1.0 else "pending"

    return "unknown"


def _prepare_clv_rows(
    snapshots: Optional[pd.DataFrame],
    finalized: Optional[pd.DataFrame],
    market_odds: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if snapshots is None or snapshots.empty or "game_id" not in snapshots.columns:
        return pd.DataFrame()

    frame = _normalise_game_id(snapshots)
    frame = _merge_closing_odds(frame, market_odds)
    
    if finalized is not None and not finalized.empty and "game_id" in finalized.columns:
        final = _normalise_game_id(finalized)
        needed = [col for col in ["game_id", "home_win"] if col in final.columns]
        if "home_win" in needed:
            frame = frame.merge(
                final[needed].drop_duplicates("game_id"),
                on="game_id",
                how="left",
                suffixes=("", "_final"),
            )
            if "home_win_final" in frame.columns and "home_win" in frame.columns:
                frame["home_win"] = frame["home_win"].combine_first(frame["home_win_final"])

    if "snapshot_created_at" in frame.columns:
        frame["_snapshot_dt"] = pd.to_datetime(
            frame["snapshot_created_at"],
            errors="coerce",
            utc=True,
        )
        frame = frame.sort_values("_snapshot_dt").groupby("game_id", as_index=False).tail(1)

    records: List[Dict[str, Any]] = []

    for _, row in frame.iterrows():
        side = _selected_side(row)
        if side not in {"home", "away"}:
            continue

        clv = _compute_clv(row, side)
        if clv is None:
            continue

        edge = _as_float(row.get("moneyline_selected_edge"))
        if edge is None:
            edge = _as_float(row.get("model_edge_home"))

        odds = _entry_odds(row, side)
        recommendation_status = str(row.get("recommendation_status", "") or "").lower()
        live_candidate = str(row.get("live_bet_candidate", "") or "").strip().lower() in {
            "true",
            "1",
            "yes",
        }

        records.append(
            {
                "game_id": str(row.get("game_id")),
                "clv": float(clv),
                "edge": abs(edge) if edge is not None else None,
                "edge_bucket": _edge_bucket(edge),
                "side": side,
                "odds_range": _odds_range(odds),
                "lineup_status": _lineup_status(row),
                "paper_bet": "paper" in recommendation_status,
                "live_bet_candidate": live_candidate,
            }
        )

    return pd.DataFrame(records)


def _summarise_slice(frame: pd.DataFrame, column: str) -> List[Dict[str, Any]]:
    if frame.empty or column not in frame.columns:
        return []

    output: List[Dict[str, Any]] = []

    for name, group in frame.groupby(column, dropna=False):
        clv_values = pd.to_numeric(group["clv"], errors="coerce").dropna()
        count = int(len(clv_values))
        if count == 0:
            continue

        positive_count = int((clv_values > 0).sum())
        negative_count = int((clv_values < 0).sum())
        positive_rate = positive_count / count if count else 0.0
        avg_clv = float(clv_values.mean())

        reasons: List[str] = []
        if count < MIN_SLICE_SAMPLE:
            reasons.append("insufficient_slice_sample")
        if avg_clv < 0:
            reasons.append("negative_avg_clv")
        if positive_rate <= MIN_POSITIVE_CLV_RATE:
            reasons.append("positive_clv_rate_below_threshold")

        output.append(
            {
                "slice": str(name),
                "count": count,
                "avg_clv": round(avg_clv, 6),
                "positive_clv_rate": round(positive_rate, 4),
                "positive_clv_count": positive_count,
                "negative_clv_count": negative_count,
                "paper_bet_count": int(group["paper_bet"].sum()) if "paper_bet" in group else 0,
                "live_bet_candidate_count": (
                    int(group["live_bet_candidate"].sum())
                    if "live_bet_candidate" in group
                    else 0
                ),
                "block_live_bet": bool(reasons),
                "block_reasons": reasons,
                "source": "per_pick_clv",
            }
        )

    return sorted(output, key=lambda item: item["slice"])
    

def _slice_from_aggregate_summary(
    summary: Dict[str, Any],
    *,
    source_name: str,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []

    for name, stats in summary.items():
        if not isinstance(stats, dict):
            continue

        count = _as_int(stats.get("settled_count"), _as_int(stats.get("count"), 0)) or 0
        avg_clv = _as_float(stats.get("avg_clv"))

        reasons: List[str] = []
        if count < MIN_SLICE_SAMPLE:
            reasons.append("insufficient_slice_sample")
        if avg_clv is not None and avg_clv < 0:
            reasons.append("negative_avg_clv")
        reasons.append("per_pick_positive_clv_unavailable")

        output.append(
            {
                "slice": str(name),
                "count": count,
                "avg_clv": round(avg_clv, 6) if avg_clv is not None else None,
                "positive_clv_rate": None,
                "positive_clv_count": None,
                "negative_clv_count": None,
                "paper_bet_count": count,
                "live_bet_candidate_count": 0,
                "block_live_bet": True,
                "block_reasons": reasons,
                "source": source_name,
            }
        )

    return sorted(output, key=lambda item: item["slice"])


def _fallback_slices_from_diagnostic(
    diagnostic: Optional[Dict[str, Any]],
    slice_type: str,
) -> List[Dict[str, Any]]:
    if not diagnostic:
        return []

    if slice_type == "edge_bucket":
        summary = diagnostic.get("edge_bucket_summary") or {}
        if isinstance(summary, dict):
            return _slice_from_aggregate_summary(
                summary,
                source_name="evaluation_clv_diagnostic.edge_bucket_summary",
            )

    if slice_type == "side":
        summary = diagnostic.get("side_summary") or {}
        if isinstance(summary, dict):
            return _slice_from_aggregate_summary(
                summary,
                source_name="evaluation_clv_diagnostic.side_summary",
            )

    return []


def _write_report(
    path: Path,
    *,
    generated_at: str,
    status: str,
    input_files: Dict[str, Any],
    slice_type: str,
    slices: List[Dict[str, Any]],
    recommendations: List[str],
) -> None:
    report = {
        "generated_at": generated_at,
        "status": status,
        "input_files": input_files,
        "slice_type": slice_type,
        "slices": slices,
        "recommendations": recommendations,
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")


def build_clv_slice_reports() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = _utc_now()

    _, prediction_status = _safe_read_json(PREDICTION_JSON)
    snapshots, snapshots_status = _safe_read_csv(SNAPSHOTS_CSV)
    market_odds, market_status = _safe_read_csv(MARKET_ODDS_CSV)
    finalized, finalized_status = _safe_read_csv(FINALIZED_CSV)
    clv_diagnostic, clv_diag_status = _safe_read_json(CLV_DIAGNOSTIC_JSON)

    input_files = {
        "prediction": prediction_status,
        "prediction_snapshots": snapshots_status,
        "market_odds_history": market_status,
        "finalized_games": finalized_status,
        "evaluation_clv_diagnostic": clv_diag_status,
    }

    clv_rows = _prepare_clv_rows(snapshots, finalized, market_odds)

    reports = {
        OUTPUT_EDGE: ("edge_bucket", "edge_bucket"),
        OUTPUT_SIDE: ("side", "side"),
        OUTPUT_ODDS: ("odds_range", "odds_range"),
        OUTPUT_LINEUP: ("lineup_status", "lineup_status"),
    }

    status_by_report: Dict[str, str] = {}
    row_count_by_report: Dict[str, int] = {}
    
    for path, (column, slice_type) in reports.items():
        if not clv_rows.empty:
            slices = _summarise_slice(clv_rows, column)
            report_status = "ok" if slices else "insufficient_samples"
            report_recommendations = (
                []
                if slices
                else [f"No per-pick CLV rows available for slice type '{slice_type}'."]
            )
        else:
            slices = _fallback_slices_from_diagnostic(clv_diagnostic, slice_type)
            if slices:
                report_status = "partial"
                report_recommendations = [
                    "Using aggregate CLV diagnostic fallback because per-pick CLV rows are not available.",
                    "Live betting remains blocked because positive CLV rate cannot be validated from aggregate fallback.",
                ]
            else:
                report_status = "insufficient_samples"
                report_recommendations = [
                    f"No per-pick or aggregate CLV data available for slice type '{slice_type}'."
                ]

        _write_report(
            path,
            generated_at=generated_at,
            status=report_status,
            input_files=input_files,
            slice_type=slice_type,
            slices=slices,
            recommendations=report_recommendations,
        )

        status_by_report[str(path)] = report_status
        row_count_by_report[str(path)] = len(slices)

    overall_status = (
        "ok"
        if any(status == "ok" for status in status_by_report.values())
        else (
            "partial"
            if any(status == "partial" for status in status_by_report.values())
            else "insufficient_samples"
        )
    )

    return {
        "generated_at": generated_at,
        "status": overall_status,
        "evaluated_clv_rows": int(len(clv_rows)),
        "outputs": status_by_report,
        "slice_counts": row_count_by_report,
    }


def main() -> None:
    summary = build_clv_slice_reports()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
