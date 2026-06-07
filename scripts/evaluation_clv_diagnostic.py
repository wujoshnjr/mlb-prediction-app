from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


EDGE_BUCKETS = ["below_threshold", "3_to_5pct", "5_to_8pct", "8pct_plus"]


# ---------------------------------------------------------------------------
# Safe helpers
# ---------------------------------------------------------------------------

def _current_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        if not isinstance(data, dict):
            return None, f"File {path} does not contain a JSON object"
        return data, ""
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON in {path}: {exc}"
    except Exception as exc:
        return None, f"Error reading {path}: {exc}"


def _safe_read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        frame = pd.read_csv(path)
        return frame, ""
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except Exception as exc:
        return None, f"Error reading CSV {path}: {exc}"


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return int(numeric)


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


def _safe_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n", "", "nan", "none", "null"}:
            return False
    if isinstance(value, (int, float)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if value == 1:
            return True
        if value == 0:
            return False
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if value is None or isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        item = value.item()
        return _json_safe(item)
    except Exception:
        return str(value)


def _extract_predictions(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("today_predictions", "predictions", "games", "recommendations"):
        value = report.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _parse_flags(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item)]
        except Exception:
            pass
        return [
            part.strip()
            for part in text.replace(";", ",").split(",")
            if part.strip()
        ]
    return []


def _edge_bucket(edge: Any) -> str:
    edge_value = _safe_float(edge)
    if edge_value is None:
        return "below_threshold"

    abs_edge = abs(edge_value)
    if abs_edge >= 0.08:
        return "8pct_plus"
    if abs_edge >= 0.05:
        return "5_to_8pct"
    if abs_edge >= 0.03:
        return "3_to_5pct"
    return "below_threshold"


def _decimal_profit(odds: Optional[float], won: Optional[int]) -> Optional[float]:
    if odds is None or won is None:
        return None
    if odds <= 1:
        return None
    return (odds - 1.0) if int(won) == 1 else -1.0


def _mean_or_none(values: List[float]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _rate_or_none(values: List[int]) -> Optional[float]:
    clean = [int(value) for value in values if value in (0, 1)]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _logloss_from_probs(probs: List[float], outcomes: List[int]) -> Optional[float]:
    if not probs or len(probs) != len(outcomes):
        return None

    eps = 1e-15
    losses = []
    for prob, outcome in zip(probs, outcomes):
        if prob is None or outcome not in (0, 1):
            continue
        p = min(max(float(prob), eps), 1.0 - eps)
        y = int(outcome)
        losses.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))

    return _mean_or_none(losses)


def _brier_from_probs(probs: List[float], outcomes: List[int]) -> Optional[float]:
    if not probs or len(probs) != len(outcomes):
        return None

    values = []
    for prob, outcome in zip(probs, outcomes):
        if prob is None or outcome not in (0, 1):
            continue
        values.append((float(prob) - int(outcome)) ** 2)

    return _mean_or_none(values)


# ---------------------------------------------------------------------------
# Prediction row extraction
# ---------------------------------------------------------------------------

def _selected_side_from_snapshot(row: pd.Series) -> str:
    explicit = _safe_str(row.get("moneyline_selected_side")).lower()
    if explicit in {"home", "away"}:
        return explicit

    home_kelly = _safe_float(row.get("home_kelly_fraction")) or 0.0
    away_kelly = _safe_float(row.get("away_kelly_fraction")) or 0.0

    if home_kelly > 0 and home_kelly >= away_kelly:
        return "home"
    if away_kelly > 0 and away_kelly > home_kelly:
        return "away"

    model_edge_home = _safe_float(row.get("model_edge_home"))
    if model_edge_home is not None:
        if model_edge_home >= 0.03:
            return "home"
        if model_edge_home <= -0.03:
            return "away"

    recommendation = _safe_str(row.get("moneyline_recommendation")).lower()
    home_team = _safe_str(row.get("home_team")).lower()
    away_team = _safe_str(row.get("away_team")).lower()

    if home_team and home_team in recommendation:
        return "home"
    if away_team and away_team in recommendation:
        return "away"

    return ""


def _selected_edge_from_snapshot(row: pd.Series, side: str) -> Optional[float]:
    explicit = _safe_float(row.get("moneyline_selected_edge"))
    if explicit is not None:
        return explicit

    model_edge_home = _safe_float(row.get("model_edge_home"))
    if model_edge_home is None:
        return None

    if side == "home":
        return model_edge_home
    if side == "away":
        return -model_edge_home

    return None


def _selected_odds_from_snapshot(row: pd.Series, side: str) -> Optional[float]:
    if side == "home":
        return _safe_float(row.get("home_moneyline_odds"))
    if side == "away":
        return _safe_float(row.get("away_moneyline_odds"))
    return None


def _selected_closing_odds_from_snapshot(row: pd.Series, side: str) -> Optional[float]:
    if side == "home":
        return _safe_float(row.get("closing_home_odds"))
    if side == "away":
        return _safe_float(row.get("closing_away_odds"))
    return None


def _selected_clv(entry_odds: Optional[float], closing_odds: Optional[float]) -> Optional[float]:
    if entry_odds is None or closing_odds is None:
        return None

    if entry_odds <= 1.0 or closing_odds <= 1.0:
        return None

    return float(math.log(entry_odds) - math.log(closing_odds))


def _selected_probability_from_snapshot(row: pd.Series, side: str) -> Optional[float]:
    home_prob = (
        _safe_float(row.get("displayed_home_win_pct"))
        or _safe_float(row.get("predicted_home_win_pct"))
        or _safe_float(row.get("premarket_model_home_prob"))
    )

    if home_prob is None:
        return None

    if side == "home":
        return home_prob
    if side == "away":
        return 1.0 - home_prob

    return None


def _is_paper_bet_snapshot(row: pd.Series, side: str, edge: Optional[float]) -> bool:
    status = _safe_str(row.get("recommendation_status")).upper()

    if status != "PAPER_BET":
        return False

    if side not in {"home", "away"}:
        return False

    if edge is None or edge < 0.03:
        return False

    return True


def _current_prediction_rows(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []

    for item in predictions:
        side = _safe_str(item.get("moneyline_selected_side")).lower()
        if side not in {"home", "away"}:
            side = ""

        entry_odds = (
            _safe_float(item.get("home_moneyline_odds"))
            if side == "home"
            else _safe_float(item.get("away_moneyline_odds"))
            if side == "away"
            else None
        )

        selected_prob = None
        home_prob = (
            _safe_float(item.get("displayed_home_win_pct"))
            or _safe_float(item.get("predicted_home_win_pct"))
            or _safe_float(item.get("premarket_model_home_prob"))
        )
        if home_prob is not None:
            if side == "home":
                selected_prob = home_prob
            elif side == "away":
                selected_prob = 1.0 - home_prob

        readiness = item.get("betting_readiness") or {}
        flags = item.get("betting_risk_flags")
        if flags is None:
            flags = readiness.get("betting_risk_flags")

        rows.append(
            {
                "source": "current_prediction",
                "game_id": _safe_str(item.get("game_id")),
                "game_date": _safe_str(item.get("game_date")),
                "home_team": _safe_str(item.get("home_team")),
                "away_team": _safe_str(item.get("away_team")),
                "recommendation_status": _safe_str(item.get("recommendation_status")),
                "moneyline_selected_side": side,
                "moneyline_selected_edge": _safe_float(
                    item.get("moneyline_selected_edge")
                ),
                "edge_bucket": _safe_str(item.get("edge_bucket"))
                or _edge_bucket(item.get("moneyline_selected_edge")),
                "entry_odds": entry_odds,
                "selected_model_probability": selected_prob,
                "model_source": _safe_str(item.get("model_source")),
                "odds_quality_status": _safe_str(item.get("odds_quality_status")),
                "betting_readiness_status": _safe_str(
                    item.get("betting_readiness_status")
                    or readiness.get("betting_readiness_status")
                ),
                "betting_readiness_score": _safe_float(
                    item.get("betting_readiness_score")
                    or readiness.get("betting_readiness_score")
                ),
                "stake_multiplier": _safe_float(
                    item.get("stake_multiplier") or readiness.get("stake_multiplier")
                ),
                "live_bet_candidate": item.get("live_bet_candidate") is True,
                "betting_risk_flags": _parse_flags(flags),
            }
        )

    return rows


def _snapshot_rows(
    snapshots: pd.DataFrame,
    finalized: Optional[pd.DataFrame],
) -> List[Dict[str, Any]]:
    if snapshots is None or snapshots.empty:
        return []

    frame = snapshots.copy()

    if "game_id" in frame.columns:
        frame["game_id"] = frame["game_id"].astype(str)

    if "home_win" not in frame.columns and finalized is not None and not finalized.empty:
        if "game_id" in finalized.columns and "home_win" in finalized.columns:
            final_frame = finalized.copy()
            final_frame["game_id"] = final_frame["game_id"].astype(str)
            frame = frame.merge(
                final_frame[["game_id", "home_win"]],
                on="game_id",
                how="left",
            )

    rows = []

    for _, row in frame.iterrows():
        side = _selected_side_from_snapshot(row)
        edge = _selected_edge_from_snapshot(row, side)
        entry_odds = _selected_odds_from_snapshot(row, side)
        closing_odds = _selected_closing_odds_from_snapshot(row, side)
        selected_prob = _selected_probability_from_snapshot(row, side)
        home_win = _safe_int(row.get("home_win"))

        won = None
        if home_win in (0, 1) and side in {"home", "away"}:
            won = 1 if (side == "home" and home_win == 1) or (side == "away" and home_win == 0) else 0

        paper_bet = _is_paper_bet_snapshot(row, side, edge)

        if not paper_bet and won is None:
            continue

        home_kelly = _safe_float(row.get("home_kelly_fraction")) or 0.0
        away_kelly = _safe_float(row.get("away_kelly_fraction")) or 0.0
        stake_multiplier = _safe_float(row.get("stake_multiplier"))
        if stake_multiplier is None:
            stake_multiplier = max(home_kelly, away_kelly)

        flags = _parse_flags(row.get("betting_risk_flags"))
        if not flags and _safe_str(row.get("recommendation_status")).upper() != "PAPER_BET":
            flags = ["tracking_only"]

        rows.append(
            {
                "source": "snapshot",
                "snapshot_created_at": _safe_str(row.get("snapshot_created_at")),
                "game_id": _safe_str(row.get("game_id")),
                "game_date": _safe_str(row.get("game_date")),
                "home_team": _safe_str(row.get("home_team")),
                "away_team": _safe_str(row.get("away_team")),
                "recommendation_status": _safe_str(row.get("recommendation_status")),
                "moneyline_recommendation": _safe_str(
                    row.get("moneyline_recommendation")
                ),
                "moneyline_selected_side": side,
                "moneyline_selected_edge": edge,
                "edge_bucket": _edge_bucket(edge),
                "entry_odds": entry_odds,
                "closing_odds": closing_odds,
                "clv": _selected_clv(entry_odds, closing_odds),
                "selected_model_probability": selected_prob,
                "home_model_probability": (
                    _safe_float(row.get("premarket_model_home_prob"))
                    or _safe_float(row.get("predicted_home_win_pct"))
                    or _safe_float(row.get("displayed_home_win_pct"))
                ),
                "home_win": home_win,
                "won": won,
                "profit_flat": _decimal_profit(entry_odds, won),
                "stake_multiplier": stake_multiplier,
                "live_bet_candidate": row.get("live_bet_candidate") is True,
                "betting_readiness_status": _safe_str(
                    row.get("betting_readiness_status")
                ),
                "betting_risk_flags": flags,
                "model_source": _safe_str(row.get("model_source")),
                "odds_quality_status": _safe_str(row.get("odds_quality_status")),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _summarize_group(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "count": 0,
            "settled_count": 0,
            "win_rate": None,
            "roi_flat": None,
            "stake_weighted_roi": None,
            "avg_edge": None,
            "avg_clv": None,
            "positive_clv_rate": None,
        }

    settled = frame[frame["won"].notna()].copy() if "won" in frame.columns else pd.DataFrame()

    profits = [
        _safe_float(value)
        for value in settled.get("profit_flat", pd.Series(dtype=float)).tolist()
    ]
    profits = [value for value in profits if value is not None]

    stake_profit = []
    stake_sum = 0.0

    for _, row in settled.iterrows():
        profit = _safe_float(row.get("profit_flat"))
        stake = _safe_float(row.get("stake_multiplier"))
        if profit is None:
            continue
        if stake is None or stake <= 0:
            stake = 1.0
        stake_profit.append(profit * stake)
        stake_sum += stake

    clv_values = [
        _safe_float(value)
        for value in frame.get("clv", pd.Series(dtype=float)).tolist()
    ]
    clv_values = [value for value in clv_values if value is not None]

    edge_values = [
        _safe_float(value)
        for value in frame.get("moneyline_selected_edge", pd.Series(dtype=float)).tolist()
    ]
    edge_values = [value for value in edge_values if value is not None]

    won_values = (
        [int(value) for value in settled["won"].dropna().tolist()]
        if not settled.empty and "won" in settled.columns
        else []
    )

    return {
        "count": int(len(frame)),
        "settled_count": int(len(settled)),
        "win_rate": _rate_or_none(won_values),
        "roi_flat": _mean_or_none(profits),
        "stake_weighted_roi": (
            float(sum(stake_profit) / stake_sum)
            if stake_sum > 0
            else None
        ),
        "avg_edge": _mean_or_none(edge_values),
        "avg_clv": _mean_or_none(clv_values),
        "positive_clv_rate": (
            float(sum(1 for value in clv_values if value > 0) / len(clv_values))
            if clv_values
            else None
        ),
    }
    

def _bucket_summary(frame: pd.DataFrame, bucket_column: str) -> Dict[str, Any]:
    if frame is None or frame.empty or bucket_column not in frame.columns:
        return {}

    output: Dict[str, Any] = {}
    for bucket, group in frame.groupby(bucket_column):
        output[str(bucket)] = _summarize_group(group)
    return output


def _risk_flag_summary(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame is None or frame.empty:
        return {}

    rows = []
    for _, row in frame.iterrows():
        flags = row.get("betting_risk_flags")
        if not isinstance(flags, list) or not flags:
            flags = ["none"]

        for flag in flags:
            completed = row.to_dict()
            completed["risk_flag"] = str(flag)
            rows.append(completed)

    if not rows:
        return {}

    risk_frame = pd.DataFrame(rows)
    return _bucket_summary(risk_frame, "risk_flag")


def _window_summary(frame: pd.DataFrame, days: int) -> Dict[str, Any]:
    if frame is None or frame.empty or "game_date" not in frame.columns:
        return _summarize_group(pd.DataFrame())

    completed = frame.copy()
    completed["game_date_dt"] = pd.to_datetime(
        completed["game_date"],
        errors="coerce",
        utc=True,
    )

    if completed["game_date_dt"].notna().sum() == 0:
        return _summarize_group(pd.DataFrame())

    max_date = completed["game_date_dt"].max()
    cutoff = max_date - pd.Timedelta(days=days)
    window = completed[completed["game_date_dt"] >= cutoff].copy()
    return _summarize_group(window)


def _closing_odds_lookup(odds_frame: Optional[pd.DataFrame]) -> Dict[Tuple[str, str], float]:
    if odds_frame is None or odds_frame.empty:
        return {}

    required = {"game_id", "market", "side", "odds"}
    if not required.issubset(set(odds_frame.columns)):
        return {}

    frame = odds_frame.copy()
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
        return {}

    if "is_closing_snapshot" not in frame.columns:
        return {}

    closing_mask = frame["is_closing_snapshot"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    frame = frame[closing_mask].copy()

    if frame.empty:
        return {}

    aggregated = (
        frame.groupby(["game_id", "side"], as_index=False)["odds"]
        .mean()
        .copy()
    )

    lookup: Dict[Tuple[str, str], float] = {}
    for _, row in aggregated.iterrows():
        game_id = _safe_str(row.get("game_id"))
        side = _safe_str(row.get("side")).lower()
        odds = _safe_float(row.get("odds"))
        if game_id and side in {"home", "away"} and odds is not None:
            lookup[(game_id, side)] = odds

    return lookup


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def build_evaluation_clv_diagnostic(
    prediction_path: str = "report/prediction.json",
    snapshots_path: str = "data/prediction_snapshots.csv",
    odds_path: str = "data/market_odds_history.csv",
    finalized_path: str = "data/finalized_games.csv",
    output_path: str = "report/evaluation_clv_diagnostic.json",
) -> Dict[str, Any]:
    generated_at = _current_utc_iso()

    prediction_json, prediction_error = _safe_read_json(Path(prediction_path))
    snapshots_frame, snapshots_error = _safe_read_csv(Path(snapshots_path))
    odds_frame, odds_error = _safe_read_csv(Path(odds_path))
    finalized_frame, finalized_error = _safe_read_csv(Path(finalized_path))

    predictions = _extract_predictions(prediction_json) if prediction_json else []
    current_rows = _current_prediction_rows(predictions)

    snapshot_rows = _snapshot_rows(snapshots_frame, finalized_frame)
    historical = pd.DataFrame(snapshot_rows)

    closing_lookup = _closing_odds_lookup(odds_frame)

    if not historical.empty:
        for idx, row in historical.iterrows():
            if _safe_float(row.get("closing_odds")) is not None:
                continue
            side = _safe_str(row.get("moneyline_selected_side")).lower()
            gid = _safe_str(row.get("game_id"))
            closing_odds = closing_lookup.get((gid, side))
            entry_odds = _safe_float(row.get("entry_odds"))
            if closing_odds is not None:
                historical.at[idx, "closing_odds"] = closing_odds
                if entry_odds is not None:
                    historical.at[idx, "clv"] = _selected_clv(entry_odds, closing_odds)

    current = pd.DataFrame(current_rows)

    paper_bets = (
        historical[
            (historical["recommendation_status"].astype(str).str.upper() == "PAPER_BET")
            & (historical["moneyline_selected_side"].isin(["home", "away"]))
            & (historical["entry_odds"].notna())
        ].copy()
        if not historical.empty
        else pd.DataFrame()
    )

    live_candidates = (
        historical[historical["live_bet_candidate"] == True].copy()  # noqa: E712
        if not historical.empty and "live_bet_candidate" in historical.columns
        else pd.DataFrame()
    )

    settled = (
        historical[historical["won"].notna()].copy()
        if not historical.empty and "won" in historical.columns
        else pd.DataFrame()
    )

    paper_settled = (
        paper_bets[paper_bets["won"].notna()].copy()
        if not paper_bets.empty and "won" in paper_bets.columns
        else pd.DataFrame()
    )

    live_settled = (
        live_candidates[live_candidates["won"].notna()].copy()
        if not live_candidates.empty and "won" in live_candidates.columns
        else pd.DataFrame()
    )

    home_probs = []
    home_outcomes = []

    if not historical.empty:
        for _, row in historical.iterrows():
            prob = _safe_float(row.get("home_model_probability"))
            outcome = _safe_int(row.get("home_win"))
            if prob is not None and outcome in (0, 1):
                home_probs.append(prob)
                home_outcomes.append(outcome)

    current_status_counts = {}
    current_risk_counts = {}
    if not current.empty:
        for status in current.get("betting_readiness_status", pd.Series(dtype=str)).tolist():
            key = _safe_str(status) or "unknown"
            current_status_counts[key] = current_status_counts.get(key, 0) + 1

        for flags in current.get("betting_risk_flags", pd.Series(dtype=object)).tolist():
            parsed = flags if isinstance(flags, list) else []
            if not parsed:
                current_risk_counts["none"] = current_risk_counts.get("none", 0) + 1
            for flag in parsed:
                key = str(flag)
                current_risk_counts[key] = current_risk_counts.get(key, 0) + 1

    clv_values = []
    if not paper_bets.empty and "clv" in paper_bets.columns:
        clv_values = [
            value
            for value in paper_bets["clv"].apply(_safe_float).tolist()
            if value is not None
        ]

    clv_by_side = {}
    if not paper_bets.empty and "clv" in paper_bets.columns:
        for side, group in paper_bets.groupby("moneyline_selected_side"):
            values = [
                value
                for value in group["clv"].apply(_safe_float).tolist()
                if value is not None
            ]
            clv_by_side[str(side)] = _mean_or_none(values)

    report: Dict[str, Any] = {
        "generated_at": generated_at,
        "input_files": {
            "prediction": {
                "exists": prediction_json is not None,
                "error": prediction_error,
            },
            "snapshots": {
                "exists": snapshots_frame is not None,
                "error": snapshots_error,
                "rows": int(len(snapshots_frame)) if snapshots_frame is not None else 0,
            },
            "market_odds_history": {
                "exists": odds_frame is not None,
                "error": odds_error,
                "rows": int(len(odds_frame)) if odds_frame is not None else 0,
            },
            "finalized_games": {
                "exists": finalized_frame is not None,
                "error": finalized_error,
                "rows": int(len(finalized_frame)) if finalized_frame is not None else 0,
            },
        },
        "current_prediction_summary": {
            "prediction_count": int(len(current)),
            "paper_bet_count": int(
                (current["recommendation_status"].astype(str).str.upper() == "PAPER_BET").sum()
            )
            if not current.empty
            else 0,
            "live_bet_candidate_count": int(current["live_bet_candidate"].sum())
            if not current.empty and "live_bet_candidate" in current.columns
            else 0,
            "betting_readiness_status_counts": current_status_counts,
            "risk_flag_counts": current_risk_counts,
        },
        "settled_performance": {
            "settled_rows": int(len(settled)),
            "paper_bet_count": int(len(paper_settled)),
            "live_bet_candidate_count": int(len(live_settled)),
            "paper_bet_win_rate": _summarize_group(paper_settled)["win_rate"],
            "live_bet_candidate_win_rate": _summarize_group(live_settled)["win_rate"],
            "paper_bet_roi_flat": _summarize_group(paper_settled)["roi_flat"],
            "live_bet_roi_flat": _summarize_group(live_settled)["roi_flat"],
            "stake_weighted_roi": _summarize_group(paper_settled)["stake_weighted_roi"],
            "brier": _brier_from_probs(home_probs, home_outcomes),
            "logloss": _logloss_from_probs(home_probs, home_outcomes),
            "window_7d": _window_summary(paper_settled, 7),
            "window_14d": _window_summary(paper_settled, 14),
            "window_30d": _window_summary(paper_settled, 30),
        },
        "clv_summary": {
            "source": "per_pick_log_decimal_clv",
            "evaluated_picks": int(len(clv_values)),
            "positive_clv_count": int(sum(1 for value in clv_values if value > 0)),
            "negative_clv_count": int(sum(1 for value in clv_values if value < 0)),
            "neutral_clv_count": int(sum(1 for value in clv_values if value == 0)),
            "avg_clv": _mean_or_none(clv_values),
            "positive_clv_rate": (
                float(sum(1 for value in clv_values if value > 0) / len(clv_values))
                if clv_values
                else None
            ),
            "avg_clv_by_side": clv_by_side,
        },
        "edge_bucket_summary": _bucket_summary(paper_bets, "edge_bucket"),
        "risk_flag_summary": _risk_flag_summary(paper_bets),
        "side_summary": _bucket_summary(paper_bets, "moneyline_selected_side"),
        "recommendations": [],
    }

    recommendations: List[str] = []

    settled_count = int(report["settled_performance"]["settled_rows"])
    if settled_count < 300:
        recommendations.append(
            "Settled sample size is below 300; evaluation is useful for debugging but not production betting."
        )

    avg_clv = report["clv_summary"].get("avg_clv")
    if avg_clv is None:
        recommendations.append(
            "CLV could not be fully evaluated yet; ensure closing odds are stored for selected sides."
        )
    elif avg_clv < 0:
        recommendations.append(
            "Average CLV is negative; the model is not beating the closing line."
        )

    edge_summary = report.get("edge_bucket_summary", {})
    if (
        isinstance(edge_summary, dict)
        and "8pct_plus" in edge_summary
        and edge_summary["8pct_plus"].get("roi_flat") is not None
        and edge_summary["8pct_plus"]["roi_flat"] < 0
    ):
        recommendations.append(
            "The 8%+ edge bucket is underperforming; large anti-market edges may be false signals."
        )

    if report["current_prediction_summary"]["live_bet_candidate_count"] == 0:
        recommendations.append(
            "Current live_bet_candidate_count is 0; risk controls are blocking all real-bet candidates, which is expected during early model validation."
        )

    risk_counts = report["current_prediction_summary"].get("risk_flag_counts", {})
    if isinstance(risk_counts, dict) and risk_counts.get("lineup_not_confirmed", 0) > 0:
        recommendations.append(
            "Lineup confirmation remains a major blocker; projected or confirmed lineup integration should be prioritized."
        )

    if isinstance(risk_counts, dict) and risk_counts.get("closer_high_fatigue", 0) > 0:
        recommendations.append(
            "Closer high fatigue risk is present; keep stake_multiplier conservative or require manual review."
        )

    if not recommendations:
        recommendations.append(
            "Evaluation and CLV diagnostics did not find major issues, but continue monitoring sample size and CLV."
        )

    report["recommendations"] = recommendations

    safe_report = _json_safe(report)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file_obj:
        json.dump(safe_report, file_obj, indent=2, ensure_ascii=True)

    return safe_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    diagnostic = build_evaluation_clv_diagnostic()

    summary = {
        "generated_at": diagnostic.get("generated_at"),
        "current_prediction_summary": diagnostic.get("current_prediction_summary"),
        "settled_performance": diagnostic.get("settled_performance"),
        "clv_summary": diagnostic.get("clv_summary"),
        "recommendations": diagnostic.get("recommendations"),
        "report_written_to": "report/evaluation_clv_diagnostic.json",
    }

    print(json.dumps(summary, indent=2, ensure_ascii=True, default=str))
