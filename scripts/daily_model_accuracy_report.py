#!/usr/bin/env python3
"""
Daily Model Accuracy Report v1.

Research-only report that strictly separates:
- official accuracy from trusted finalized outcomes
- daily and rolling settled accuracy
- pending predictions that are not counted in the denominator
- CLV metrics, explicitly not win/loss accuracy
- forecast probability diagnostics

No live betting. No automated wagering. No production model replacement.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPORT_PATH = Path("report/daily_model_accuracy_report.json")

PREDICTION_SNAPSHOTS_PATH = Path("data/prediction_snapshots.csv")
FINALIZED_SNAPSHOT_OUTCOMES_PATH = Path("data/finalized_snapshot_outcomes.csv")
TRAINING_SAMPLES_PATH = Path("data/training_samples.csv")
MARKET_ODDS_HISTORY_PATH = Path("data/market_odds_history.csv")
SAMPLE_STATE_PATH = Path("data/sample_state.json")
EVALUATION_CLV_DIAGNOSTIC_PATH = Path("report/evaluation_clv_diagnostic.json")

PIPELINE_VERSION = "baseline_v2_clean"

PROBABILITY_COLUMNS = [
    "displayed_home_win_pct",
    "predicted_home_win_pct",
    "premarket_model_home_prob",
    "home_win_probability",
]

EXPLICIT_SIDE_COLUMNS = [
    "moneyline_selected_side",
    "selected_side",
    "model_pick_side",
    "pick_side",
]

LEAKAGE_COLUMNS_FROM_SNAPSHOTS = [
    "home_win",
    "home_score",
    "away_score",
    "settled_at",
    "final_status",
    "outcome_source",
    "actual_winner",
    "actual_result",
    "final_home_score",
    "final_away_score",
    "postgame_win_probability",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except Exception:
        pass
    try:
        return value.item()
    except Exception:
        return value


def safe_json_dump(data: dict[str, Any], filepath: str | Path) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(data), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def load_csv(path: str | Path) -> pd.DataFrame | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        return pd.read_csv(file_path)
    except Exception:
        return None


def load_json(path: str | Path) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return parsed
    except Exception:
        return None


def _safe_probability(value: Any) -> float | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if 1.0 < parsed <= 100.0:
        parsed = parsed / 100.0
    if parsed < 0.0 or parsed > 1.0:
        return None
    return parsed


def _normalize_game_id(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()
    if not text:
        return ""

    try:
        parsed = float(text)
        if parsed.is_integer():
            return str(int(parsed))
    except Exception:
        pass

    return text


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "valid", "ok"})
    )


def _get_home_probability(row: pd.Series) -> float | None:
    for column in PROBABILITY_COLUMNS:
        if column in row.index:
            value = _safe_probability(row.get(column))
            if value is not None:
                return value
    return None


def _recommendation_side(row: pd.Series) -> str | None:
    recommendation = str(row.get("moneyline_recommendation") or "").strip().lower()
    if not recommendation or recommendation in {"no bet", "pass", "tracking only", "no data"}:
        return None

    home_team = str(row.get("home_team") or "").strip().lower()
    away_team = str(row.get("away_team") or "").strip().lower()

    if away_team and away_team in recommendation:
        return "away"
    if home_team and home_team in recommendation:
        return "home"
    if "away" in recommendation:
        return "away"
    if "home" in recommendation:
        return "home"

    return None


def determine_side(row: pd.Series) -> str | None:
    for column in EXPLICIT_SIDE_COLUMNS:
        if column in row.index:
            text = str(row.get(column) or "").strip().lower()
            if text in {"home", "away"}:
                return text

    side_from_recommendation = _recommendation_side(row)
    if side_from_recommendation in {"home", "away"}:
        return side_from_recommendation

    edge_home = _safe_float(row.get("model_edge_home"))
    if edge_home is not None:
        if edge_home >= 0.03:
            return "home"
        if edge_home <= -0.03:
            return "away"

    home_probability = _get_home_probability(row)
    if home_probability is None:
        return None

    return "home" if home_probability >= 0.5 else "away"


def _is_paper_signal(row: pd.Series) -> bool:
    if "is_paper_signal" in row.index:
        value = str(row.get("is_paper_signal") or "").strip().lower()
        if value in {"true", "1", "yes", "y"}:
            return True

    status = str(row.get("recommendation_status") or "").strip().upper()
    return status in {"PAPER_BET", "PAPER_SIGNAL"}


def _is_tracking_only(row: pd.Series) -> bool:
    if "tracking_only" in row.index:
        value = str(row.get("tracking_only") or "").strip().lower()
        if value in {"true", "1", "yes", "y"}:
            return True

    status = str(row.get("recommendation_status") or "").strip().upper()
    return status in {"TRACKING_ONLY", "NO_BET", "BLOCKED_BY_RISK"}


def _prepare_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    frame = predictions_df.copy()

    if "game_id" not in frame.columns:
        return pd.DataFrame()

    frame["game_id"] = frame["game_id"].apply(_normalize_game_id)
    frame = frame[frame["game_id"] != ""].copy()

    if "pipeline_version" in frame.columns:
        preferred = frame[frame["pipeline_version"].astype(str) == PIPELINE_VERSION].copy()
        if not preferred.empty:
            frame = preferred

    if "snapshot_valid" in frame.columns:
        frame = frame[_normalize_bool_series(frame["snapshot_valid"])].copy()

    frame = frame.drop(
        columns=[column for column in LEAKAGE_COLUMNS_FROM_SNAPSHOTS if column in frame.columns],
        errors="ignore",
    )

    if "game_date" in frame.columns:
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    else:
        frame["game_date"] = pd.NaT

    if "snapshot_created_at" in frame.columns:
        frame["_snapshot_created_at"] = pd.to_datetime(
            frame["snapshot_created_at"],
            errors="coerce",
            utc=True,
        )
        frame = frame.sort_values("_snapshot_created_at")
        frame = frame.groupby("game_id", as_index=False).tail(1)
        frame = frame.drop(columns=["_snapshot_created_at"], errors="ignore")
    else:
        frame = frame.drop_duplicates("game_id", keep="last")

    return frame.reset_index(drop=True)


def _prepare_outcomes(outcomes_df: pd.DataFrame) -> pd.DataFrame:
    frame = outcomes_df.copy()

    if "game_id" not in frame.columns:
        return pd.DataFrame()

    frame["game_id"] = frame["game_id"].apply(_normalize_game_id)
    frame = frame[frame["game_id"] != ""].copy()

    if "home_win" not in frame.columns:
        if {"home_score", "away_score"}.issubset(set(frame.columns)):
            home_score = pd.to_numeric(frame["home_score"], errors="coerce")
            away_score = pd.to_numeric(frame["away_score"], errors="coerce")
            frame["home_win"] = (home_score > away_score).astype("Int64")
        else:
            return pd.DataFrame()

    frame["home_win"] = pd.to_numeric(frame["home_win"], errors="coerce")
    frame = frame[frame["home_win"].isin([0, 1])].copy()
    frame["home_win"] = frame["home_win"].astype(int)

    if "game_date" in frame.columns:
        frame["game_date_outcome"] = pd.to_datetime(frame["game_date"], errors="coerce")

    keep_columns = ["game_id", "home_win"]
    if "game_date_outcome" in frame.columns:
        keep_columns.append("game_date_outcome")

    return frame[keep_columns].drop_duplicates("game_id", keep="last").reset_index(drop=True)


def _accuracy_bucket(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"sample_count": 0, "correct": 0, "accuracy": None}

    sample_count = int(len(frame))
    correct = int(frame["correct"].sum())

    return {
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": float(correct / sample_count) if sample_count else None,
    }


def _brier_score(frame: pd.DataFrame) -> float | None:
    if frame.empty or "home_probability" not in frame.columns:
        return None

    scored = frame[["home_probability", "home_win"]].dropna().copy()
    if scored.empty:
        return None

    return float(((scored["home_probability"] - scored["home_win"]) ** 2).mean())


def _logloss_score(frame: pd.DataFrame) -> float | None:
    if frame.empty or "home_probability" not in frame.columns:
        return None

    scored = frame[["home_probability", "home_win"]].dropna().copy()
    if scored.empty:
        return None

    eps = 1e-15
    probability = scored["home_probability"].clip(eps, 1.0 - eps)
    outcome = scored["home_win"].astype(float)

    return float(-(outcome * probability.map(math.log) + (1.0 - outcome) * (1.0 - probability).map(math.log)).mean())


def _extract_clv_metrics(clv_data: dict[str, Any] | None) -> dict[str, Any]:
    metrics = {
        "available": False,
        "avg_clv": None,
        "positive_clv_rate": None,
        "sample_count": 0,
        "note": "CLV is price movement, not win/loss accuracy.",
    }

    if not isinstance(clv_data, dict):
        return metrics

    source = clv_data.get("clv_summary") if isinstance(clv_data.get("clv_summary"), dict) else clv_data

    avg_clv = _safe_float(source.get("avg_clv"))
    positive_clv_rate = _safe_float(source.get("positive_clv_rate"))
    sample_count = int(_safe_float(source.get("evaluated_picks") or source.get("clv_samples") or 0) or 0)

    metrics.update(
        {
            "available": avg_clv is not None or positive_clv_rate is not None,
            "avg_clv": avg_clv,
            "positive_clv_rate": positive_clv_rate,
            "sample_count": sample_count,
            "note": "CLV is price movement, not win/loss accuracy.",
        }
    )
    return metrics


def _rolling_window(scored: pd.DataFrame, days: int) -> dict[str, Any]:
    if scored.empty or "game_date" not in scored.columns:
        return {"sample_count": 0, "correct": 0, "accuracy": None}

    dated = scored.dropna(subset=["game_date"]).copy()
    if dated.empty:
        return {"sample_count": 0, "correct": 0, "accuracy": None}

    anchor = dated["game_date"].max().normalize()
    cutoff = anchor - pd.Timedelta(days=days - 1)

    window = dated[(dated["game_date"] >= cutoff) & (dated["game_date"] <= anchor)].copy()
    return _accuracy_bucket(window)


def _daily_accuracy(scored: pd.DataFrame, pending: pd.DataFrame) -> list[dict[str, Any]]:
    date_values: set[str] = set()

    if not scored.empty and "game_date" in scored.columns:
        date_values |= set(scored["game_date"].dropna().dt.strftime("%Y-%m-%d").tolist())

    if not pending.empty and "game_date" in pending.columns:
        date_values |= set(pending["game_date"].dropna().dt.strftime("%Y-%m-%d").tolist())

    rows: list[dict[str, Any]] = []

    for date_text in sorted(date_values):
        settled_day = (
            scored[scored["game_date"].dt.strftime("%Y-%m-%d") == date_text].copy()
            if not scored.empty
            else pd.DataFrame()
        )
        pending_day = (
            pending[pending["game_date"].dt.strftime("%Y-%m-%d") == date_text].copy()
            if not pending.empty
            else pd.DataFrame()
        )

        paper_day = settled_day[settled_day["is_paper_signal"]].copy() if not settled_day.empty else pd.DataFrame()

        rows.append(
            {
                "game_date": date_text,
                "sample_count": int(len(settled_day)),
                "correct": int(settled_day["correct"].sum()) if not settled_day.empty else 0,
                "accuracy": float(settled_day["correct"].mean()) if not settled_day.empty else None,
                "paper_signal_count": int(len(paper_day)),
                "paper_signal_correct": int(paper_day["correct"].sum()) if not paper_day.empty else 0,
                "paper_signal_accuracy": float(paper_day["correct"].mean()) if not paper_day.empty else None,
                "pending_count": int(len(pending_day)),
            }
        )

    return rows


def compute_report(
    predictions_df: pd.DataFrame | None,
    outcomes_df: pd.DataFrame | None,
    odds_df: pd.DataFrame | None = None,
    clv_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    recommendations: list[str] = []

    report: dict[str, Any] = {
        "generated_at": _utc_now(),
        "status": "ok",
        "pipeline_version": PIPELINE_VERSION,
        "input_files": {
            "prediction_snapshots": {"required": True, "available": predictions_df is not None},
            "finalized_snapshot_outcomes": {"required": True, "available": outcomes_df is not None},
            "market_odds_history": {"required": False, "available": odds_df is not None},
            "evaluation_clv_diagnostic": {"required": False, "available": clv_data is not None},
        },
        "official_accuracy": {
            "sample_count": 0,
            "correct": 0,
            "accuracy": None,
            "brier": None,
            "logloss": None,
            "source": "trusted_finalized_snapshot_outcomes",
        },
        "daily_accuracy": [],
        "rolling_windows": {
            "7d": {"sample_count": 0, "correct": 0, "accuracy": None},
            "30d": {"sample_count": 0, "correct": 0, "accuracy": None},
        },
        "slices": {
            "home_picks": {"sample_count": 0, "correct": 0, "accuracy": None},
            "away_picks": {"sample_count": 0, "correct": 0, "accuracy": None},
            "favorites": {"sample_count": 0, "correct": 0, "accuracy": None},
            "underdogs": {"sample_count": 0, "correct": 0, "accuracy": None},
            "paper_signals": {"sample_count": 0, "correct": 0, "accuracy": None},
            "tracking_only": {"sample_count": 0, "correct": 0, "accuracy": None},
        },
        "pending_predictions": {
            "count": 0,
            "latest_game_date": None,
            "invalid_scored_count": 0,
            "reason": "unsettled_or_missing_outcome",
        },
        "clv_metrics": _extract_clv_metrics(clv_data),
        "prediction_probability_metrics": {
            "forecast_only": True,
            "prediction_count": 0,
            "avg_home_probability": None,
            "avg_selected_probability": None,
            "note": "Forecast probabilities are not settled win/loss accuracy.",
        },
        "interpretation": {
            "official_accuracy_available": False,
            "display_warning": "Accuracy only includes settled games with trusted outcomes.",
            "do_not_mix_with_clv": True,
        },
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
        "warnings": warnings,
        "errors": errors,
        "recommendations": recommendations,
    }

    if predictions_df is None:
        errors.append("Missing data/prediction_snapshots.csv")
        recommendations.append("Run prediction.py before daily_model_accuracy_report.py.")
        report["status"] = "partial"
        return report

    if outcomes_df is None:
        errors.append("Missing data/finalized_snapshot_outcomes.csv")
        recommendations.append("Run scripts/repair_finalized_linkage.py before daily_model_accuracy_report.py.")
        report["status"] = "partial"
        return report

    predictions = _prepare_predictions(predictions_df)
    outcomes = _prepare_outcomes(outcomes_df)

    if predictions.empty:
        warnings.append("No clean prediction snapshots available.")
        report["status"] = "partial"
        return report

    report["prediction_probability_metrics"]["prediction_count"] = int(len(predictions))

    prediction_home_probs = predictions.apply(_get_home_probability, axis=1)
    clean_home_probs = [value for value in prediction_home_probs.tolist() if value is not None]
    if clean_home_probs:
        report["prediction_probability_metrics"]["avg_home_probability"] = float(sum(clean_home_probs) / len(clean_home_probs))

    if outcomes.empty:
        warnings.append("No trusted finalized outcomes with valid home_win are available.")
        report["pending_predictions"]["count"] = int(len(predictions))
        if "game_date" in predictions.columns and not predictions["game_date"].dropna().empty:
            report["pending_predictions"]["latest_game_date"] = predictions["game_date"].max().strftime("%Y-%m-%d")
        report["status"] = "partial"
        return report

    merged = predictions.merge(outcomes, on="game_id", how="left")
    settled = merged[merged["home_win"].isin([0, 1])].copy()
    pending = merged[~merged["home_win"].isin([0, 1])].copy()

    if not pending.empty and "game_date" in pending.columns and not pending["game_date"].dropna().empty:
        report["pending_predictions"]["latest_game_date"] = pending["game_date"].max().strftime("%Y-%m-%d")

    report["pending_predictions"]["count"] = int(len(pending))

    if settled.empty:
        warnings.append("No prediction snapshots join trusted finalized outcomes.")
        report["daily_accuracy"] = _daily_accuracy(pd.DataFrame(), pending)
        report["status"] = "partial"
        return report

    settled["home_win"] = settled["home_win"].astype(int)
    settled["home_probability"] = settled.apply(_get_home_probability, axis=1)
    settled["pick"] = settled.apply(determine_side, axis=1)
    settled["is_paper_signal"] = settled.apply(_is_paper_signal, axis=1)
    settled["is_tracking_only"] = settled.apply(_is_tracking_only, axis=1)

    scored = settled[settled["pick"].isin(["home", "away"])].copy()
    invalid_scored = settled[~settled["pick"].isin(["home", "away"])].copy()

    if not invalid_scored.empty:
        warnings.append(f"{len(invalid_scored)} settled rows have no valid pick side and are excluded from accuracy.")
        report["pending_predictions"]["invalid_scored_count"] = int(len(invalid_scored))

    if scored.empty:
        warnings.append("No settled rows have a valid pick side.")
        report["status"] = "partial"
        return report

    scored["correct"] = (
        ((scored["pick"] == "home") & (scored["home_win"] == 1))
        | ((scored["pick"] == "away") & (scored["home_win"] == 0))
    )

    selected_probs: list[float] = []
    for _, row in scored.iterrows():
        home_probability = _safe_probability(row.get("home_probability"))
        if home_probability is None:
            continue
        if row.get("pick") == "home":
            selected_probs.append(home_probability)
        elif row.get("pick") == "away":
            selected_probs.append(1.0 - home_probability)

    if selected_probs:
        report["prediction_probability_metrics"]["avg_selected_probability"] = float(sum(selected_probs) / len(selected_probs))

    official = _accuracy_bucket(scored)
    official["brier"] = _brier_score(scored)
    official["logloss"] = _logloss_score(scored)
    official["source"] = "trusted_finalized_snapshot_outcomes"

    report["official_accuracy"] = official
    report["interpretation"]["official_accuracy_available"] = official["sample_count"] > 0

    report["daily_accuracy"] = _daily_accuracy(scored, pending)
    report["rolling_windows"]["7d"] = _rolling_window(scored, 7)
    report["rolling_windows"]["30d"] = _rolling_window(scored, 30)

    market_prob = None
    if "market_no_vig_home_prob" in scored.columns:
        market_prob = scored["market_no_vig_home_prob"].map(_safe_probability)
    elif "market_home_prob" in scored.columns:
        market_prob = scored["market_home_prob"].map(_safe_probability)

    scored["_market_home_probability"] = market_prob if market_prob is not None else None

    if "_market_home_probability" in scored.columns:
        with_market = scored[scored["_market_home_probability"].notna()].copy()
        if not with_market.empty:
            with_market["market_favorite_side"] = with_market["_market_home_probability"].apply(
                lambda value: "home" if value >= 0.5 else "away"
            )
            favorites = with_market[with_market["pick"] == with_market["market_favorite_side"]].copy()
            underdogs = with_market[with_market["pick"] != with_market["market_favorite_side"]].copy()
            report["slices"]["favorites"] = _accuracy_bucket(favorites)
            report["slices"]["underdogs"] = _accuracy_bucket(underdogs)
        else:
            warnings.append("Market home probability is unavailable; favorite/underdog slices remain empty.")
    else:
        warnings.append("Market home probability column is unavailable; favorite/underdog slices remain empty.")

    report["slices"]["home_picks"] = _accuracy_bucket(scored[scored["pick"] == "home"])
    report["slices"]["away_picks"] = _accuracy_bucket(scored[scored["pick"] == "away"])
    report["slices"]["paper_signals"] = _accuracy_bucket(scored[scored["is_paper_signal"]])
    report["slices"]["tracking_only"] = _accuracy_bucket(scored[scored["is_tracking_only"]])

    if report["pending_predictions"]["count"] > 0:
        recommendations.append(
            f"{report['pending_predictions']['count']} predictions are pending and excluded from official accuracy."
        )

    if not report["clv_metrics"]["available"]:
        warnings.append("CLV diagnostic is unavailable; CLV metrics are not included.")

    report["status"] = "partial" if errors or warnings else "ok"
    return report


def main() -> None:
    predictions = load_csv(PREDICTION_SNAPSHOTS_PATH)
    outcomes = load_csv(FINALIZED_SNAPSHOT_OUTCOMES_PATH)
    odds = load_csv(MARKET_ODDS_HISTORY_PATH)
    clv = load_json(EVALUATION_CLV_DIAGNOSTIC_PATH)

    report = compute_report(
        predictions_df=predictions,
        outcomes_df=outcomes,
        odds_df=odds,
        clv_data=clv,
    )

    safe_json_dump(report, REPORT_PATH)
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
