# scripts/away_pick_diagnostic_report.py
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPORT_PATH = Path("report/away_pick_diagnostic_report.json")
PREDICTION_SNAPSHOTS_PATH = Path("data/prediction_snapshots.csv")
FINALIZED_SNAPSHOT_OUTCOMES_PATH = Path("data/finalized_snapshot_outcomes.csv")
EVALUATION_CLV_DIAGNOSTIC_PATH = Path("report/evaluation_clv_diagnostic.json")
SAMPLE_STATE_PATH = Path("data/sample_state.json")

PIPELINE_VERSION = "baseline_v2_clean"
REPORT_TYPE = "away_pick_diagnostic_v1"

LEAKAGE_COLUMNS = [
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

PROBABILITY_COLUMNS = [
    "displayed_home_win_pct",
    "predicted_home_win_pct",
    "premarket_model_home_prob",
    "home_win_probability",
]

PICK_SIDE_COLUMNS = [
    "moneyline_selected_side",
    "selected_side",
    "model_pick_side",
    "pick_side",
]

MARKET_HOME_PROBABILITY_COLUMNS = [
    "market_no_vig_home_prob",
    "market_home_prob",
]

CLV_COLUMNS = [
    "clv",
    "moneyline_clv",
    "selected_clv",
    "closing_line_value",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
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


def _truthy(value: Any) -> bool:
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    return text in {
        "true",
        "1",
        "yes",
        "y",
        "valid",
        "ok",
        "confirmed",
        "available",
    }


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "valid", "ok"})
    )


def _get_first_probability(row: pd.Series, columns: list[str]) -> float | None:
    for column in columns:
        if column in row.index:
            value = _safe_probability(row.get(column))
            if value is not None:
                return value
    return None


def _get_home_probability(row: pd.Series) -> float | None:
    return _get_first_probability(row, PROBABILITY_COLUMNS)


def _get_market_home_probability(row: pd.Series) -> float | None:
    return _get_first_probability(row, MARKET_HOME_PROBABILITY_COLUMNS)


def _get_clv(row: pd.Series) -> float | None:
    for column in CLV_COLUMNS:
        if column in row.index:
            value = _safe_float(row.get(column))
            if value is not None:
                return value
    return None


def _recommendation_side(row: pd.Series) -> str | None:
    recommendation = str(row.get("moneyline_recommendation") or "").strip().lower()
    if not recommendation or recommendation in {
        "no bet",
        "pass",
        "tracking only",
        "no data",
    }:
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


def determine_pick_side(row: pd.Series) -> str | None:
    for column in PICK_SIDE_COLUMNS:
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


def _selected_probability(row: pd.Series) -> float | None:
    home_probability = _safe_probability(row.get("home_probability"))
    if home_probability is None:
        return None

    if row.get("pick_side") == "home":
        return home_probability
    if row.get("pick_side") == "away":
        return 1.0 - home_probability

    return None


def _away_selected_edge(row: pd.Series) -> float | None:
    edge_home = _safe_float(row.get("model_edge_home"))
    if edge_home is None:
        return None
    return -edge_home


def _availability_from_columns(
    row: pd.Series,
    columns: list[str],
    require_all: bool = False,
) -> bool | None:
    found = [column for column in columns if column in row.index]
    if not found:
        return None

    values = [_truthy(row.get(column)) for column in found]
    return all(values) if require_all else any(values)


def _lineup_confirmed(row: pd.Series) -> bool | None:
    for column in [
        "lineup_confirmed",
        "lineup_context_confirmed",
        "all_lineups_confirmed",
    ]:
        if column in row.index:
            return _truthy(row.get(column))

    home_status = str(row.get("home_lineup_status") or "").strip().lower()
    away_status = str(row.get("away_lineup_status") or "").strip().lower()

    if home_status or away_status:
        return home_status == "confirmed" and away_status == "confirmed"

    return _availability_from_columns(row, ["lineup_context_available"])


def _pitcher_advanced_available(row: pd.Series) -> bool | None:
    return _availability_from_columns(
        row,
        [
            "pitcher_advanced_available",
            "sp_fip_diff_available",
            "sp_csw_diff_available",
            "sp_stuff_plus_diff_available",
        ],
    )


def _bullpen_context_available(row: pd.Series) -> bool | None:
    return _availability_from_columns(
        row,
        [
            "bullpen_context_available",
            "bullpen_ip_diff_available",
            "bullpen_availability_diff_available",
        ],
    )


def _context_confirmed(row: pd.Series) -> bool:
    explicit = _availability_from_columns(
        row,
        ["context_confirmed", "all_context_confirmed"],
        require_all=True,
    )
    if explicit is not None:
        return explicit

    pitcher = _pitcher_advanced_available(row)
    bullpen = _bullpen_context_available(row)
    lineup = _lineup_confirmed(row)

    checks = [value for value in [pitcher, bullpen, lineup] if value is not None]
    return bool(checks and all(checks))


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
        columns=[column for column in LEAKAGE_COLUMNS if column in frame.columns],
        errors="ignore",
    )

    if "game_date" in frame.columns:
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")

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

    frame["home_probability"] = frame.apply(_get_home_probability, axis=1)
    frame["market_home_probability"] = frame.apply(_get_market_home_probability, axis=1)
    frame["pick_side"] = frame.apply(determine_pick_side, axis=1)
    frame["selected_probability"] = frame.apply(_selected_probability, axis=1)
    frame["away_selected_edge"] = frame.apply(_away_selected_edge, axis=1)
    frame["row_clv"] = frame.apply(_get_clv, axis=1)

    frame["pitcher_advanced_available_flag"] = frame.apply(
        _pitcher_advanced_available,
        axis=1,
    )
    frame["bullpen_context_available_flag"] = frame.apply(
        _bullpen_context_available,
        axis=1,
    )
    frame["lineup_confirmed_flag"] = frame.apply(_lineup_confirmed, axis=1)
    frame["context_confirmed_flag"] = frame.apply(_context_confirmed, axis=1)

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

    return (
        frame[["game_id", "home_win"]]
        .drop_duplicates("game_id", keep="last")
        .reset_index(drop=True)
    )


def _accuracy_bucket(frame: pd.DataFrame, include_brier: bool = True) -> dict[str, Any]:
    sample = frame[
        frame["home_win"].isin([0, 1])
        & frame["pick_side"].isin(["home", "away"])
    ].copy()

    if sample.empty:
        result = {"sample_count": 0, "correct": 0, "accuracy": None}
        if include_brier:
            result["brier"] = None
        return result

    correct_series = (
        ((sample["pick_side"] == "home") & (sample["home_win"] == 1))
        | ((sample["pick_side"] == "away") & (sample["home_win"] == 0))
    )

    sample_count = int(len(sample))
    correct = int(correct_series.sum())

    result = {
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": float(correct / sample_count) if sample_count else None,
    }

    if include_brier:
        scored = sample[["home_probability", "home_win"]].dropna().copy()
        result["brier"] = (
            float(((scored["home_probability"] - scored["home_win"]) ** 2).mean())
            if not scored.empty
            else None
        )

    return result


def _segment_summary(frame: pd.DataFrame) -> dict[str, Any]:
    sample = frame[
        frame["home_win"].isin([0, 1])
        & (frame["pick_side"] == "away")
    ].copy()

    if sample.empty:
        return {
            "sample_count": 0,
            "correct": 0,
            "accuracy": None,
            "avg_selected_probability": None,
            "avg_model_edge": None,
            "avg_market_home_probability": None,
        }

    bucket = _accuracy_bucket(sample, include_brier=False)

    selected_probability = pd.to_numeric(
        sample["selected_probability"],
        errors="coerce",
    ).dropna()
    away_edge = pd.to_numeric(sample["away_selected_edge"], errors="coerce").dropna()
    market_home = pd.to_numeric(
        sample["market_home_probability"],
        errors="coerce",
    ).dropna()

    bucket.update(
        {
            "avg_selected_probability": (
                float(selected_probability.mean())
                if not selected_probability.empty
                else None
            ),
            "avg_model_edge": float(away_edge.mean()) if not away_edge.empty else None,
            "avg_market_home_probability": (
                float(market_home.mean()) if not market_home.empty else None
            ),
        }
    )

    return bucket


def _clv_summary(clv_data: dict[str, Any] | None) -> dict[str, Any]:
    result = {
        "available": False,
        "avg_clv": None,
        "positive_clv_rate": None,
        "sample_count": 0,
        "note": "CLV is price movement, not win/loss accuracy.",
    }

    if not isinstance(clv_data, dict):
        return result

    source = clv_data.get("clv_summary") if isinstance(clv_data.get("clv_summary"), dict) else clv_data
    source = source.get("clv_metrics") if isinstance(source.get("clv_metrics"), dict) else source

    avg_clv = _safe_float(source.get("avg_clv") or source.get("average_clv"))
    positive_clv_rate = _safe_float(source.get("positive_clv_rate"))
    sample_count = int(
        _safe_float(
            source.get("evaluated_picks")
            or source.get("clv_samples")
            or source.get("sample_count")
            or 0
        )
        or 0
    )

    result.update(
        {
            "available": avg_clv is not None or positive_clv_rate is not None or sample_count > 0,
            "avg_clv": avg_clv,
            "positive_clv_rate": positive_clv_rate,
            "sample_count": sample_count,
        }
    )

    return result


def _sample_state_clean_count(sample_state: dict[str, Any] | None) -> int | None:
    if not isinstance(sample_state, dict):
        return None

    for key in (
        "clean_sample_count",
        "clean_settled_samples",
        "training_sample_count",
        "train_eligible_samples",
        "clean_train_eligible_samples",
    ):
        value = _safe_float(sample_state.get(key))
        if value is not None:
            return int(value)

    return None


def _market_bucket_label(value: float | None) -> str | None:
    if value is None:
        return None
    if value < 0.45:
        return "home_market_prob < 0.45: away favorite / strong away market"
    if value < 0.50:
        return "0.45-0.50: slight away market"
    if value < 0.55:
        return "0.50-0.55: slight home market"
    return ">= 0.55: away underdog / strong home market"


def _edge_bucket_label(value: float | None) -> str | None:
    if value is None:
        return None

    abs_value = abs(value)

    if abs_value < 0.03:
        return "<3%"
    if abs_value < 0.05:
        return "3-5%"
    if abs_value < 0.08:
        return "5-8%"
    return ">=8%"


def _recommend_guardrails(
    *,
    official_accuracy: dict[str, Any],
    away_segments: dict[str, Any],
    away_by_edge_bucket: list[dict[str, Any]],
) -> list[str]:
    guardrails: list[str] = []

    away_bucket = official_accuracy.get("away_picks", {})
    home_bucket = official_accuracy.get("home_picks", {})

    away_sample = int(away_bucket.get("sample_count") or 0)
    away_accuracy = away_bucket.get("accuracy")
    home_accuracy = home_bucket.get("accuracy")

    if away_sample < 30:
        guardrails.append(
            "Away pick sample is still small; keep away picks tracking-only until more settled evidence accumulates."
        )

    if away_accuracy is not None and home_accuracy is not None and away_sample >= 30:
        if float(away_accuracy) < float(home_accuracy) - 0.08:
            guardrails.append(
                "Away picks materially underperform home picks; raise away edge threshold."
            )

    away_underdogs = away_segments.get("away_underdogs", {})
    if (
        away_underdogs.get("accuracy") is not None
        and int(away_underdogs.get("sample_count") or 0) >= 20
        and float(away_underdogs.get("accuracy")) < 0.48
    ):
        guardrails.append(
            "Away underdogs are weak; require stronger edge or downgrade to tracking-only."
        )

    def maybe_context_guardrail(
        unavailable_key: str,
        available_key: str,
        message: str,
    ) -> None:
        unavailable = away_segments.get(unavailable_key, {})
        available = away_segments.get(available_key, {})

        if unavailable.get("accuracy") is None or available.get("accuracy") is None:
            return

        if int(unavailable.get("sample_count") or 0) < 10:
            return

        if int(available.get("sample_count") or 0) < 10:
            return

        if float(unavailable["accuracy"]) < float(available["accuracy"]) - 0.05:
            guardrails.append(message)

    maybe_context_guardrail(
        "away_pitcher_advanced_unavailable",
        "away_pitcher_advanced_available",
        "Require pitcher advanced context for away paper signals.",
    )
    maybe_context_guardrail(
        "away_bullpen_context_unavailable",
        "away_bullpen_context_available",
        "Require bullpen context for away paper signals.",
    )
    maybe_context_guardrail(
        "away_unconfirmed_context",
        "away_confirmed_context",
        "Require confirmed starter/lineup context for away paper signals.",
    )

    large_edge = next(
        (bucket for bucket in away_by_edge_bucket if bucket.get("label") == ">=8%"),
        None,
    )
    if (
        large_edge
        and large_edge.get("accuracy") is not None
        and int(large_edge.get("sample_count") or 0) >= 10
        and float(large_edge["accuracy"]) < 0.50
    ):
        guardrails.append(
            "Large away edges may be market-disagreement artifacts; cap or downgrade large away edges until CLV evidence improves."
        )

    if not guardrails:
        guardrails.append("Insufficient data to generate specific guardrails; continue monitoring.")

    return guardrails


def _base_report(
    predictions_df: pd.DataFrame | None,
    outcomes_df: pd.DataFrame | None,
    clv_data: dict[str, Any] | None,
    sample_state: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "metadata": {
            "generated_at": _utc_now(),
            "status": "ok",
            "pipeline_version": PIPELINE_VERSION,
            "report_type": REPORT_TYPE,
            "betting_mode": "paper_research",
            "live_betting_allowed": False,
            "automated_wagering_allowed": False,
            "production_model_replacement_allowed": False,
        },
        "status": "ok",
        "pipeline_version": PIPELINE_VERSION,
        "report_type": REPORT_TYPE,
        "betting_mode": "paper_research",
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
        "input_files": {
            "prediction_snapshots": {
                "required": True,
                "available": predictions_df is not None,
            },
            "finalized_snapshot_outcomes": {
                "required": True,
                "available": outcomes_df is not None,
            },
            "evaluation_clv_diagnostic": {
                "required": False,
                "available": clv_data is not None,
            },
            "sample_state": {
                "required": False,
                "available": sample_state is not None,
            },
        },
        "sample_summary": {
            "total_predictions": 0,
            "settled_predictions": 0,
            "pending_predictions": 0,
            "home_pick_count": 0,
            "away_pick_count": 0,
            "away_pick_settled_count": 0,
            "away_pick_pending_count": 0,
            "away_pick_rate": None,
            "clean_sample_count_from_sample_state": _sample_state_clean_count(sample_state),
        },
        "official_accuracy": {
            "all_picks": {
                "sample_count": 0,
                "correct": 0,
                "accuracy": None,
                "brier": None,
            },
            "home_picks": {
                "sample_count": 0,
                "correct": 0,
                "accuracy": None,
                "brier": None,
            },
            "away_picks": {
                "sample_count": 0,
                "correct": 0,
                "accuracy": None,
                "brier": None,
            },
        },
        "away_segments": {},
        "away_by_edge_bucket": [],
        "away_by_market_prob_bucket": [],
        "clv_summary": _clv_summary(clv_data),
        "recommended_guardrails": [],
        "interpretation": {
            "official_accuracy_note": "Only settled predictions with trusted finalized outcomes are counted.",
            "pending_note": "Pending predictions are excluded from accuracy denominators.",
            "clv_note": "CLV is price movement, not win/loss accuracy.",
            "recommended_use": "Use this report to decide whether away picks need stricter paper-signal guardrails.",
        },
        "warnings": [],
        "errors": [],
        "recommendations": [],
    }


def compute_report(
    predictions_df: pd.DataFrame | None,
    outcomes_df: pd.DataFrame | None,
    clv_data: dict[str, Any] | None = None,
    sample_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = _base_report(
        predictions_df=predictions_df,
        outcomes_df=outcomes_df,
        clv_data=clv_data,
        sample_state=sample_state,
    )

    if predictions_df is None:
        report["metadata"]["status"] = "partial"
        report["status"] = "partial"
        report["errors"].append("Missing data/prediction_snapshots.csv")
        report["recommendations"].append("Run prediction.py before away_pick_diagnostic_report.py.")
        report["recommended_guardrails"] = _recommend_guardrails(
            official_accuracy=report["official_accuracy"],
            away_segments={},
            away_by_edge_bucket=[],
        )
        return report

    if outcomes_df is None:
        report["metadata"]["status"] = "partial"
        report["status"] = "partial"
        report["errors"].append("Missing data/finalized_snapshot_outcomes.csv")

    predictions = _prepare_predictions(predictions_df)
    if predictions.empty:
        report["metadata"]["status"] = "partial"
        report["status"] = "partial"
        report["errors"].append("No clean prediction snapshots with valid game_id are available.")
        report["recommended_guardrails"] = _recommend_guardrails(
            official_accuracy=report["official_accuracy"],
            away_segments={},
            away_by_edge_bucket=[],
        )
        return report

    outcomes = _prepare_outcomes(outcomes_df) if outcomes_df is not None else pd.DataFrame()

    if outcomes.empty:
        merged = predictions.copy()
        merged["home_win"] = pd.NA
    else:
        merged = predictions.merge(outcomes, on="game_id", how="left")

    pick_known = merged["pick_side"].isin(["home", "away"])
    unknown_pick_count = int((~pick_known).sum())
    if unknown_pick_count > 0:
        report["warnings"].append(
            f"{unknown_pick_count} predictions have no valid pick side and are excluded from accuracy."
        )

    settled = merged[merged["home_win"].isin([0, 1]) & pick_known].copy()
    pending = merged[~merged["home_win"].isin([0, 1]) & pick_known].copy()

    home_picks = merged[pick_known & (merged["pick_side"] == "home")].copy()
    away_picks = merged[pick_known & (merged["pick_side"] == "away")].copy()
    away_settled = settled[settled["pick_side"] == "away"].copy()

    total_predictions = int(len(merged))
    away_pick_count = int(len(away_picks))

    report["sample_summary"].update(
        {
            "total_predictions": total_predictions,
            "settled_predictions": int(len(settled)),
            "pending_predictions": int(len(pending)),
            "home_pick_count": int(len(home_picks)),
            "away_pick_count": away_pick_count,
            "away_pick_settled_count": int(len(away_settled)),
            "away_pick_pending_count": int(
                len(away_picks[~away_picks["home_win"].isin([0, 1])])
            ),
            "away_pick_rate": (
                float(away_pick_count / total_predictions)
                if total_predictions
                else None
            ),
        }
    )

    report["official_accuracy"] = {
        "all_picks": _accuracy_bucket(settled),
        "home_picks": _accuracy_bucket(settled[settled["pick_side"] == "home"]),
        "away_picks": _accuracy_bucket(away_settled),
    }

    away_all = away_picks.copy()

    away_segments = {
        "away_favorites": _segment_summary(
            away_all[away_all["market_home_probability"] < 0.5]
        ),
        "away_underdogs": _segment_summary(
            away_all[away_all["market_home_probability"] >= 0.5]
        ),
        "away_high_edge": _segment_summary(
            away_all[away_all["away_selected_edge"].abs() >= 0.05]
        ),
        "away_low_edge": _segment_summary(
            away_all[away_all["away_selected_edge"].abs() < 0.03]
        ),
        "away_positive_clv": _segment_summary(away_all[away_all["row_clv"] > 0]),
        "away_negative_clv": _segment_summary(away_all[away_all["row_clv"] <= 0]),
        "away_confirmed_context": _segment_summary(
            away_all[away_all["context_confirmed_flag"] == True]
        ),
        "away_unconfirmed_context": _segment_summary(
            away_all[away_all["context_confirmed_flag"] != True]
        ),
        "away_pitcher_advanced_available": _segment_summary(
            away_all[away_all["pitcher_advanced_available_flag"] == True]
        ),
        "away_pitcher_advanced_unavailable": _segment_summary(
            away_all[away_all["pitcher_advanced_available_flag"] != True]
        ),
        "away_bullpen_context_available": _segment_summary(
            away_all[away_all["bullpen_context_available_flag"] == True]
        ),
        "away_bullpen_context_unavailable": _segment_summary(
            away_all[away_all["bullpen_context_available_flag"] != True]
        ),
        "away_lineup_confirmed": _segment_summary(
            away_all[away_all["lineup_confirmed_flag"] == True]
        ),
        "away_lineup_unconfirmed": _segment_summary(
            away_all[away_all["lineup_confirmed_flag"] != True]
        ),
    }
    report["away_segments"] = away_segments

    edge_labels = ["<3%", "3-5%", "5-8%", ">=8%"]
    away_all = away_all.copy()
    away_all["edge_bucket"] = away_all["away_selected_edge"].apply(_edge_bucket_label)

    for label in edge_labels:
        summary = _segment_summary(away_all[away_all["edge_bucket"] == label])
        summary["label"] = label
        report["away_by_edge_bucket"].append(summary)

    market_labels = [
        "home_market_prob < 0.45: away favorite / strong away market",
        "0.45-0.50: slight away market",
        "0.50-0.55: slight home market",
        ">= 0.55: away underdog / strong home market",
    ]
    away_all["market_bucket"] = away_all["market_home_probability"].apply(
        _market_bucket_label
    )

    for label in market_labels:
        summary = _segment_summary(away_all[away_all["market_bucket"] == label])
        summary["label"] = label
        report["away_by_market_prob_bucket"].append(summary)

    report["recommended_guardrails"] = _recommend_guardrails(
        official_accuracy=report["official_accuracy"],
        away_segments=away_segments,
        away_by_edge_bucket=report["away_by_edge_bucket"],
    )

    if not report["clv_summary"]["available"]:
        report["warnings"].append(
            "CLV diagnostic is unavailable or does not contain CLV summary metrics."
        )

    if report["errors"]:
        report["metadata"]["status"] = "partial"
        report["status"] = "partial"

    return report


def generate_report() -> dict[str, Any]:
    predictions = load_csv(PREDICTION_SNAPSHOTS_PATH)
    outcomes = load_csv(FINALIZED_SNAPSHOT_OUTCOMES_PATH)
    clv = load_json(EVALUATION_CLV_DIAGNOSTIC_PATH)
    sample_state = load_json(SAMPLE_STATE_PATH)

    return compute_report(
        predictions_df=predictions,
        outcomes_df=outcomes,
        clv_data=clv,
        sample_state=sample_state,
    )


def main() -> None:
    report = generate_report()
    safe_json_dump(report, REPORT_PATH)
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
