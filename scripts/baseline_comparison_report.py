from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REPORT_DIR = Path("report")
DATA_DIR = Path("data")

PREDICTION_JSON = REPORT_DIR / "prediction.json"
SNAPSHOTS_CSV = DATA_DIR / "prediction_snapshots.csv"
FINALIZED_CSV = DATA_DIR / "finalized_games.csv"
MARKET_ODDS_CSV = DATA_DIR / "market_odds_history.csv"

OUTPUT_JSON = REPORT_DIR / "baseline_comparison_report.json"

EPSILON = 1e-12
MIN_PRODUCTION_SETTLED_SAMPLES = 300
REQUIRED_PROMOTION_BASELINES = ["constant_50", "home_historical_rate", "market_no_vig"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(child) for child in value]
    return str(value)


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
    predictions = data.get("predictions") or data.get("today_predictions") or data.get("games") or []
    status["rows"] = len(predictions) if isinstance(predictions, list) else None
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


def _clip_probability(value: Any) -> Optional[float]:
    parsed = _as_float(value)
    if parsed is None:
        return None
    if parsed > 1.0 and parsed <= 100.0:
        parsed = parsed / 100.0
    if parsed < 0.0 or parsed > 1.0:
        return None
    return float(max(EPSILON, min(1.0 - EPSILON, parsed)))


def _brier(probabilities: List[float], outcomes: List[int]) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def _logloss(probabilities: List[float], outcomes: List[int]) -> float:
    clipped = [_clip_probability(prob) for prob in probabilities]
    p = np.asarray([prob for prob in clipped if prob is not None], dtype=float)
    y = np.asarray(outcomes[: len(p)], dtype=float)
    if len(p) == 0:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _accuracy(probabilities: List[float], outcomes: List[int]) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    return float(np.mean((p >= 0.5).astype(int) == y))


def _no_vig_home_probability(home_odds: Any, away_odds: Any) -> Optional[float]:
    home = _as_float(home_odds)
    away = _as_float(away_odds)
    if home is None or away is None or home <= 1.0 or away <= 1.0:
        return None
    home_raw = 1.0 / home
    away_raw = 1.0 / away
    total = home_raw + away_raw
    if total <= 0:
        return None
    return _clip_probability(home_raw / total)


def _elo_probability_from_diff(elo_diff: Any) -> Optional[float]:
    diff = _as_float(elo_diff)
    if diff is None:
        return None
    return _clip_probability(1.0 / (1.0 + math.pow(10.0, -diff / 400.0)))


def _normalise_game_id_series(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "game_id" in frame.columns:
        frame["game_id"] = frame["game_id"].astype(str)
    return frame


def _prepare_settled_rows(
    snapshots: Optional[pd.DataFrame],
    finalized: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if snapshots is None or snapshots.empty or "game_id" not in snapshots.columns:
        return pd.DataFrame()
    if finalized is None or finalized.empty or "game_id" not in finalized.columns:
        return pd.DataFrame()

    frame = _normalise_game_id_series(snapshots)
    final = _normalise_game_id_series(finalized)

    if "home_win" not in final.columns:
        if {"home_score", "away_score"}.issubset(final.columns):
            final["home_win"] = (
                pd.to_numeric(final["home_score"], errors="coerce")
                > pd.to_numeric(final["away_score"], errors="coerce")
            ).astype("Int64")
        else:
            return pd.DataFrame()

    final["home_win"] = pd.to_numeric(final["home_win"], errors="coerce")
    final = final[final["home_win"].isin([0, 1])].copy()
    if final.empty:
        return pd.DataFrame()

    leaked_snapshot_columns = [
        column for column in ["home_win", "home_score", "away_score"] if column in frame.columns
    ]
    if leaked_snapshot_columns:
        frame = frame.drop(columns=leaked_snapshot_columns)

    needed = [
        column for column in ["game_id", "home_win", "home_score", "away_score"] if column in final.columns
    ]
    frame = frame.merge(final[needed].drop_duplicates("game_id"), on="game_id", how="inner")
    if frame.empty:
        return frame

    if "snapshot_created_at" in frame.columns:
        frame["_snapshot_dt"] = pd.to_datetime(frame["snapshot_created_at"], errors="coerce", utc=True)
        frame = frame.sort_values("_snapshot_dt").groupby("game_id", as_index=False).tail(1)

    return frame.reset_index(drop=True)


def _prob_from_row(row: pd.Series, candidates: List[str]) -> Optional[float]:
    for column in candidates:
        if column in row.index:
            prob = _clip_probability(row.get(column))
            if prob is not None:
                return prob
    return None


def _baseline_metrics(
    frame: pd.DataFrame,
    probability_column: str,
    label: str,
) -> Optional[Dict[str, Any]]:
    if probability_column not in frame.columns or "home_win" not in frame.columns:
        return None
    probabilities: List[float] = []
    outcomes: List[int] = []
    for _, row in frame.iterrows():
        prob = _clip_probability(row.get(probability_column))
        outcome = _as_int(row.get("home_win"))
        if prob is None or outcome not in (0, 1):
            continue
        probabilities.append(prob)
        outcomes.append(int(outcome))
    if not probabilities:
        return None
    return {
        "name": label,
        "count": len(probabilities),
        "brier": round(_brier(probabilities, outcomes), 6),
        "logloss": round(_logloss(probabilities, outcomes), 6),
        "accuracy": round(_accuracy(probabilities, outcomes), 6),
        "probability_mean": round(float(np.mean(probabilities)), 6),
        "probability_std": round(float(np.std(probabilities)), 6),
    }


def _compare_against_baselines(baselines: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if "model" not in baselines:
        return {"status": "model_unavailable", "comparisons": {}, "reasons": ["model baseline unavailable"]}
    model = baselines["model"]
    comparisons: Dict[str, Any] = {}
    for label, metrics in baselines.items():
        if label == "model":
            continue
        comparisons[label] = {
            "delta_brier": round(float(model["brier"] - metrics["brier"]), 6),
            "beats_brier": bool(model["brier"] < metrics["brier"]),
            "delta_logloss": round(float(model["logloss"] - metrics["logloss"]), 6),
            "beats_logloss": bool(model["logloss"] < metrics["logloss"]),
            "delta_accuracy": round(float(model["accuracy"] - metrics["accuracy"]), 6),
            "beats_accuracy": bool(model["accuracy"] > metrics["accuracy"]),
        }
    return {"status": "ok", "comparisons": comparisons, "reasons": []}


def _quality_gate(
    *,
    settled_count: int,
    baselines: Dict[str, Dict[str, Any]],
    comparison: Dict[str, Any],
    skipped_baselines: List[str],
) -> Dict[str, Any]:
    reasons: List[str] = []
    warnings: List[str] = []
    if settled_count < MIN_PRODUCTION_SETTLED_SAMPLES:
        reasons.append(f"settled_sample_count_below_{MIN_PRODUCTION_SETTLED_SAMPLES}")
    if "model" not in baselines:
        reasons.append("model_probability_unavailable")
    for baseline in REQUIRED_PROMOTION_BASELINES:
        if baseline not in baselines:
            reasons.append(f"required_baseline_unavailable:{baseline}")
    comparisons = comparison.get("comparisons", {}) if isinstance(comparison, dict) else {}
    for baseline in REQUIRED_PROMOTION_BASELINES:
        item = comparisons.get(baseline)
        if not isinstance(item, dict):
            continue
        if not item.get("beats_brier", False):
            reasons.append(f"model_does_not_beat_{baseline}_brier")
        if not item.get("beats_logloss", False):
            reasons.append(f"model_does_not_beat_{baseline}_logloss")
    if skipped_baselines:
        warnings.append("one_or_more_baselines_skipped")
    promotion_allowed = len(reasons) == 0
    return {
        "status": "passed" if promotion_allowed else "blocked",
        "promotion_allowed": promotion_allowed,
        "production_model_replacement_allowed": False,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "minimum_production_settled_samples": MIN_PRODUCTION_SETTLED_SAMPLES,
        "required_promotion_baselines": REQUIRED_PROMOTION_BASELINES,
        "reasons": sorted(set(reasons)),
        "warnings": sorted(set(warnings)),
    }


def build_report() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = _utc_now()
    _, prediction_status = _safe_read_json(PREDICTION_JSON)
    snapshots, snapshots_status = _safe_read_csv(SNAPSHOTS_CSV)
    finalized, finalized_status = _safe_read_csv(FINALIZED_CSV)
    _, market_status = _safe_read_csv(MARKET_ODDS_CSV)
    input_files = {
        "prediction": prediction_status,
        "prediction_snapshots": snapshots_status,
        "finalized_games": finalized_status,
        "market_odds_history": market_status,
    }
    settled = _prepare_settled_rows(snapshots, finalized)
    skipped_baselines: List[str] = []
    recommendations: List[str] = []

    if settled.empty:
        report = {
            "generated_at": generated_at,
            "report_type": "baseline_comparison_report",
            "status": "insufficient_samples",
            "input_files": input_files,
            "settled_prediction_count": 0,
            "baselines": {},
            "comparison": {},
            "quality_gate": {
                "status": "blocked",
                "promotion_allowed": False,
                "reasons": ["no_settled_prediction_snapshots_with_home_win"],
            },
            "skipped_baselines": [
                "model",
                "market_no_vig",
                "elo",
                "constant_50",
                "home_historical_rate",
                "log5",
            ],
            "recommendations": ["No settled prediction snapshots with home_win were available."],
            "promotion_allowed": False,
            "production_model_replacement_allowed": False,
            "live_betting_allowed": False,
            "automated_wagering_allowed": False,
        }
        OUTPUT_JSON.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=True), encoding="utf-8")
        return report

    settled = settled.copy()
    settled["model_probability"] = settled.apply(
        lambda row: _prob_from_row(
            row,
            [
                "predicted_home_win_pct",
                "premarket_model_home_prob",
                "displayed_home_win_pct",
                "model_prob",
                "home_win_probability",
            ],
        ),
        axis=1,
    )

    if "market_no_vig_home_prob" in settled.columns:
        settled["market_no_vig_probability"] = settled["market_no_vig_home_prob"].apply(_clip_probability)
    else:
        settled["market_no_vig_probability"] = None
    if settled["market_no_vig_probability"].isna().all():
        if {"home_moneyline_odds", "away_moneyline_odds"}.issubset(settled.columns):
            settled["market_no_vig_probability"] = settled.apply(
                lambda row: _no_vig_home_probability(row.get("home_moneyline_odds"), row.get("away_moneyline_odds")),
                axis=1,
            )
        else:
            skipped_baselines.append("market_no_vig: missing market_no_vig_home_prob and moneyline odds columns")

    if "elo_diff" in settled.columns:
        settled["elo_probability"] = settled["elo_diff"].apply(_elo_probability_from_diff)
    else:
        settled["elo_probability"] = None
        skipped_baselines.append("elo: missing elo_diff")

    if "log5_prob" in settled.columns:
        settled["log5_probability"] = settled["log5_prob"].apply(_clip_probability)
    else:
        settled["log5_probability"] = None
        skipped_baselines.append("log5: missing log5_prob")

    settled["constant_50_probability"] = 0.5
    if finalized is not None and not finalized.empty and "home_win" in finalized.columns:
        home_rate_series = pd.to_numeric(finalized["home_win"], errors="coerce").dropna()
        home_rate_value = float(home_rate_series.mean()) if not home_rate_series.empty else float(settled["home_win"].mean())
    else:
        home_rate_value = float(settled["home_win"].mean())
    settled["home_historical_rate_probability"] = _clip_probability(home_rate_value)

    baseline_columns = {
        "model": "model_probability",
        "market_no_vig": "market_no_vig_probability",
        "elo": "elo_probability",
        "constant_50": "constant_50_probability",
        "home_historical_rate": "home_historical_rate_probability",
        "log5": "log5_probability",
    }
    baselines: Dict[str, Dict[str, Any]] = {}
    for label, column in baseline_columns.items():
        metrics = _baseline_metrics(settled, column, label)
        if metrics is None:
            skipped_baselines.append(f"{label}: no valid probabilities")
            continue
        baselines[label] = metrics

    comparison = _compare_against_baselines(baselines)
    quality_gate = _quality_gate(
        settled_count=int(len(settled)),
        baselines=baselines,
        comparison=comparison,
        skipped_baselines=skipped_baselines,
    )
    status = "ok" if "model" in baselines else "insufficient_samples"
    if quality_gate["status"] == "blocked":
        status = "warning" if status == "ok" else status

    if len(settled) < MIN_PRODUCTION_SETTLED_SAMPLES:
        recommendations.append("Settled sample count is below 300; baseline comparisons are not production-reliable.")
    if "market_no_vig" not in baselines:
        recommendations.append("Market no-vig baseline unavailable; odds history or market probability mapping should be checked.")
    if not quality_gate.get("promotion_allowed", False):
        recommendations.append("Do not promote model until it beats required baselines on Brier and logloss.")

    report = {
        "generated_at": generated_at,
        "report_type": "baseline_comparison_report",
        "status": status,
        "input_files": input_files,
        "settled_prediction_count": int(len(settled)),
        "baselines": baselines,
        "comparison": comparison,
        "quality_gate": quality_gate,
        "skipped_baselines": sorted(set(skipped_baselines)),
        "recommendations": recommendations,
        "promotion_allowed": bool(quality_gate.get("promotion_allowed", False)),
        "production_model_replacement_allowed": False,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
    }
    OUTPUT_JSON.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    report = build_report()
    print(json.dumps({"status": report["status"], "output_path": str(OUTPUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
