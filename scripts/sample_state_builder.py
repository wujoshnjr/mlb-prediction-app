from __future__ import annotations

import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
REPORT_DIR = Path("report")

SNAPSHOT_PATH = DATA_DIR / "prediction_snapshots.csv"
FINALIZED_PATH = DATA_DIR / "finalized_games.csv"
FINALIZED_SNAPSHOT_OUTCOMES_PATH = DATA_DIR / "finalized_snapshot_outcomes.csv"
LINK_REPORT_PATH = REPORT_DIR / "settled_prediction_link_report.json"
FINALIZED_LINKAGE_DIAGNOSTIC_PATH = REPORT_DIR / "finalized_linkage_diagnostic_report.json"
ROLLING_WALKFORWARD_PATH = REPORT_DIR / "rolling_walkforward_evaluation.json"
TRAINING_STATUS_PATH = DATA_DIR / "training_status.json"
CALIBRATOR_PATH = DATA_DIR / "calibrator.pkl"

SAMPLE_STATE_PATH = DATA_DIR / "sample_state.json"
SAMPLE_STATE_REPORT_PATH = REPORT_DIR / "sample_state_report.json"

MIN_CLEAN_TRAIN_SAMPLES = 300
MIN_PROMOTION_SAMPLES = 500
MIN_WALKFORWARD_PREDICTIONS = 300
MIN_CALIBRATION_SAMPLES = 500


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}

    if isinstance(value, list):
        return [_json_safe(child) for child in value]

    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if pd.isna(value) if not isinstance(value, (dict, list, tuple, str, bool)) else False:
        return None

    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _json_safe(payload)
    path.write_text(
        json.dumps(
            safe_payload,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    status = {
        "path": str(path),
        "exists": path.exists(),
        "rows": 0,
        "error": "",
    }

    if not path.exists():
        status["error"] = "file_missing"
        return pd.DataFrame(), status

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        status["error"] = str(exc)
        return pd.DataFrame(), status

    status["rows"] = int(len(frame))
    return frame, status


def _read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    status = {
        "path": str(path),
        "exists": path.exists(),
        "error": "",
        "type": None,
    }

    if not path.exists():
        status["error"] = "file_missing"
        return None, status

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status["error"] = str(exc)
        return None, status

    status["type"] = type(payload).__name__

    if not isinstance(payload, dict):
        status["error"] = "json_not_object"
        return None, status

    return payload, status


def _load_pickle(path: Path) -> Tuple[Optional[Any], Dict[str, Any]]:
    status = {
        "path": str(path),
        "exists": path.exists(),
        "error": "",
        "type": None,
    }

    if not path.exists():
        status["error"] = "file_missing"
        return None, status

    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        status["error"] = str(exc)
        return None, status

    status["type"] = type(payload).__name__
    return payload, status


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
        if math.isfinite(parsed) and parsed.is_integer():
            return str(int(parsed))
    except Exception:
        pass

    return text


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None

        parsed = float(value)
        if not math.isfinite(parsed):
            return None

        return parsed
    except Exception:
        return None


def _to_int(value: Any) -> Optional[int]:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "valid", "ok"})


def _prepare_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    if "game_id" not in frame.columns:
        return pd.DataFrame()

    result = frame.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    result = result[result["game_id"] != ""].copy()

    if "snapshot_valid" in result.columns:
        result["_sample_state_valid"] = _to_bool_series(result["snapshot_valid"])
    else:
        result["_sample_state_valid"] = True

    # Pregame snapshots must never provide trusted outcomes.
    # Outcomes for sample_state must come only from finalized_games.csv.
    leakage_columns = [
        "home_win",
        "home_score",
        "away_score",
        "final_score",
        "home_final_score",
        "away_final_score",
        "settled_at",
        "actual_winner",
        "actual_result",
        "final_home_score",
        "final_away_score",
        "postgame_win_probability",
    ]
    existing_leakage_columns = [
        column for column in leakage_columns if column in result.columns
    ]
    if existing_leakage_columns:
        result = result.drop(columns=existing_leakage_columns)

    return result.reset_index(drop=True)
    

def _prepare_finalized(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    if "game_id" not in frame.columns:
        return pd.DataFrame()

    result = frame.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    result = result[result["game_id"] != ""].copy()

    if "home_win" not in result.columns:
        if {"home_score", "away_score"}.issubset(result.columns):
            home_score = pd.to_numeric(result["home_score"], errors="coerce")
            away_score = pd.to_numeric(result["away_score"], errors="coerce")
            result["home_win"] = (home_score > away_score).astype("Int64")
        else:
            result["home_win"] = pd.NA

    result["home_win"] = pd.to_numeric(result["home_win"], errors="coerce")
    result = result[result["home_win"].isin([0, 1])].copy()

    return result.drop_duplicates("game_id", keep="last").reset_index(drop=True)


def _combine_finalized_sources(
    finalized: pd.DataFrame,
    snapshot_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Combine canonical finalized_games and trusted snapshot-outcome cache."""
    frames = []

    prepared_finalized = _prepare_finalized(finalized)
    if not prepared_finalized.empty:
        frames.append(prepared_finalized)

    prepared_cache = _prepare_finalized(snapshot_outcomes)
    if not prepared_cache.empty:
        frames.append(prepared_cache)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["game_id"] = combined["game_id"].apply(_normalize_game_id)
    combined = combined[combined["game_id"] != ""].copy()

    return combined.drop_duplicates("game_id", keep="last").reset_index(drop=True)


def _extract_training_sample_count_from_artifact(artifact: Any) -> Optional[int]:
    if artifact is None:
        return None

    candidates = []

    if isinstance(artifact, dict):
        candidates.extend(
            [
                artifact.get("training_sample_count"),
                artifact.get("sample_count"),
                artifact.get("n_samples"),
            ]
        )

        metadata = artifact.get("metadata")
        if isinstance(metadata, dict):
            candidates.extend(
                [
                    metadata.get("training_sample_count"),
                    metadata.get("sample_count"),
                    metadata.get("n_samples"),
                ]
            )
    else:
        candidates.extend(
            [
                getattr(artifact, "training_sample_count", None),
                getattr(artifact, "sample_count", None),
                getattr(artifact, "n_samples", None),
            ]
        )

    for candidate in candidates:
        parsed = _to_int(candidate)
        if parsed is not None:
            return parsed

    return None


def _count_unique_games(frame: pd.DataFrame) -> int:
    if frame.empty or "game_id" not in frame.columns:
        return 0
    return int(frame["game_id"].dropna().astype(str).nunique())


def build_sample_state() -> Dict[str, Any]:
    errors = []
    warnings = []
    recommendations = []

    snapshots_raw, snapshot_status = _read_csv(SNAPSHOT_PATH)
    finalized_raw, finalized_status = _read_csv(FINALIZED_PATH)
    finalized_snapshot_outcomes_raw, finalized_snapshot_outcomes_status = _read_csv(
        FINALIZED_SNAPSHOT_OUTCOMES_PATH
    )
    link_report, link_status = _read_json(LINK_REPORT_PATH)
    finalized_linkage_report, finalized_linkage_status = _read_json(
        FINALIZED_LINKAGE_DIAGNOSTIC_PATH
    )
    rolling_report, rolling_status = _read_json(ROLLING_WALKFORWARD_PATH)
    training_status, training_status_file = _read_json(TRAINING_STATUS_PATH)
    calibrator, calibrator_status = _load_pickle(CALIBRATOR_PATH)

    if snapshot_status["error"]:
        errors.append(f"prediction_snapshots unavailable: {snapshot_status['error']}")

    if finalized_status["error"]:
        errors.append(f"finalized_games unavailable: {finalized_status['error']}")

    if link_status["error"]:
        warnings.append(f"settled_prediction_link_report unavailable: {link_status['error']}")

    if rolling_status["error"]:
        warnings.append(f"rolling_walkforward_evaluation unavailable: {rolling_status['error']}")

    if training_status_file["error"]:
        warnings.append(f"training_status unavailable: {training_status_file['error']}")

    snapshots = _prepare_snapshots(snapshots_raw)
    finalized = _combine_finalized_sources(
        finalized_raw,
        finalized_snapshot_outcomes_raw,
    )

    raw_snapshots = int(len(snapshots_raw))
    valid_snapshots = int(snapshots["_sample_state_valid"].sum()) if not snapshots.empty else 0
    finalized_games = _count_unique_games(finalized)

    settled_snapshots = 0
    clean_settled_snapshots = 0

    if not snapshots.empty and not finalized.empty:
        finalized_outcomes = finalized[["game_id", "home_win"]].copy()
        finalized_outcomes = finalized_outcomes.rename(
            columns={"home_win": "_final_home_win"}
        )

        joined = snapshots.merge(
            finalized_outcomes,
            on="game_id",
            how="inner",
        )

        settled_snapshots = int(len(joined))

        clean = joined[
            (joined["_sample_state_valid"] == True)
            & (
                pd.to_numeric(
                    joined["_final_home_win"],
                    errors="coerce",
                ).isin([0, 1])
            )
        ].copy()

        clean_settled_snapshots = int(len(clean))
        
    train_eligible_samples = clean_settled_snapshots

    linked_games = clean_settled_snapshots
    link_rate = (
        float(clean_settled_snapshots / valid_snapshots)
        if valid_snapshots > 0
        else 0.0
    )

    if isinstance(link_report, dict):
        legacy_linked_games = _to_int(
            link_report.get("linked_game_count", link_report.get("linked_games"))
        )
        legacy_link_rate = _to_float(link_report.get("link_rate"))

        if legacy_linked_games is not None and legacy_linked_games != linked_games:
            warnings.append(
                "Legacy settled_prediction_link_report linked count differs from "
                f"actual finalized join: legacy={legacy_linked_games}, actual={linked_games}."
            )

        if legacy_link_rate is not None and abs(float(legacy_link_rate) - float(link_rate)) > 0.001:
            warnings.append(
                "Legacy settled_prediction_link_report link_rate differs from "
                f"actual finalized join: legacy={legacy_link_rate}, actual={link_rate:.6f}."
            )
            
    walkforward_predictions = 0
    if isinstance(rolling_report, dict):
        walkforward_predictions = _to_int(rolling_report.get("total_oos_predictions")) or 0

    model_artifact_training_samples = _extract_training_sample_count_from_artifact(calibrator)

    if model_artifact_training_samples is None and isinstance(training_status, dict):
        model_artifact_training_samples = _to_int(training_status.get("sample_count"))

    trained = bool(training_status.get("trained", False)) if isinstance(training_status, dict) else False
    model_artifact_exists = bool(calibrator_status["exists"] and not calibrator_status["error"])

    training_allowed = train_eligible_samples >= MIN_CLEAN_TRAIN_SAMPLES
    promotion_sample_ready = train_eligible_samples >= MIN_PROMOTION_SAMPLES
    walkforward_ready = walkforward_predictions >= MIN_WALKFORWARD_PREDICTIONS
    calibration_sample_ready = clean_settled_snapshots >= MIN_CALIBRATION_SAMPLES

    if train_eligible_samples < MIN_CLEAN_TRAIN_SAMPLES:
        recommendations.append(
            f"Train-eligible samples are below threshold: "
            f"{train_eligible_samples} < {MIN_CLEAN_TRAIN_SAMPLES}."
        )

    if clean_settled_snapshots < MIN_PROMOTION_SAMPLES:
        recommendations.append(
            f"Promotion sample threshold not met: "
            f"{clean_settled_snapshots} < {MIN_PROMOTION_SAMPLES}."
        )

    if walkforward_predictions < MIN_WALKFORWARD_PREDICTIONS:
        recommendations.append(
            f"Rolling walk-forward predictions below threshold: "
            f"{walkforward_predictions} < {MIN_WALKFORWARD_PREDICTIONS}."
        )

    if raw_snapshots > 0 and settled_snapshots == 0:
        warnings.append(
            "No prediction snapshots currently join to finalized_games.csv. "
            "Training, calibration, and settled evidence will remain unavailable."
        )

    if isinstance(finalized_linkage_report, dict):
        warnings.append(
            "Finalized linkage diagnostic: "
            f"overlap_after={finalized_linkage_report.get('overlap_count_after')}, "
            f"api_final_written={finalized_linkage_report.get('api_final_written_count')}, "
            f"pending_not_final={finalized_linkage_report.get('pending_not_final_count')}, "
            f"api_not_found_or_failed={finalized_linkage_report.get('api_not_found_or_failed_count')}."
        )
        diagnostic_overlap = _to_int(finalized_linkage_report.get("overlap_count_after"))
        if diagnostic_overlap and diagnostic_overlap > 0 and clean_settled_snapshots == 0:
            errors.append(
                "Finalized linkage diagnostic reports positive overlap, but actual "
                "combined finalized outcome join is zero. This indicates stale report "
                "or finalized outcome persistence failure."
            )
        
    if model_artifact_exists and not trained:
        warnings.append(
            "calibrator.pkl exists, but training_status.trained is false. "
            "Sample state keeps trained=false to avoid overstating model readiness."
        )

    status = "ok" if not errors else "partial"

    state = {
        "generated_at": _utc_now(),
        "last_updated": _utc_now(),
        "status": status,
        "input_files": {
            "prediction_snapshots": snapshot_status,
            "finalized_games": finalized_status,
            "finalized_snapshot_outcomes": finalized_snapshot_outcomes_status,
            "settled_prediction_link_report": link_status,
            "finalized_linkage_diagnostic": finalized_linkage_status,
            "rolling_walkforward_evaluation": rolling_status,
            "training_status": training_status_file,
            "calibrator": calibrator_status,
        },
        "raw_snapshots": raw_snapshots,
        "valid_snapshots": valid_snapshots,
        "settled_snapshots": settled_snapshots,
        "clean_settled_snapshots": clean_settled_snapshots,
        "train_eligible_samples": train_eligible_samples,
        "model_artifact_training_samples": model_artifact_training_samples,
        "walkforward_predictions": walkforward_predictions,
        "finalized_games": finalized_games,
        "linked_games": linked_games,
        "link_rate": round(float(link_rate), 6),
        "minimum_clean_train_samples": MIN_CLEAN_TRAIN_SAMPLES,
        "minimum_promotion_samples": MIN_PROMOTION_SAMPLES,
        "minimum_walkforward_predictions": MIN_WALKFORWARD_PREDICTIONS,
        "trained": trained,
        "model_artifact_exists": model_artifact_exists,
        "training_allowed": training_allowed,
        "promotion_sample_ready": promotion_sample_ready,
        "walkforward_ready": walkforward_ready,
        "calibration_sample_ready": calibration_sample_ready,
        "live_betting_allowed": False,
        "shadow_live_allowed": False,
        "production_allowed": False,
        "errors": errors,
        "warnings": warnings,
        "recommendations": recommendations,
    }

    return state


def main() -> None:
    state = build_sample_state()

    _write_json(SAMPLE_STATE_PATH, state)
    _write_json(SAMPLE_STATE_REPORT_PATH, state)

    print(json.dumps(_json_safe(state), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
