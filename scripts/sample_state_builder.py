from __future__ import annotations

import json
import math
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
TRAINING_SAMPLES_PATH = DATA_DIR / "training_samples.csv"
TRAINING_STATUS_PATH = DATA_DIR / "training_status.json"
MODEL_ARTIFACT_STATUS_PATH = DATA_DIR / "model_artifact_status.json"
CALIBRATOR_PATH = DATA_DIR / "calibrator.pkl"

LINK_REPORT_PATH = REPORT_DIR / "settled_prediction_link_report.json"
FINALIZED_LINKAGE_DIAGNOSTIC_PATH = REPORT_DIR / "finalized_linkage_diagnostic_report.json"
ROLLING_WALKFORWARD_PATH = REPORT_DIR / "rolling_walkforward_evaluation.json"

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

    try:
        if pd.isna(value) and not isinstance(value, (dict, list, tuple, str, bool)):
            return None
    except Exception:
        pass

    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(payload),
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
        frame = pd.read_csv(path, dtype=str)
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
    if frame.empty or "game_id" not in frame.columns:
        return pd.DataFrame()

    result = frame.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    result = result[result["game_id"] != ""].copy()

    if "snapshot_valid" in result.columns:
        result["_sample_state_valid"] = _to_bool_series(result["snapshot_valid"])
    else:
        result["_sample_state_valid"] = True

    return result.reset_index(drop=True)


def _prepare_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "game_id" not in frame.columns:
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


def _count_unique_games(frame: pd.DataFrame) -> int:
    if frame.empty or "game_id" not in frame.columns:
        return 0
    return int(frame["game_id"].dropna().astype(str).map(_normalize_game_id).nunique())


def _training_sample_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    if "game_id" not in frame.columns:
        return 0

    valid = frame.copy()
    valid["game_id"] = valid["game_id"].apply(_normalize_game_id)
    valid = valid[valid["game_id"] != ""].copy()

    if "home_win" in valid.columns:
        home_win = pd.to_numeric(valid["home_win"], errors="coerce")
        valid = valid[home_win.isin([0, 1])].copy()

    return int(len(valid))


def _read_training_status_sample_count(training_status: Optional[Dict[str, Any]]) -> int:
    if not isinstance(training_status, dict):
        return 0
    return _to_int(training_status.get("sample_count")) or 0


def _read_model_artifact_valid(model_artifact_status: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(model_artifact_status, dict):
        return False
    return bool(model_artifact_status.get("valid") is True)


def _read_loaded_artifact_sample_count(model_artifact_status: Optional[Dict[str, Any]]) -> int:
    if not isinstance(model_artifact_status, dict):
        return 0

    for key in (
        "training_sample_count",
        "loaded_artifact_sample_count",
        "artifact_sample_count",
    ):
        parsed = _to_int(model_artifact_status.get(key))
        if parsed is not None:
            return parsed

    return 0


def build_sample_state() -> Dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    recommendations: list[str] = []

    snapshots_raw, snapshot_status = _read_csv(SNAPSHOT_PATH)
    finalized_raw, finalized_status = _read_csv(FINALIZED_PATH)
    finalized_snapshot_outcomes_raw, finalized_snapshot_outcomes_status = _read_csv(
        FINALIZED_SNAPSHOT_OUTCOMES_PATH
    )
    training_samples_raw, training_samples_status = _read_csv(TRAINING_SAMPLES_PATH)

    link_report, link_status = _read_json(LINK_REPORT_PATH)
    finalized_linkage_report, finalized_linkage_status = _read_json(
        FINALIZED_LINKAGE_DIAGNOSTIC_PATH
    )
    rolling_report, rolling_status = _read_json(ROLLING_WALKFORWARD_PATH)
    training_status, training_status_file = _read_json(TRAINING_STATUS_PATH)
    model_artifact_status, model_artifact_status_file = _read_json(
        MODEL_ARTIFACT_STATUS_PATH
    )

    if snapshot_status["error"]:
        warnings.append(f"prediction_snapshots unavailable: {snapshot_status['error']}")

    if finalized_status["error"]:
        warnings.append(f"finalized_games unavailable: {finalized_status['error']}")

    if finalized_snapshot_outcomes_status["error"]:
        warnings.append(
            "finalized_snapshot_outcomes unavailable: "
            f"{finalized_snapshot_outcomes_status['error']}"
        )

    if training_samples_status["error"]:
        warnings.append(
            f"training_samples unavailable: {training_samples_status['error']}"
        )
        recommendations.append("Run scripts/training_samples_builder.py before sample_state_builder.py.")

    if link_status["error"]:
        warnings.append(f"settled_prediction_link_report unavailable: {link_status['error']}")

    if rolling_status["error"]:
        warnings.append(f"rolling_walkforward_evaluation unavailable: {rolling_status['error']}")

    if training_status_file["error"]:
        warnings.append(f"training_status unavailable: {training_status_file['error']}")

    if model_artifact_status_file["error"]:
        warnings.append(
            f"model_artifact_status unavailable: {model_artifact_status_file['error']}"
        )

    snapshots = _prepare_snapshots(snapshots_raw)
    finalized_games_frame = _prepare_outcomes(finalized_raw)
    finalized_snapshot_outcomes = _prepare_outcomes(finalized_snapshot_outcomes_raw)

    raw_snapshots = int(len(snapshots_raw))
    valid_snapshots = int(snapshots["_sample_state_valid"].sum()) if not snapshots.empty else 0
    finalized_games = _count_unique_games(finalized_games_frame)
    finalized_snapshot_outcome_rows = int(len(finalized_snapshot_outcomes))
    linked_games = _count_unique_games(finalized_snapshot_outcomes)

    training_sample_count = _training_sample_count(training_samples_raw)

    settled_snapshots = training_sample_count
    clean_settled_snapshots = training_sample_count
    train_eligible_samples = training_sample_count
    clean_settled_sample_count = clean_settled_snapshots

    link_rate = (
        float(linked_games / valid_snapshots)
        if valid_snapshots > 0
        else 0.0
    )

    if linked_games > 0 and train_eligible_samples == 0:
        warnings.append(
            "Canonical finalized_snapshot_outcomes has linked games, but "
            "training_samples.csv has zero eligible rows."
        )
        recommendations.append("Rebuild data/training_samples.csv from canonical outcomes.")

    if isinstance(link_report, dict):
        legacy_linked_games = _to_int(
            link_report.get("linked_game_count", link_report.get("linked_games"))
        )
        legacy_link_rate = _to_float(link_report.get("link_rate"))

        if legacy_linked_games is not None and legacy_linked_games != linked_games:
            warnings.append(
                "Legacy settled_prediction_link_report linked count differs from "
                f"canonical finalized snapshot outcomes: legacy={legacy_linked_games}, "
                f"canonical={linked_games}."
            )

        if legacy_link_rate is not None and abs(float(legacy_link_rate) - float(link_rate)) > 0.001:
            warnings.append(
                "Legacy settled_prediction_link_report link_rate differs from "
                f"canonical finalized snapshot outcomes: legacy={legacy_link_rate}, "
                f"canonical={link_rate:.6f}."
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
        if diagnostic_overlap and diagnostic_overlap > 0 and linked_games == 0:
            errors.append(
                "Finalized linkage diagnostic reports positive overlap, but canonical "
                "finalized_snapshot_outcomes has zero linked games."
            )

    walkforward_predictions = 0
    if isinstance(rolling_report, dict):
        walkforward_predictions = _to_int(rolling_report.get("total_oos_predictions")) or 0

    training_status_sample_count = _read_training_status_sample_count(training_status)
    trained = bool(training_status.get("trained", False)) if isinstance(training_status, dict) else False

    if trained and training_status_sample_count != train_eligible_samples:
        errors.append(
            "training_status.sample_count does not match sample_state.train_eligible_samples: "
            f"training_status={training_status_sample_count}, "
            f"sample_state={train_eligible_samples}."
        )

    loaded_artifact_sample_count = _read_loaded_artifact_sample_count(model_artifact_status)
    model_artifact_valid = _read_model_artifact_valid(model_artifact_status)
    model_artifact_error = ""
    if isinstance(model_artifact_status, dict):
        model_artifact_error = str(model_artifact_status.get("error") or "")

    active_model_sample_count = (
        loaded_artifact_sample_count
        if (
            model_artifact_valid
            and trained
            and loaded_artifact_sample_count >= MIN_CLEAN_TRAIN_SAMPLES
        )
        else 0
    )

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

    if raw_snapshots > 0 and linked_games == 0:
        warnings.append(
            "No canonical finalized snapshot outcomes currently link to prediction snapshots. "
            "Training, calibration, and settled evidence will remain unavailable."
        )

    if not model_artifact_valid:
        warnings.append(
            "Model artifact is not valid for active ML use; prediction must use manual or market baseline."
        )

    status = "partial" if errors or training_samples_status["error"] else "ok"

    state = {
        "generated_at": _utc_now(),
        "last_updated": _utc_now(),
        "status": status,
        "current_stage": (
            "STAGE_2_BASELINE_TRAINING_READY"
            if train_eligible_samples > 0
            else "STAGE_1_EVIDENCE_LINKING"
        ),
        "input_files": {
            "prediction_snapshots": snapshot_status,
            "finalized_games": finalized_status,
            "finalized_snapshot_outcomes": finalized_snapshot_outcomes_status,
            "training_samples": training_samples_status,
            "settled_prediction_link_report": link_status,
            "finalized_linkage_diagnostic": finalized_linkage_status,
            "rolling_walkforward_evaluation": rolling_status,
            "training_status": training_status_file,
            "model_artifact_status": model_artifact_status_file,
        },
        "raw_snapshots": raw_snapshots,
        "valid_snapshots": valid_snapshots,
        "finalized_games": finalized_games,
        "finalized_snapshot_outcome_rows": finalized_snapshot_outcome_rows,
        "linked_games": linked_games,
        "settled_snapshots": settled_snapshots,
        "clean_settled_snapshots": clean_settled_snapshots,
        "clean_settled_sample_count": clean_settled_sample_count,
        "train_eligible_samples": train_eligible_samples,
        "training_status_sample_count": training_status_sample_count,
        "loaded_artifact_sample_count": loaded_artifact_sample_count,
        "active_model_sample_count": active_model_sample_count,
        "model_artifact_valid": model_artifact_valid,
        "model_artifact_error": model_artifact_error,
        "walkforward_predictions": walkforward_predictions,
        "link_rate": round(float(link_rate), 6),
        "minimum_clean_train_samples": MIN_CLEAN_TRAIN_SAMPLES,
        "minimum_promotion_samples": MIN_PROMOTION_SAMPLES,
        "minimum_walkforward_predictions": MIN_WALKFORWARD_PREDICTIONS,
        "trained": trained,
        "training_allowed": training_allowed,
        "promotion_sample_ready": promotion_sample_ready,
        "walkforward_ready": walkforward_ready,
        "calibration_sample_ready": calibration_sample_ready,
        "live_betting_allowed": False,
        "shadow_live_allowed": False,
        "production_allowed": False,
        "production_model_replacement_allowed": False,
        "errors": errors,
        "warnings": warnings,
        "recommendations": sorted(set(recommendations)),
    }

    return state


def main() -> None:
    state = build_sample_state()

    _write_json(SAMPLE_STATE_PATH, state)
    _write_json(SAMPLE_STATE_REPORT_PATH, state)

    print(json.dumps(_json_safe(state), indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
