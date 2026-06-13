from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import config
except ImportError:
    class config:
        PIPELINE_VERSION = "baseline_v2_clean"
        SNAPSHOT_STORE_FILE = "data/prediction_snapshots.csv"


SNAPSHOT_PATH = Path(
    str(getattr(config, "SNAPSHOT_STORE_FILE", "data/prediction_snapshots.csv"))
)
OUTCOME_PATH = Path("data/finalized_snapshot_outcomes.csv")
REPORT_PATH = Path("report/outcome_linkage_diagnostic.json")
PIPELINE_VERSION = str(getattr(config, "PIPELINE_VERSION", "baseline_v2_clean"))
REPORT_TYPE = "outcome_linkage_diagnostic_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe value with NaN/Infinity converted to None."""
    if value is None:
        return None

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    try:
        if pd.api.types.is_scalar(value) and pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, str):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))

    if isinstance(value, pd.Series):
        return _json_safe(value.tolist())

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())

    return str(value)


def safe_json_dump(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, ensure_ascii=False, allow_nan=False)


def read_csv_safe(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "rows": 0,
        "error": "",
    }

    if not path.exists():
        return pd.DataFrame(), status

    try:
        df = pd.read_csv(path)
        status["rows"] = int(len(df))
        return df, status
    except Exception as exc:
        status["error"] = str(exc)
        return pd.DataFrame(), status


def normalize_game_id(value: Any) -> str:
    """Normalize numeric and string game_id values without dropping alphanumeric IDs."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    clean = str(value).strip()
    if not clean or clean.lower() in {"nan", "none", "null"}:
        return ""

    try:
        numeric = float(clean)
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except (ValueError, TypeError):
        pass

    if clean.endswith(".0") and clean[:-2].isdigit():
        return clean[:-2]

    return clean


def bool_series(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "valid", "ok"}
    return values.isin(truthy)


def _date_min_max(frame: pd.DataFrame, columns: list[str]) -> tuple[str | None, str | None]:
    for column in columns:
        if column not in frame.columns:
            continue

        values = pd.to_datetime(frame[column], errors="coerce").dropna()
        if values.empty:
            continue

        return values.min().isoformat(), values.max().isoformat()

    return None, None


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}

    counts = frame[column].value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _examples(values: list | set | pd.Series, limit: int = 10) -> list[str]:
    cleaned = [str(value) for value in list(values) if str(value)]
    return sorted(cleaned)[:limit]


def _default_input_status(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": True,
        "rows": int(len(frame)),
        "error": "",
    }


def compute_report(
    snapshots: pd.DataFrame,
    outcomes: pd.DataFrame,
    snapshot_status: dict[str, Any] | None = None,
    outcome_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot_status = snapshot_status or _default_input_status(SNAPSHOT_PATH, snapshots)
    outcome_status = outcome_status or _default_input_status(OUTCOME_PATH, outcomes)

    report: dict[str, Any] = {
        "generated_at": _utc_now(),
        "status": "ok",
        "report_type": REPORT_TYPE,
        "pipeline_version": PIPELINE_VERSION,
        "snapshot_path": str(SNAPSHOT_PATH),
        "outcome_path": str(OUTCOME_PATH),
        "input_files": {
            "snapshots": snapshot_status,
            "outcomes": outcome_status,
        },
        "raw_snapshot_rows": int(len(snapshots)),
        "valid_snapshot_rows": 0,
        "raw_outcome_rows": int(len(outcomes)),
        "valid_outcome_rows": 0,
        "snapshot_game_count": 0,
        "outcome_game_count": 0,
        "overlap_game_count": 0,
        "overlap_rate_vs_snapshots": None,
        "overlap_rate_vs_outcomes": None,
        "snapshot_game_date_min": None,
        "snapshot_game_date_max": None,
        "outcome_game_date_min": None,
        "outcome_game_date_max": None,
        "snapshot_pipeline_versions": {},
        "dropped_pipeline_mismatch_rows": 0,
        "snapshot_valid_rows": 0,
        "snapshot_invalid_rows": 0,
        "missing_outcome_snapshot_examples": [],
        "outcome_without_snapshot_examples": [],
        "warnings": [],
        "errors": [],
        "recommendations": [],
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }

    snapshot_frame = snapshots.copy()
    outcome_frame = outcomes.copy()

    if "snapshot_valid" in snapshot_frame.columns:
        snapshot_valid_mask = bool_series(snapshot_frame["snapshot_valid"])
        report["snapshot_valid_rows"] = int(snapshot_valid_mask.sum())
        report["snapshot_invalid_rows"] = int(
            report["raw_snapshot_rows"] - report["snapshot_valid_rows"]
        )

    if "game_id" in snapshot_frame.columns:
        snapshot_frame["__norm_game_id"] = snapshot_frame["game_id"].apply(
            normalize_game_id
        )
    else:
        snapshot_frame["__norm_game_id"] = ""

    if "game_id" in outcome_frame.columns:
        outcome_frame["__norm_game_id"] = outcome_frame["game_id"].apply(
            normalize_game_id
        )
    else:
        outcome_frame["__norm_game_id"] = ""

    valid_snapshot_mask = snapshot_frame["__norm_game_id"] != ""

    if "pipeline_version" in snapshot_frame.columns:
        pipeline_mismatch = (
            snapshot_frame["pipeline_version"].notna()
            & (snapshot_frame["pipeline_version"] != PIPELINE_VERSION)
        )
        report["dropped_pipeline_mismatch_rows"] = int(pipeline_mismatch.sum())
        valid_snapshot_mask = valid_snapshot_mask & ~pipeline_mismatch

    if "snapshot_valid" in snapshot_frame.columns:
        valid_snapshot_mask = valid_snapshot_mask & bool_series(
            snapshot_frame["snapshot_valid"]
        )

    valid_snapshots = snapshot_frame[valid_snapshot_mask].copy()
    report["valid_snapshot_rows"] = int(len(valid_snapshots))

    valid_outcome_mask = outcome_frame["__norm_game_id"] != ""

    if "home_win" in outcome_frame.columns:
        outcome_frame["__home_win_num"] = pd.to_numeric(
            outcome_frame["home_win"],
            errors="coerce",
        )
    elif {"home_score", "away_score"}.issubset(outcome_frame.columns):
        home_score = pd.to_numeric(outcome_frame["home_score"], errors="coerce")
        away_score = pd.to_numeric(outcome_frame["away_score"], errors="coerce")

        outcome_frame["__home_win_num"] = np.nan
        score_valid = home_score.notna() & away_score.notna() & (home_score != away_score)

        outcome_frame.loc[score_valid, "__home_win_num"] = (
            home_score.loc[score_valid] > away_score.loc[score_valid]
        ).astype(int)
    else:
        outcome_frame["__home_win_num"] = np.nan

    valid_outcome_mask = valid_outcome_mask & outcome_frame["__home_win_num"].isin(
        [0, 1]
    )

    valid_outcomes = outcome_frame[valid_outcome_mask].copy()
    report["valid_outcome_rows"] = int(len(valid_outcomes))

    snapshot_game_ids = set(valid_snapshots["__norm_game_id"])
    outcome_game_ids = set(valid_outcomes["__norm_game_id"])
    overlap_game_ids = snapshot_game_ids & outcome_game_ids

    report["snapshot_game_count"] = int(len(snapshot_game_ids))
    report["outcome_game_count"] = int(len(outcome_game_ids))
    report["overlap_game_count"] = int(len(overlap_game_ids))

    report["overlap_rate_vs_snapshots"] = (
        len(overlap_game_ids) / report["snapshot_game_count"]
        if report["snapshot_game_count"]
        else None
    )
    report["overlap_rate_vs_outcomes"] = (
        len(overlap_game_ids) / report["outcome_game_count"]
        if report["outcome_game_count"]
        else None
    )

    report["snapshot_pipeline_versions"] = _value_counts(
        snapshot_frame,
        "pipeline_version",
    )

    snapshot_date_min, snapshot_date_max = _date_min_max(
        valid_snapshots,
        ["game_date", "snapshot_created_at", "prediction_created_at", "generated_at"],
    )
    outcome_date_min, outcome_date_max = _date_min_max(
        valid_outcomes,
        ["game_date", "finalized_at", "settled_at", "updated_at"],
    )

    report["snapshot_game_date_min"] = snapshot_date_min
    report["snapshot_game_date_max"] = snapshot_date_max
    report["outcome_game_date_min"] = outcome_date_min
    report["outcome_game_date_max"] = outcome_date_max

    report["missing_outcome_snapshot_examples"] = _examples(
        snapshot_game_ids - outcome_game_ids,
        limit=10,
    )
    report["outcome_without_snapshot_examples"] = _examples(
        outcome_game_ids - snapshot_game_ids,
        limit=10,
    )

    if report["snapshot_game_count"] == 0:
        report["recommendations"].append(
            "Prediction snapshots contain no valid game_id after filters."
        )

    if report["outcome_game_count"] == 0:
        report["recommendations"].append(
            "Finalized snapshot outcomes contain no valid finalized games."
        )

    if report["dropped_pipeline_mismatch_rows"] > 0:
        report["recommendations"].append(
            "Some prediction snapshots were dropped by pipeline_version filter; "
            "inspect snapshot_pipeline_versions."
        )

    if (
        report["overlap_game_count"] == 0
        and report["snapshot_game_count"] > 0
        and report["outcome_game_count"] > 0
    ):
        report["recommendations"].append(
            "No game_id overlap between prediction snapshots and finalized outcomes. "
            "Check game_id normalization, outcome refresh, and snapshot/outcome date ranges."
        )

    if (
        report["overlap_rate_vs_snapshots"] is not None
        and report["overlap_rate_vs_snapshots"] < 0.5
    ):
        report["recommendations"].append(
            "Less than half of valid snapshot games have finalized outcomes; "
            "model lab sample size may remain low."
        )

    snapshot_exists = bool(snapshot_status.get("exists"))
    outcome_exists = bool(outcome_status.get("exists"))
    missing_any = (not snapshot_exists) or (not outcome_exists)

    snapshot_read_error = str(snapshot_status.get("error") or "")
    outcome_read_error = str(outcome_status.get("error") or "")

    if snapshot_read_error:
        report["errors"].append(f"snapshot read error: {snapshot_read_error}")

    if outcome_read_error:
        report["errors"].append(f"outcome read error: {outcome_read_error}")

    if report["raw_snapshot_rows"] > 0 and "game_id" not in snapshots.columns:
        report["errors"].append("Snapshot file missing 'game_id' column.")
        report["status"] = "failed"
    elif report["raw_outcome_rows"] > 0 and "game_id" not in outcomes.columns:
        report["errors"].append("Outcome file missing 'game_id' column.")
        report["status"] = "failed"
    elif snapshot_read_error or outcome_read_error:
        report["status"] = "failed"
    elif missing_any:
        report["status"] = "partial"
        if not snapshot_exists and not outcome_exists:
            report["warnings"].append("Both snapshot and outcome files are missing.")
        else:
            report["warnings"].append("One of the required files is missing.")
    elif report["overlap_game_count"] == 0:
        report["status"] = "partial"
    else:
        report["status"] = "ok"

    return report


def generate_report() -> dict[str, Any]:
    snapshot_frame, snapshot_status = read_csv_safe(SNAPSHOT_PATH)
    outcome_frame, outcome_status = read_csv_safe(OUTCOME_PATH)

    report = compute_report(
        snapshot_frame,
        outcome_frame,
        snapshot_status=snapshot_status,
        outcome_status=outcome_status,
    )
    safe_json_dump(report, REPORT_PATH)
    return report


def main() -> int:
    report = generate_report()
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
