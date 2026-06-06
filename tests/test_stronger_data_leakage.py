"""
test_stronger_data_leakage.py

Stronger no-leakage tests for pregame prediction snapshots.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


SNAPSHOTS_PATH = Path("data/prediction_snapshots.csv")

POSTGAME_FIELDS = [
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


def _load_snapshots() -> pd.DataFrame:
    if not SNAPSHOTS_PATH.exists():
        pytest.skip("data/prediction_snapshots.csv not found")

    frame = pd.read_csv(SNAPSHOTS_PATH)
    if frame.empty:
        pytest.skip("prediction_snapshots.csv is empty")

    return frame


def test_snapshot_created_before_game_start() -> None:
    frame = _load_snapshots()

    required = {"snapshot_created_at", "start_time"}
    missing = required - set(frame.columns)
    if missing:
        pytest.skip(f"missing timestamp columns: {sorted(missing)}")

    snapshot_time = pd.to_datetime(frame["snapshot_created_at"], errors="coerce", utc=True)
    start_time = pd.to_datetime(frame["start_time"], errors="coerce", utc=True)

    valid = frame.assign(_snapshot_time=snapshot_time, _start_time=start_time).dropna(
        subset=["_snapshot_time", "_start_time"]
    )

    if valid.empty:
        pytest.skip("no valid snapshot/start timestamps")

    violations = valid[valid["_snapshot_time"] > valid["_start_time"]]
    assert violations.empty, (
        f"found {len(violations)} snapshots created after game start; possible leakage"
    )


def test_pregame_snapshots_do_not_contain_postgame_fields() -> None:
    frame = _load_snapshots()

    if "snapshot_created_at" not in frame.columns or "start_time" not in frame.columns:
        pytest.skip("missing timestamp columns")

    snapshot_time = pd.to_datetime(frame["snapshot_created_at"], errors="coerce", utc=True)
    start_time = pd.to_datetime(frame["start_time"], errors="coerce", utc=True)

    valid = frame.assign(_snapshot_time=snapshot_time, _start_time=start_time).dropna(
        subset=["_snapshot_time", "_start_time"]
    )

    if valid.empty:
        pytest.skip("no valid timestamp rows")

    pregame = valid[valid["_snapshot_time"] <= valid["_start_time"]]
    if pregame.empty:
        pytest.skip("no pregame snapshots found")

    present_fields = [field for field in POSTGAME_FIELDS if field in pregame.columns]
    if not present_fields:
        pytest.skip("no postgame fields present in snapshot schema")

    leaking = {}
    for field in present_fields:
        non_empty = pregame[field].notna()
        if non_empty.any():
            leaking[field] = int(non_empty.sum())

    assert not leaking, f"pregame snapshots contain non-empty postgame fields: {leaking}"


def test_home_win_not_filled_near_game_start() -> None:
    frame = _load_snapshots()

    if "home_win" not in frame.columns:
        pytest.skip("home_win column not present")

    if "snapshot_created_at" not in frame.columns or "start_time" not in frame.columns:
        pytest.skip("missing timestamp columns")

    snapshot_time = pd.to_datetime(frame["snapshot_created_at"], errors="coerce", utc=True)
    start_time = pd.to_datetime(frame["start_time"], errors="coerce", utc=True)

    valid = frame.assign(_snapshot_time=snapshot_time, _start_time=start_time).dropna(
        subset=["_snapshot_time", "_start_time"]
    )

    if valid.empty:
        pytest.skip("no valid timestamp rows")

    with_home_win = valid[valid["home_win"].notna()].copy()
    if with_home_win.empty:
        pytest.skip("home_win is empty in snapshots")

    hours_after_start = (
        with_home_win["_snapshot_time"] - with_home_win["_start_time"]
    ).dt.total_seconds() / 3600.0

    # If home_win is present pregame or within six hours after first pitch,
    # it is too early for a reliable finalized result and should be treated as leakage.
    suspicious = with_home_win[(hours_after_start >= -0.01) & (hours_after_start <= 6.0)]

    assert suspicious.empty, (
        f"home_win appears in snapshots within 6h of game start; count={len(suspicious)}"
    )
