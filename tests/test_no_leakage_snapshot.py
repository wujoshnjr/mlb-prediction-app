from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


SNAPSHOTS_PATH = Path("data/prediction_snapshots.csv")


def test_snapshot_created_before_or_at_start_time() -> None:
    if not SNAPSHOTS_PATH.exists():
        pytest.skip("prediction_snapshots.csv not found")

    frame = pd.read_csv(SNAPSHOTS_PATH)
    if "snapshot_created_at" not in frame.columns or "start_time" not in frame.columns:
        pytest.skip("snapshot_created_at or start_time column missing")

    frame["snapshot_created_at"] = pd.to_datetime(
        frame["snapshot_created_at"],
        errors="coerce",
        utc=True,
    )
    frame["start_time"] = pd.to_datetime(
        frame["start_time"],
        errors="coerce",
        utc=True,
    )

    valid = frame.dropna(subset=["snapshot_created_at", "start_time"])
    if valid.empty:
        pytest.skip("No valid snapshot/start_time rows")

    violations = valid[valid["snapshot_created_at"] > valid["start_time"]]
    assert violations.empty, f"Found {len(violations)} leakage violations"
