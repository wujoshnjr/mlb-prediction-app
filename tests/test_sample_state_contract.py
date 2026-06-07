from __future__ import annotations

import json
from pathlib import Path

import pytest


SNAPSHOT_PATH = Path("data/prediction_snapshots.csv")
FINALIZED_PATH = Path("data/finalized_games.csv")
SAMPLE_STATE_PATH = Path("data/sample_state.json")
SAMPLE_STATE_REPORT_PATH = Path("report/sample_state_report.json")


def test_sample_state_contract() -> None:
    if not SNAPSHOT_PATH.exists():
        pytest.skip("data/prediction_snapshots.csv not found")

    if not FINALIZED_PATH.exists():
        pytest.skip("data/finalized_games.csv not found")

    from scripts.sample_state_builder import build_sample_state, main

    state = build_sample_state()

    required_keys = {
        "generated_at",
        "last_updated",
        "status",
        "raw_snapshots",
        "valid_snapshots",
        "settled_snapshots",
        "clean_settled_snapshots",
        "train_eligible_samples",
        "model_artifact_training_samples",
        "walkforward_predictions",
        "finalized_games",
        "linked_games",
        "link_rate",
        "minimum_clean_train_samples",
        "minimum_promotion_samples",
        "minimum_walkforward_predictions",
        "trained",
        "training_allowed",
        "promotion_sample_ready",
        "walkforward_ready",
        "calibration_sample_ready",
        "live_betting_allowed",
        "shadow_live_allowed",
        "production_allowed",
        "errors",
        "warnings",
        "recommendations",
    }

    assert required_keys.issubset(set(state.keys()))

    for key in (
        "raw_snapshots",
        "valid_snapshots",
        "settled_snapshots",
        "clean_settled_snapshots",
        "train_eligible_samples",
        "walkforward_predictions",
        "finalized_games",
        "linked_games",
    ):
        assert isinstance(state[key], int)
        assert state[key] >= 0

    assert state["minimum_clean_train_samples"] == 300
    assert state["minimum_promotion_samples"] == 500
    assert state["minimum_walkforward_predictions"] == 300

    assert 0.0 <= float(state["link_rate"]) <= 1.0

    assert state["live_betting_allowed"] is False
    assert state["shadow_live_allowed"] is False
    assert state["production_allowed"] is False

    assert isinstance(state["errors"], list)
    assert isinstance(state["warnings"], list)
    assert isinstance(state["recommendations"], list)

    main()

    assert SAMPLE_STATE_PATH.exists()
    assert SAMPLE_STATE_REPORT_PATH.exists()

    with SAMPLE_STATE_PATH.open("r", encoding="utf-8") as handle:
        saved_state = json.load(handle)

    with SAMPLE_STATE_REPORT_PATH.open("r", encoding="utf-8") as handle:
        saved_report = json.load(handle)

    assert saved_state["train_eligible_samples"] == state["train_eligible_samples"]
    assert saved_report["train_eligible_samples"] == state["train_eligible_samples"]
    assert saved_state["live_betting_allowed"] is False
    assert saved_report["production_allowed"] is False
