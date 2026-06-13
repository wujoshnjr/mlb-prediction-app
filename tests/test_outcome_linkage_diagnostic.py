import json

import pandas as pd
import pytest

from scripts.outcome_linkage_diagnostic import (
    _json_safe,
    compute_report,
    generate_report,
    safe_json_dump,
)


def test_missing_files_partial_no_crash(tmp_path, monkeypatch):
    import scripts.outcome_linkage_diagnostic as diag

    snapshot_path = tmp_path / "missing_prediction_snapshots.csv"
    outcome_path = tmp_path / "missing_finalized_snapshot_outcomes.csv"
    report_path = tmp_path / "outcome_linkage_diagnostic.json"

    monkeypatch.setattr(diag, "SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(diag, "OUTCOME_PATH", outcome_path)
    monkeypatch.setattr(diag, "REPORT_PATH", report_path)

    report = generate_report()

    assert report["status"] == "partial"
    assert report["input_files"]["snapshots"]["exists"] is False
    assert report["input_files"]["outcomes"]["exists"] is False
    assert report["live_betting_allowed"] is False
    assert report["automated_wagering_allowed"] is False
    assert report["production_model_replacement_allowed"] is False
    assert report_path.exists()


def test_basic_overlap_by_game_id():
    snapshots = pd.DataFrame({"game_id": ["1", "2", "3"]})
    outcomes = pd.DataFrame({"game_id": ["1", "2"], "home_win": [1, 0]})

    report = compute_report(snapshots, outcomes)

    assert report["status"] == "ok"
    assert report["overlap_game_count"] == 2
    assert report["snapshot_game_count"] == 3
    assert report["outcome_game_count"] == 2
    assert report["overlap_rate_vs_snapshots"] == pytest.approx(2 / 3)
    assert report["overlap_rate_vs_outcomes"] == 1.0


def test_numeric_string_game_id_normalization_join():
    snapshots = pd.DataFrame(
        {
            "game_id": ["123.0", " 456 "],
            "snapshot_valid": ["valid", "valid"],
        }
    )
    outcomes = pd.DataFrame({"game_id": [123, 456], "home_win": [1, 0]})

    report = compute_report(snapshots, outcomes)

    assert report["status"] == "ok"
    assert report["overlap_game_count"] == 2


def test_alphanumeric_game_id_is_preserved():
    snapshots = pd.DataFrame(
        {
            "game_id": ["mlb-2026-abc"],
            "snapshot_valid": ["valid"],
        }
    )
    outcomes = pd.DataFrame({"game_id": ["mlb-2026-abc"], "home_win": [1]})

    report = compute_report(snapshots, outcomes)

    assert report["status"] == "ok"
    assert report["snapshot_game_count"] == 1
    assert report["outcome_game_count"] == 1
    assert report["overlap_game_count"] == 1


def test_pipeline_version_mismatch_rows_counted():
    snapshots = pd.DataFrame(
        {
            "game_id": ["1", "2", "3"],
            "pipeline_version": ["baseline_v2_clean", "old_version", None],
            "snapshot_valid": ["valid", "valid", "valid"],
        }
    )
    outcomes = pd.DataFrame({"game_id": ["1", "2"], "home_win": [1, 0]})

    report = compute_report(snapshots, outcomes)

    assert report["dropped_pipeline_mismatch_rows"] == 1
    assert report["valid_snapshot_rows"] == 2
    assert report["overlap_game_count"] == 1
    assert any(
        "pipeline_version filter" in recommendation
        for recommendation in report["recommendations"]
    )


def test_snapshot_valid_false_rows_are_filtered():
    snapshots = pd.DataFrame(
        {
            "game_id": ["1", "2", "3"],
            "snapshot_valid": ["valid", 0, "invalid"],
        }
    )
    outcomes = pd.DataFrame({"game_id": ["1", "3"], "home_win": [1, 0]})

    report = compute_report(snapshots, outcomes)

    assert report["valid_snapshot_rows"] == 1
    assert report["snapshot_valid_rows"] == 1
    assert report["snapshot_invalid_rows"] == 2
    assert report["snapshot_game_count"] == 1
    assert report["overlap_game_count"] == 1


def test_outcome_can_derive_home_win_from_scores():
    snapshots = pd.DataFrame({"game_id": ["1", "2", "3"]})
    outcomes = pd.DataFrame(
        {
            "game_id": ["1", "2", "3"],
            "home_score": [5, 3, 2],
            "away_score": [3, 3, 4],
        }
    )

    report = compute_report(snapshots, outcomes)

    assert report["valid_outcome_rows"] == 2
    assert report["outcome_game_count"] == 2
    assert report["overlap_game_count"] == 2


def test_no_overlap_adds_recommendation():
    snapshots = pd.DataFrame({"game_id": ["1", "2"]})
    outcomes = pd.DataFrame({"game_id": ["3", "4"], "home_win": [1, 0]})

    report = compute_report(snapshots, outcomes)

    assert report["status"] == "partial"
    assert report["overlap_game_count"] == 0
    assert any(
        "No game_id overlap" in recommendation
        for recommendation in report["recommendations"]
    )


def test_missing_outcome_snapshot_examples_capped_at_10():
    snapshots = pd.DataFrame({"game_id": [str(i) for i in range(20)]})
    outcomes = pd.DataFrame({"game_id": [str(i) for i in range(5)], "home_win": [1] * 5})

    report = compute_report(snapshots, outcomes)

    assert len(report["missing_outcome_snapshot_examples"]) == 10
    assert len(report["outcome_without_snapshot_examples"]) == 0


def test_outcome_without_snapshot_examples_capped_at_10():
    snapshots = pd.DataFrame({"game_id": ["1"]})
    outcomes = pd.DataFrame({"game_id": [str(i) for i in range(20)], "home_win": [1] * 20})

    report = compute_report(snapshots, outcomes)

    assert len(report["outcome_without_snapshot_examples"]) == 10


def test_json_safe_output_contains_no_nan(tmp_path):
    data = {
        "value": float("nan"),
        "positive_inf": float("inf"),
        "negative_inf": float("-inf"),
        "nested": [1, float("nan"), 3],
        "frame": pd.DataFrame({"a": [1, float("nan")]}),
    }

    safe = _json_safe(data)
    dumped = json.dumps(safe, allow_nan=False)

    assert "NaN" not in dumped
    assert "Infinity" not in dumped
    assert "null" in dumped

    output_path = tmp_path / "safe.json"
    safe_json_dump(data, output_path)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["value"] is None
    assert loaded["positive_inf"] is None
    assert loaded["negative_inf"] is None


def test_safety_flags_are_false():
    report = compute_report(pd.DataFrame(), pd.DataFrame())

    assert report["live_betting_allowed"] is False
    assert report["automated_wagering_allowed"] is False
    assert report["production_model_replacement_allowed"] is False


def test_date_min_max_are_populated_when_dates_exist():
    snapshots = pd.DataFrame(
        {
            "game_id": ["1", "2"],
            "game_date": ["2023-10-01", "2023-10-05"],
        }
    )
    outcomes = pd.DataFrame(
        {
            "game_id": ["1", "2"],
            "home_win": [1, 0],
            "finalized_at": ["2023-10-03T00:00:00", "2023-10-06T00:00:00"],
        }
    )

    report = compute_report(snapshots, outcomes)

    assert report["snapshot_game_date_min"] is not None
    assert report["snapshot_game_date_max"] is not None
    assert report["outcome_game_date_min"] is not None
    assert report["outcome_game_date_max"] is not None


def test_read_error_marks_report_failed():
    snapshots = pd.DataFrame({"game_id": ["1"]})
    outcomes = pd.DataFrame({"game_id": ["1"], "home_win": [1]})

    report = compute_report(
        snapshots,
        outcomes,
        snapshot_status={
            "path": "data/prediction_snapshots.csv",
            "exists": True,
            "rows": 0,
            "error": "bad csv",
        },
        outcome_status={
            "path": "data/finalized_snapshot_outcomes.csv",
            "exists": True,
            "rows": 1,
            "error": "",
        },
    )

    assert report["status"] == "failed"
    assert "snapshot read error: bad csv" in report["errors"]


def test_missing_game_id_column_marks_failed():
    snapshots = pd.DataFrame({"not_game_id": ["1"]})
    outcomes = pd.DataFrame({"game_id": ["1"], "home_win": [1]})

    report = compute_report(snapshots, outcomes)

    assert report["status"] == "failed"
    assert "Snapshot file missing 'game_id' column." in report["errors"]
