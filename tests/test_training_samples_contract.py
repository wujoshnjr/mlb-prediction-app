from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.training_samples_builder import build_training_samples


PIPELINE_VERSION = "baseline_v2_clean"


def _write_snapshots(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_outcomes(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _base_snapshot(**overrides):
    row = {
        "snapshot_id": "baseline_v2_clean:1",
        "game_id": "1",
        "game_date": "2026-05-27",
        "snapshot_created_at": "2026-05-27T10:00:00Z",
        "start_time": "2026-05-27T12:00:00Z",
        "snapshot_valid": "True",
        "pipeline_version": PIPELINE_VERSION,
        "snapshot_policy": "first_seen_pregame",
        "home_team": "Dodgers",
        "away_team": "Giants",
        "market_no_vig_home_prob": "0.55",
        "premarket_model_home_prob": "0.57",
        "model_edge_home": "0.02",
        "home_win": "0",
        "home_score": "99",
        "away_score": "98",
        "settled_at": "2026-05-28T00:00:00Z",
        "actual_winner": "home",
    }
    row.update(overrides)
    return row


def _base_outcome(**overrides):
    row = {
        "game_id": "1",
        "game_date": "2026-05-27",
        "home_team": "Dodgers",
        "away_team": "Giants",
        "home_score": "5",
        "away_score": "3",
        "home_win": "1",
        "status": "Final",
    }
    row.update(overrides)
    return row


def test_pregame_snapshot_is_included_and_uses_outcome_home_win(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "prediction_snapshots.csv"
    outcome_path = tmp_path / "finalized_snapshot_outcomes.csv"
    output_path = tmp_path / "training_samples.csv"
    report_path = tmp_path / "training_samples_report.json"

    _write_snapshots(snapshot_path, [_base_snapshot(home_win="0")])
    _write_outcomes(outcome_path, [_base_outcome(home_win="1")])

    report = build_training_samples(
        snapshot_path=snapshot_path,
        finalized_snapshot_outcomes_path=outcome_path,
        output_path=output_path,
        report_path=report_path,
        pipeline_version=PIPELINE_VERSION,
    )

    assert report["status"] == "ok"
    assert report["clean_training_rows"] == 1

    frame = pd.read_csv(output_path, dtype=str)
    assert len(frame) == 1
    assert frame.iloc[0]["game_id"] == "1"
    assert frame.iloc[0]["home_win"] == "1"


def test_snapshot_at_or_after_start_time_is_excluded(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "prediction_snapshots.csv"
    outcome_path = tmp_path / "finalized_snapshot_outcomes.csv"
    output_path = tmp_path / "training_samples.csv"
    report_path = tmp_path / "training_samples_report.json"

    _write_snapshots(
        snapshot_path,
        [
            _base_snapshot(
                snapshot_created_at="2026-05-27T12:00:00Z",
                start_time="2026-05-27T12:00:00Z",
            )
        ],
    )
    _write_outcomes(outcome_path, [_base_outcome()])

    report = build_training_samples(
        snapshot_path=snapshot_path,
        finalized_snapshot_outcomes_path=outcome_path,
        output_path=output_path,
        report_path=report_path,
        pipeline_version=PIPELINE_VERSION,
    )

    assert report["dropped_post_start_rows"] == 1
    assert report["clean_training_rows"] == 0


def test_invalid_home_win_outcome_is_excluded(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "prediction_snapshots.csv"
    outcome_path = tmp_path / "finalized_snapshot_outcomes.csv"
    output_path = tmp_path / "training_samples.csv"
    report_path = tmp_path / "training_samples_report.json"

    _write_snapshots(
        snapshot_path,
        [
            _base_snapshot(game_id="1"),
            _base_snapshot(snapshot_id="baseline_v2_clean:2", game_id="2"),
        ],
    )
    _write_outcomes(
        outcome_path,
        [
            _base_outcome(game_id="1", home_win="1"),
            _base_outcome(game_id="2", home_win="2"),
        ],
    )

    report = build_training_samples(
        snapshot_path=snapshot_path,
        finalized_snapshot_outcomes_path=outcome_path,
        output_path=output_path,
        report_path=report_path,
        pipeline_version=PIPELINE_VERSION,
    )

    frame = pd.read_csv(output_path, dtype=str)
    assert report["clean_training_rows"] == 1
    assert list(frame["game_id"]) == ["1"]


def test_empty_game_id_is_excluded(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "prediction_snapshots.csv"
    outcome_path = tmp_path / "finalized_snapshot_outcomes.csv"
    output_path = tmp_path / "training_samples.csv"
    report_path = tmp_path / "training_samples_report.json"

    _write_snapshots(
        snapshot_path,
        [
            _base_snapshot(game_id=""),
            _base_snapshot(snapshot_id="baseline_v2_clean:2", game_id="2"),
        ],
    )
    _write_outcomes(outcome_path, [_base_outcome(game_id="2", home_win="0")])

    report = build_training_samples(
        snapshot_path=snapshot_path,
        finalized_snapshot_outcomes_path=outcome_path,
        output_path=output_path,
        report_path=report_path,
        pipeline_version=PIPELINE_VERSION,
    )

    frame = pd.read_csv(output_path, dtype=str)
    assert report["dropped_empty_game_id_rows"] == 1
    assert len(frame) == 1
    assert frame.iloc[0]["game_id"] == "2"


def test_leakage_columns_are_removed(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "prediction_snapshots.csv"
    outcome_path = tmp_path / "finalized_snapshot_outcomes.csv"
    output_path = tmp_path / "training_samples.csv"
    report_path = tmp_path / "training_samples_report.json"

    _write_snapshots(snapshot_path, [_base_snapshot()])
    _write_outcomes(outcome_path, [_base_outcome()])

    report = build_training_samples(
        snapshot_path=snapshot_path,
        finalized_snapshot_outcomes_path=outcome_path,
        output_path=output_path,
        report_path=report_path,
        pipeline_version=PIPELINE_VERSION,
    )

    frame = pd.read_csv(output_path, dtype=str)
    for column in ["home_score", "away_score", "settled_at", "actual_winner"]:
        assert column not in frame.columns

    assert "home_win" in frame.columns
    assert "home_score" in report["leakage_columns_removed"]


def test_pipeline_version_mismatch_is_excluded(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "prediction_snapshots.csv"
    outcome_path = tmp_path / "finalized_snapshot_outcomes.csv"
    output_path = tmp_path / "training_samples.csv"
    report_path = tmp_path / "training_samples_report.json"

    _write_snapshots(snapshot_path, [_base_snapshot(pipeline_version="legacy_pipeline")])
    _write_outcomes(outcome_path, [_base_outcome()])

    report = build_training_samples(
        snapshot_path=snapshot_path,
        finalized_snapshot_outcomes_path=outcome_path,
        output_path=output_path,
        report_path=report_path,
        pipeline_version=PIPELINE_VERSION,
    )

    assert report["dropped_pipeline_mismatch_rows"] == 1
    assert report["clean_training_rows"] == 0


def test_report_and_safety_flags_are_written(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "prediction_snapshots.csv"
    outcome_path = tmp_path / "finalized_snapshot_outcomes.csv"
    output_path = tmp_path / "training_samples.csv"
    report_path = tmp_path / "training_samples_report.json"

    _write_snapshots(snapshot_path, [_base_snapshot()])
    _write_outcomes(outcome_path, [_base_outcome()])

    report = build_training_samples(
        snapshot_path=snapshot_path,
        finalized_snapshot_outcomes_path=outcome_path,
        output_path=output_path,
        report_path=report_path,
        pipeline_version=PIPELINE_VERSION,
    )

    assert output_path.exists()
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["live_betting_allowed"] is False
    assert payload["automated_wagering_allowed"] is False
    assert payload["production_model_replacement_allowed"] is False
    assert report["live_betting_allowed"] is False
