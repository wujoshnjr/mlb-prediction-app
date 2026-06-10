from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.sample_state_builder import build_sample_state


def test_sample_state_uses_finalized_snapshot_outcome_cache(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    report_dir = tmp_path / "report"
    data_dir.mkdir()
    report_dir.mkdir()

    snapshots = pd.DataFrame(
        [
            {
                "game_id": "823295",
                "pipeline_version": "baseline_v2_clean",
                "snapshot_valid": "true",
                "displayed_home_win_pct": 0.55,
            }
        ]
    )
    finalized_games = pd.DataFrame(
        [
            {
                "game_id": "776906",
                "home_win": 1,
            }
        ]
    )
    outcome_cache = pd.DataFrame(
        [
            {
                "game_id": "823295",
                "game_date": "2026-05-27",
                "home_team": "Padres",
                "away_team": "Phillies",
                "home_score": 4,
                "away_score": 3,
                "home_win": 1,
                "status": "Final",
            }
        ]
    )
    training_samples = pd.DataFrame(
        [
            {
                "snapshot_id": "test-snapshot-1",
                "game_id": "823295",
                "game_date": "2026-05-27",
                "snapshot_created_at": "2026-05-27T12:00:00Z",
                "start_time": "2026-05-27T20:10:00Z",
                "home_team": "Padres",
                "away_team": "Phillies",
                "home_win": 1,
                "pipeline_version": "baseline_v2_clean",
                "snapshot_policy": "first_seen_pregame",
            }
        ]
    )

    snapshots.to_csv(data_dir / "prediction_snapshots.csv", index=False)
    finalized_games.to_csv(data_dir / "finalized_games.csv", index=False)
    outcome_cache.to_csv(data_dir / "finalized_snapshot_outcomes.csv", index=False)
    training_samples.to_csv(data_dir / "training_samples.csv", index=False)

    (report_dir / "settled_prediction_link_report.json").write_text("{}", encoding="utf-8")
    (report_dir / "finalized_linkage_diagnostic_report.json").write_text(
        '{"overlap_count_after": 1}',
        encoding="utf-8",
    )
    (report_dir / "rolling_walkforward_evaluation.json").write_text("{}", encoding="utf-8")
    (data_dir / "training_status.json").write_text(
        '{"trained": false, "sample_count": 1}',
        encoding="utf-8",
    )
    (data_dir / "model_artifact_status.json").write_text(
        '{"valid": false, "training_sample_count": 0, "reason": "missing"}',
        encoding="utf-8",
    )

    monkeypatch.setattr("scripts.sample_state_builder.SNAPSHOT_PATH", data_dir / "prediction_snapshots.csv")
    monkeypatch.setattr("scripts.sample_state_builder.FINALIZED_PATH", data_dir / "finalized_games.csv")
    monkeypatch.setattr(
        "scripts.sample_state_builder.FINALIZED_SNAPSHOT_OUTCOMES_PATH",
        data_dir / "finalized_snapshot_outcomes.csv",
    )
    monkeypatch.setattr(
        "scripts.sample_state_builder.TRAINING_SAMPLES_PATH",
        data_dir / "training_samples.csv",
    )
    monkeypatch.setattr(
        "scripts.sample_state_builder.LINK_REPORT_PATH",
        report_dir / "settled_prediction_link_report.json",
    )
    monkeypatch.setattr(
        "scripts.sample_state_builder.FINALIZED_LINKAGE_DIAGNOSTIC_PATH",
        report_dir / "finalized_linkage_diagnostic_report.json",
    )
    monkeypatch.setattr(
        "scripts.sample_state_builder.ROLLING_WALKFORWARD_PATH",
        report_dir / "rolling_walkforward_evaluation.json",
    )
    monkeypatch.setattr(
        "scripts.sample_state_builder.TRAINING_STATUS_PATH",
        data_dir / "training_status.json",
    )
    monkeypatch.setattr(
        "scripts.sample_state_builder.MODEL_ARTIFACT_STATUS_PATH",
        data_dir / "model_artifact_status.json",
    )
    monkeypatch.setattr(
        "scripts.sample_state_builder.CALIBRATOR_PATH",
        data_dir / "calibrator.pkl",
    )

    state = build_sample_state()

    assert state["clean_settled_snapshots"] == 1
    assert state["clean_settled_sample_count"] == 1
    assert state["train_eligible_samples"] == 1
    assert state["linked_games"] == 1
    assert state["link_rate"] == 1.0
