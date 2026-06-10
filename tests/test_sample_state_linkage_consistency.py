from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import scripts.sample_state_builder as sample_state_builder


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False)


def _patch_paths(monkeypatch, tmp_path: Path) -> dict[str, Path]:
    paths = {
        "snapshot": tmp_path / "prediction_snapshots.csv",
        "finalized": tmp_path / "finalized_games.csv",
        "finalized_snapshot_outcomes": tmp_path / "finalized_snapshot_outcomes.csv",
        "training_samples": tmp_path / "training_samples.csv",
        "training_status": tmp_path / "training_status.json",
        "diagnostic": tmp_path / "finalized_linkage_diagnostic_report.json",
        "link_report": tmp_path / "settled_prediction_link_report.json",
        "rolling": tmp_path / "rolling_walkforward_evaluation.json",
        "calibrator": tmp_path / "calibrator.pkl",
        "sample_state": tmp_path / "sample_state.json",
        "sample_state_report": tmp_path / "sample_state_report.json",
        "model_artifact_status": tmp_path / "model_artifact_status.json",
    }

    monkeypatch.setattr(sample_state_builder, "SNAPSHOT_PATH", paths["snapshot"], raising=False)
    monkeypatch.setattr(sample_state_builder, "FINALIZED_PATH", paths["finalized"], raising=False)
    monkeypatch.setattr(
        sample_state_builder,
        "FINALIZED_SNAPSHOT_OUTCOMES_PATH",
        paths["finalized_snapshot_outcomes"],
        raising=False,
    )
    monkeypatch.setattr(
        sample_state_builder,
        "TRAINING_SAMPLES_PATH",
        paths["training_samples"],
        raising=False,
    )
    monkeypatch.setattr(
        sample_state_builder,
        "TRAINING_STATUS_PATH",
        paths["training_status"],
        raising=False,
    )
    monkeypatch.setattr(
        sample_state_builder,
        "FINALIZED_LINKAGE_DIAGNOSTIC_PATH",
        paths["diagnostic"],
        raising=False,
    )
    monkeypatch.setattr(sample_state_builder, "LINK_REPORT_PATH", paths["link_report"], raising=False)
    monkeypatch.setattr(sample_state_builder, "ROLLING_WALKFORWARD_PATH", paths["rolling"], raising=False)
    monkeypatch.setattr(sample_state_builder, "CALIBRATOR_PATH", paths["calibrator"], raising=False)
    monkeypatch.setattr(sample_state_builder, "SAMPLE_STATE_PATH", paths["sample_state"], raising=False)
    monkeypatch.setattr(
        sample_state_builder,
        "SAMPLE_STATE_REPORT_PATH",
        paths["sample_state_report"],
        raising=False,
    )
    monkeypatch.setattr(
        sample_state_builder,
        "MODEL_ARTIFACT_STATUS_PATH",
        paths["model_artifact_status"],
        raising=False,
    )

    return paths


def test_positive_diagnostic_overlap_but_empty_cache_is_error(monkeypatch, tmp_path: Path) -> None:
    paths = _patch_paths(monkeypatch, tmp_path)

    _write_csv(
        paths["snapshot"],
        [
            {
                "game_id": "1",
                "snapshot_valid": "True",
                "pipeline_version": "baseline_v2_clean",
            }
        ],
    )
    _write_csv(paths["finalized"], [], columns=["game_id", "home_win"])
    _write_csv(paths["finalized_snapshot_outcomes"], [], columns=["game_id", "home_win"])
    _write_csv(paths["training_samples"], [], columns=["game_id", "home_win"])
    _write_json(paths["diagnostic"], {"overlap_count_after": 1})
    _write_json(paths["training_status"], {"trained": False, "sample_count": 0})

    state = sample_state_builder.build_sample_state()

    assert state["status"] == "partial"
    assert state["linked_games"] == 0
    assert state["errors"]


def test_training_samples_drive_clean_counts(monkeypatch, tmp_path: Path) -> None:
    paths = _patch_paths(monkeypatch, tmp_path)

    _write_csv(
        paths["snapshot"],
        [
            {
                "game_id": "1",
                "snapshot_valid": "True",
                "pipeline_version": "baseline_v2_clean",
            }
        ],
    )
    _write_csv(paths["finalized"], [], columns=["game_id", "home_win"])
    _write_csv(paths["finalized_snapshot_outcomes"], [{"game_id": "1", "home_win": "1"}])
    _write_csv(paths["training_samples"], [{"game_id": "1", "home_win": "1"}])
    _write_json(paths["diagnostic"], {"overlap_count_after": 1})
    _write_json(paths["training_status"], {"trained": False, "sample_count": 1})

    state = sample_state_builder.build_sample_state()

    assert state["linked_games"] == 1
    assert state["clean_settled_snapshots"] == 1
    assert state["train_eligible_samples"] == 1
    assert state["clean_settled_snapshots"] == state["train_eligible_samples"]


def test_trained_status_sample_mismatch_is_error(monkeypatch, tmp_path: Path) -> None:
    paths = _patch_paths(monkeypatch, tmp_path)

    _write_csv(paths["snapshot"], [{"game_id": "1", "snapshot_valid": "True"}])
    _write_csv(paths["finalized"], [], columns=["game_id", "home_win"])
    _write_csv(paths["finalized_snapshot_outcomes"], [{"game_id": "1", "home_win": "1"}])
    _write_csv(paths["training_samples"], [{"game_id": "1", "home_win": "1"}])
    _write_json(paths["diagnostic"], {"overlap_count_after": 1})
    _write_json(paths["training_status"], {"trained": True, "sample_count": 999})

    state = sample_state_builder.build_sample_state()

    assert state["status"] == "partial"
    assert state["errors"]


def test_sample_state_safety_flags_false(monkeypatch, tmp_path: Path) -> None:
    paths = _patch_paths(monkeypatch, tmp_path)

    _write_csv(paths["snapshot"], [{"game_id": "1", "snapshot_valid": "True"}])
    _write_csv(paths["finalized"], [], columns=["game_id", "home_win"])
    _write_csv(paths["finalized_snapshot_outcomes"], [{"game_id": "1", "home_win": "1"}])
    _write_csv(paths["training_samples"], [{"game_id": "1", "home_win": "1"}])
    _write_json(paths["diagnostic"], {"overlap_count_after": 1})
    _write_json(paths["training_status"], {"trained": False, "sample_count": 1})

    state = sample_state_builder.build_sample_state()

    assert state["live_betting_allowed"] is False
    assert state["shadow_live_allowed"] is False
    assert state["production_allowed"] is False
