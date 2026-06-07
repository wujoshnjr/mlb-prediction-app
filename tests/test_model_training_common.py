from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.model_training_common import (
    build_feature_matrix,
    build_training_frame,
    classification_metrics,
    expected_calibration_error,
    time_ordered_split,
)


def _write_fixture(tmp_path: Path, rows: int = 12) -> tuple[Path, Path]:
    snapshot_rows = []
    finalized_rows = []

    for i in range(rows):
        game_id = 1000 + i
        snapshot_rows.append(
            {
                "game_id": str(game_id),
                "snapshot_created_at": f"2026-05-{1 + i:02d}T12:00:00Z",
                "pipeline_version": "baseline_v2_clean",
                "snapshot_valid": "true",
                "elo_diff": float(i),
                "bt_strength_diff": float(i % 3),
                "sp_era_diff": float(i % 5),
                "home_win": 999,
                "home_score": 10,
                "away_score": 2,
            }
        )
        finalized_rows.append(
            {
                "game_id": float(game_id),
                "home_win": int(i % 2),
                "home_score": 5 + (i % 2),
                "away_score": 4,
            }
        )

    snapshot_path = tmp_path / "prediction_snapshots.csv"
    finalized_path = tmp_path / "finalized_games.csv"
    pd.DataFrame(snapshot_rows).to_csv(snapshot_path, index=False)
    pd.DataFrame(finalized_rows).to_csv(finalized_path, index=False)
    return snapshot_path, finalized_path


def test_missing_snapshots_does_not_crash(tmp_path: Path) -> None:
    finalized = tmp_path / "finalized_games.csv"
    pd.DataFrame([{"game_id": "1", "home_win": 1}]).to_csv(finalized, index=False)

    result = build_training_frame(
        snapshot_path=tmp_path / "missing.csv",
        finalized_path=finalized,
    )

    assert result["ok"] is False
    assert result["skipped"] is True
    assert "unavailable" in result["skip_reason"]


def test_missing_finalized_does_not_crash(tmp_path: Path) -> None:
    snapshots = tmp_path / "prediction_snapshots.csv"
    pd.DataFrame([{"game_id": "1", "snapshot_valid": "true"}]).to_csv(snapshots, index=False)

    result = build_training_frame(
        snapshot_path=snapshots,
        finalized_path=tmp_path / "missing.csv",
    )

    assert result["ok"] is False
    assert result["skipped"] is True
    assert "unavailable" in result["skip_reason"]


def test_leakage_columns_do_not_enter_feature_matrix(tmp_path: Path) -> None:
    snapshot_path, finalized_path = _write_fixture(tmp_path, rows=12)
    result = build_training_frame(snapshot_path=snapshot_path, finalized_path=finalized_path)

    assert result["ok"] is True
    frame = result["frame"]

    features = build_feature_matrix(
        frame,
        base_features=["elo_diff", "home_score", "home_win"],
    )

    assert features["ok"] is True
    assert "home_score" not in features["features_used"]
    assert "home_win" not in features["features_used"]
    assert any("leakage" in warning.lower() for warning in features["warnings"])


def test_time_split_keeps_order(tmp_path: Path) -> None:
    snapshot_path, finalized_path = _write_fixture(tmp_path, rows=20)
    result = build_training_frame(snapshot_path=snapshot_path, finalized_path=finalized_path)
    features = build_feature_matrix(result["frame"], base_features=["elo_diff"])

    split = time_ordered_split(
        features["X"],
        features["y"],
        features["frame"],
        min_train_samples=5,
        min_calibration_samples=3,
        min_validation_samples=3,
    )

    assert split["ok"] is True
    train_ids = split["frame_train"]["game_id"].tolist()
    validation_ids = split["frame_validation"]["game_id"].tolist()
    assert max(train_ids) < min(validation_ids)


def test_single_target_class_metrics_do_not_crash() -> None:
    metrics = classification_metrics([1, 1, 1], [0.6, 0.7, 0.8])

    assert metrics["ok"] is True
    assert metrics["auc"] is None
    assert metrics["brier"] is not None
    assert metrics["logloss"] is not None
    assert any("auc unavailable" in warning for warning in metrics["warnings"])


def test_ece_function_returns_value() -> None:
    ece = expected_calibration_error([0, 1, 1, 0], [0.2, 0.8, 0.7, 0.3])
    assert ece is not None
    assert 0.0 <= ece <= 1.0


def test_missing_feature_columns_are_filled_with_nan(tmp_path: Path) -> None:
    snapshot_path, finalized_path = _write_fixture(tmp_path, rows=12)
    result = build_training_frame(snapshot_path=snapshot_path, finalized_path=finalized_path)
    features = build_feature_matrix(result["frame"], base_features=["elo_diff", "missing_feature"])

    assert features["ok"] is True
    assert "missing_feature" in features["features_used"]
    missing_index = features["features_used"].index("missing_feature")
    assert np.isnan(features["X"][:, missing_index]).all()


def test_game_id_join_normalizes_float_ids(tmp_path: Path) -> None:
    snapshot_path, finalized_path = _write_fixture(tmp_path, rows=12)
    result = build_training_frame(snapshot_path=snapshot_path, finalized_path=finalized_path)

    assert result["ok"] is True
    assert result["sample_count"] == 12


def test_duplicated_game_id_keeps_latest_snapshot(tmp_path: Path) -> None:
    snapshots = pd.DataFrame(
        [
            {
                "game_id": "1",
                "snapshot_created_at": "2026-05-01T10:00:00Z",
                "pipeline_version": "baseline_v2_clean",
                "snapshot_valid": "true",
                "elo_diff": 1.0,
            },
            {
                "game_id": "1",
                "snapshot_created_at": "2026-05-01T12:00:00Z",
                "pipeline_version": "baseline_v2_clean",
                "snapshot_valid": "true",
                "elo_diff": 9.0,
            },
        ]
    )
    finalized = pd.DataFrame([{"game_id": "1", "home_win": 1}])

    snapshot_path = tmp_path / "prediction_snapshots.csv"
    finalized_path = tmp_path / "finalized_games.csv"
    snapshots.to_csv(snapshot_path, index=False)
    finalized.to_csv(finalized_path, index=False)

    result = build_training_frame(snapshot_path=snapshot_path, finalized_path=finalized_path)

    assert result["ok"] is True
    assert result["sample_count"] == 1
    assert float(result["frame"].iloc[0]["elo_diff"]) == 9.0
