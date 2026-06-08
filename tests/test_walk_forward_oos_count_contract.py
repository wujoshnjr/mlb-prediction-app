from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_walk_forward_validation import build_report


def test_walk_forward_oos_count_is_per_model_not_row_sum(tmp_path: Path) -> None:
    snapshots = []
    finalized = []

    for i in range(100):
        game_id = 5000 + i
        snapshots.append(
            {
                "game_id": str(game_id),
                "snapshot_created_at": f"2026-04-{(i % 28) + 1:02d}T12:00:00Z",
                "pipeline_version": "baseline_v2_clean",
                "snapshot_valid": "true",
                "elo_diff": float(i % 5),
                "bt_strength_diff": float(i % 3),
                "market_no_vig_home_prob": 0.50,
            }
        )
        finalized.append({"game_id": str(game_id), "home_win": int(i % 2)})

    snapshot_path = tmp_path / "prediction_snapshots.csv"
    finalized_path = tmp_path / "finalized_games.csv"
    report_path = tmp_path / "report" / "walk_forward_validation_report.json"
    predictions_path = tmp_path / "data" / "walk_forward_predictions.csv"

    pd.DataFrame(snapshots).to_csv(snapshot_path, index=False)
    pd.DataFrame(finalized).to_csv(finalized_path, index=False)

    report = build_report(
        snapshot_path=snapshot_path,
        finalized_path=finalized_path,
        report_path=report_path,
        predictions_path=predictions_path,
        minimum_train_samples=40,
        validation_window_size=10,
        step_size=10,
    )

    assert report_path.exists()
    assert predictions_path.exists()

    assert "model_oos_counts" in report
    assert "max_model_oos_predictions" in report
    assert report["max_model_oos_predictions"] <= report["unique_oos_games"]
    assert report["total_oos_predictions"] == report["unique_oos_games"]
