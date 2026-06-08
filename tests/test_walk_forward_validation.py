from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_walk_forward_validation import build_report


def _make_dataset(tmp_path: Path, rows: int = 100) -> tuple[Path, Path]:
    snapshots = []
    finalized = []

    for i in range(rows):
        gid = 10000 + i
        snapshots.append(
            {
                "game_id": str(gid),
                "snapshot_created_at": f"2026-03-{(i % 28) + 1:02d}T12:00:00Z",
                "pipeline_version": "baseline_v2_clean",
                "snapshot_valid": "true",
                "elo_diff": float(i % 10),
                "bt_strength_diff": float(i % 5),
                "market_no_vig_home_prob": 0.45 + (i % 10) / 100,
            }
        )
        finalized.append({"game_id": str(gid), "home_win": int(i % 2)})

    sp = tmp_path / "prediction_snapshots.csv"
    fp = tmp_path / "finalized_games.csv"
    pd.DataFrame(snapshots).to_csv(sp, index=False)
    pd.DataFrame(finalized).to_csv(fp, index=False)
    return sp, fp


def test_walk_forward_small_sample_skipped_report(tmp_path: Path) -> None:
    sp, fp = _make_dataset(tmp_path, rows=20)
    rp = tmp_path / "report" / "walk_forward_validation_report.json"
    pp = tmp_path / "data" / "walk_forward_predictions.csv"

    report = build_report(snapshot_path=sp, finalized_path=fp, report_path=rp, predictions_path=pp)

    assert rp.exists()
    assert pp.exists()
    assert report["skipped"] is True
    assert report["walkforward_ready"] is False


def test_walk_forward_outputs_oos_predictions(tmp_path: Path) -> None:
    sp, fp = _make_dataset(tmp_path, rows=100)
    rp = tmp_path / "report" / "walk_forward_validation_report.json"
    pp = tmp_path / "data" / "walk_forward_predictions.csv"

    report = build_report(
        snapshot_path=sp,
        finalized_path=fp,
        report_path=rp,
        predictions_path=pp,
        minimum_train_samples=40,
        validation_window_size=10,
        step_size=10,
    )

    assert rp.exists()
    assert pp.exists()
    assert report["skipped"] is False
    assert report["total_oos_predictions"] > 0

    predictions = pd.read_csv(pp)
    assert not predictions.empty
    assert {"game_id", "model_name", "predicted_prob", "actual_home_win"}.issubset(predictions.columns)
