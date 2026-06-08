from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.feature_promotion_report import (
    _feature_leakage_risk,
    build_report,
)


def _make_dataset(tmp_path: Path, rows: int = 40) -> tuple[Path, Path]:
    snapshot_rows = []
    finalized_rows = []

    for i in range(rows):
        game_id = 9000 + i
        snapshot_rows.append(
            {
                "game_id": str(game_id),
                "snapshot_created_at": f"2026-04-{(i % 28) + 1:02d}T12:00:00Z",
                "pipeline_version": "baseline_v2_clean",
                "snapshot_valid": "true",
                "elo_diff": float((i % 11) - 5),
                "bt_strength_diff": float((i % 7) - 3),
                "sp_era_diff": float((i % 5) - 2),
                "sp_fip_diff": float((i % 6) - 3),
                "sp_csw_diff": float((i % 4) - 2),
                "sp_stuff_plus_diff": float((i % 9) - 4),
                "k_pct_diff": float((i % 8) - 4),
                "bb_pct_diff": float((i % 10) - 5),
                "pitcher_rating_diff": float((i % 13) - 6),
                "bullpen_ip_diff": float((i % 3) - 1),
                "bullpen_availability_diff": float(i % 2),
                "dynamic_park_factor": float(1.0 + (i % 4) * 0.01),
                "winrate_diff": float((i % 10) / 100),
                "timezone_diff": float(i % 3),
                "sp_fip_diff_available": 1,
                "sp_csw_diff_available": 1,
                "sp_stuff_plus_diff_available": 1,
                "pitcher_advanced_available": 1,
                "bullpen_context_available": 1,
                "statcast_woba_available": 1,
                "top3_woba_available": 1,
                "weather_available": 1,
                "team_form_available": 1,
                "lineup_context_available": 1,
                "starter_context_available": 1,
                "odds_available": 1,
                "wind_effect": 1.0 if i < int(rows * 0.8) else None,
                "temp_effect": 0.5 if i % 2 == 0 else 0.0,
                "statcast_woba_diff": 0.01 * (i % 5),
            }
        )
        finalized_rows.append(
            {
                "game_id": str(game_id),
                "home_win": int(i % 2),
            }
        )

    snapshot_path = tmp_path / "prediction_snapshots.csv"
    finalized_path = tmp_path / "finalized_games.csv"

    pd.DataFrame(snapshot_rows).to_csv(snapshot_path, index=False)
    pd.DataFrame(finalized_rows).to_csv(finalized_path, index=False)

    return snapshot_path, finalized_path


def test_tracking_only_feature_missing_does_not_crash(tmp_path: Path) -> None:
    snapshot_path, finalized_path = _make_dataset(tmp_path, rows=40)
    report_path = tmp_path / "report" / "feature_promotion_report.json"

    report = build_report(
        snapshot_path=snapshot_path,
        finalized_path=finalized_path,
        report_path=report_path,
    )

    assert report_path.exists()
    assert isinstance(report, dict)
    assert report["feature_count"] > 0


def test_availability_missing_non_zero_rates_are_computed(tmp_path: Path) -> None:
    snapshot_path, finalized_path = _make_dataset(tmp_path, rows=40)
    report = build_report(
        snapshot_path=snapshot_path,
        finalized_path=finalized_path,
        report_path=tmp_path / "report" / "feature_promotion_report.json",
    )

    wind = [item for item in report["features"] if item["feature_name"] == "wind_effect"][0]

    assert 0.70 <= wind["availability_rate"] <= 0.90
    assert wind["missing_rate"] > 0
    assert wind["non_zero_rate"] > 0


def test_leakage_risk_feature_is_high() -> None:
    assert _feature_leakage_risk("home_score") == "high"
    assert _feature_leakage_risk("home_win") == "high"


def test_small_sample_never_ready_for_review(tmp_path: Path) -> None:
    snapshot_path, finalized_path = _make_dataset(tmp_path, rows=40)
    report = build_report(
        snapshot_path=snapshot_path,
        finalized_path=finalized_path,
        report_path=tmp_path / "report" / "feature_promotion_report.json",
    )

    assert report["sample_count"] == 40
    assert report["ready_for_review_count"] == 0
    assert all(
        item["recommended_status"] != "ready_for_review"
        for item in report["features"]
    )


def test_report_is_generated_when_inputs_missing(tmp_path: Path) -> None:
    report_path = tmp_path / "report" / "feature_promotion_report.json"
    report = build_report(
        snapshot_path=tmp_path / "missing_snapshots.csv",
        finalized_path=tmp_path / "missing_finalized.csv",
        report_path=report_path,
    )

    assert report_path.exists()
    assert report["status"] == "partial"
    assert report["features"] == []
    assert report["blockers"]
