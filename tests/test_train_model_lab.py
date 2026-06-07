from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.train_model_lab import build_report


def _make_dataset(tmp_path: Path, rows: int, include_market: bool = True) -> tuple[Path, Path]:
    snapshot_rows = []
    finalized_rows = []

    for i in range(rows):
        game_id = 7000 + i
        home_win = int(i % 2)

        row = {
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
        }
        if include_market:
            row["market_no_vig_home_prob"] = 0.45 + (i % 10) / 100

        snapshot_rows.append(row)
        finalized_rows.append({"game_id": str(game_id), "home_win": home_win})

    snapshot_path = tmp_path / "prediction_snapshots.csv"
    finalized_path = tmp_path / "finalized_games.csv"
    pd.DataFrame(snapshot_rows).to_csv(snapshot_path, index=False)
    pd.DataFrame(finalized_rows).to_csv(finalized_path, index=False)
    return snapshot_path, finalized_path


def test_small_sample_report_is_created_and_models_skipped(tmp_path: Path) -> None:
    snapshot_path, finalized_path = _make_dataset(tmp_path, rows=20)
    report_path = tmp_path / "report" / "model_lab_report.json"
    artifact_dir = tmp_path / "data" / "model_lab"

    report = build_report(
        snapshot_path=snapshot_path,
        finalized_path=finalized_path,
        report_path=report_path,
        artifact_dir=artifact_dir,
    )

    assert report_path.exists()
    assert report["sample_count"] == 20
    assert any("sample_count below shadow training threshold" in item for item in report["global_blockers"])
    assert all(
        model["skipped"] is True
        for model in report["models"]
        if model["model_name"] != "market_no_vig_baseline"
    )


def test_missing_market_probability_skips_market_baseline(tmp_path: Path) -> None:
    snapshot_path, finalized_path = _make_dataset(tmp_path, rows=90, include_market=False)
    report_path = tmp_path / "report" / "model_lab_report.json"
    artifact_dir = tmp_path / "data" / "model_lab"

    report = build_report(
        snapshot_path=snapshot_path,
        finalized_path=finalized_path,
        report_path=report_path,
        artifact_dir=artifact_dir,
    )

    market = [m for m in report["models"] if m["model_name"] == "market_no_vig_baseline"][0]
    assert market["skipped"] is True
    assert report["market_baseline_available"] is False


def test_logistic_baseline_trains_with_enough_fake_samples(tmp_path: Path) -> None:
    snapshot_path, finalized_path = _make_dataset(tmp_path, rows=120)
    report_path = tmp_path / "report" / "model_lab_report.json"
    artifact_dir = tmp_path / "data" / "model_lab"

    report = build_report(
        snapshot_path=snapshot_path,
        finalized_path=finalized_path,
        report_path=report_path,
        artifact_dir=artifact_dir,
    )

    logistic = [m for m in report["models"] if m["model_name"] == "logistic_baseline"][0]
    assert logistic["trained"] is True
    assert logistic["skipped"] is False
    assert report_path.exists()
    assert artifact_dir.exists()
    assert (artifact_dir / "logistic_baseline.pkl").exists()


def test_model_lab_never_writes_main_calibrator(tmp_path: Path) -> None:
    snapshot_path, finalized_path = _make_dataset(tmp_path, rows=120)
    report_path = tmp_path / "report" / "model_lab_report.json"
    artifact_dir = tmp_path / "data" / "model_lab"
    main_calibrator = tmp_path / "data" / "calibrator.pkl"

    build_report(
        snapshot_path=snapshot_path,
        finalized_path=finalized_path,
        report_path=report_path,
        artifact_dir=artifact_dir,
    )

    assert not main_calibrator.exists()


def test_xgboost_lightgbm_import_failures_do_not_crash(tmp_path: Path, monkeypatch) -> None:
    snapshot_path, finalized_path = _make_dataset(tmp_path, rows=90)
    report_path = tmp_path / "report" / "model_lab_report.json"
    artifact_dir = tmp_path / "data" / "model_lab"

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name in {"lightgbm", "xgboost"}:
            raise ImportError("forced missing optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    report = build_report(
        snapshot_path=snapshot_path,
        finalized_path=finalized_path,
        report_path=report_path,
        artifact_dir=artifact_dir,
    )

    lgb = [m for m in report["models"] if m["model_name"] == "lightgbm_classifier"][0]
    xgb = [m for m in report["models"] if m["model_name"] == "xgboost_classifier"][0]

    assert lgb["skipped"] is True
    assert xgb["skipped"] is True
    assert report_path.exists()
