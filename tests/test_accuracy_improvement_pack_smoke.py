from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.confidence_bucket_guardrail import build_report as build_confidence_report
from scripts.feature_freshness_report import build_report as build_freshness_report
from scripts.lineup_quality_builder import build_lineup_quality
from scripts.model_correctness_report import build_report as build_correctness_report
from scripts.slice_promotion_gate import build_report as build_slice_gate_report
from scripts.underdog_diagnostic_report import build_report as build_underdog_report


def test_accuracy_improvement_pack_missing_files_do_not_crash(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    report_dir = tmp_path / "report"
    data_dir.mkdir()
    report_dir.mkdir()

    # underdog
    monkeypatch.setattr("scripts.underdog_diagnostic_report.SNAPSHOT_PATH", data_dir / "prediction_snapshots.csv")
    monkeypatch.setattr("scripts.underdog_diagnostic_report.FINALIZED_PATH", data_dir / "finalized_games.csv")
    monkeypatch.setattr("scripts.underdog_diagnostic_report.FINALIZED_SNAPSHOT_OUTCOMES_PATH", data_dir / "finalized_snapshot_outcomes.csv")
    monkeypatch.setattr("scripts.underdog_diagnostic_report.MARKET_ODDS_PATH", data_dir / "market_odds_history.csv")
    monkeypatch.setattr("scripts.underdog_diagnostic_report.SAMPLE_STATE_PATH", data_dir / "sample_state.json")

    # confidence
    monkeypatch.setattr("scripts.confidence_bucket_guardrail.SNAPSHOT_PATH", data_dir / "prediction_snapshots.csv")
    monkeypatch.setattr("scripts.confidence_bucket_guardrail.FINALIZED_PATH", data_dir / "finalized_games.csv")
    monkeypatch.setattr("scripts.confidence_bucket_guardrail.FINALIZED_SNAPSHOT_OUTCOMES_PATH", data_dir / "finalized_snapshot_outcomes.csv")

    # lineup
    monkeypatch.setattr("scripts.lineup_quality_builder.SNAPSHOT_PATH", data_dir / "prediction_snapshots.csv")
    monkeypatch.setattr("scripts.lineup_quality_builder.DAILY_CONTEXT_PATH", data_dir / "daily_game_context.csv")
    monkeypatch.setattr("scripts.lineup_quality_builder.SAVANT_TOP3_CONTEXT_PATH", data_dir / "savant_top3_context.csv")
    monkeypatch.setattr("scripts.lineup_quality_builder.PROJECTED_LINEUP_CONTEXT_PATH", data_dir / "projected_lineup_context.csv")
    monkeypatch.setattr("scripts.lineup_quality_builder.OUTPUT_CSV_PATH", data_dir / "lineup_quality_context.csv")
    monkeypatch.setattr("scripts.lineup_quality_builder.REPORT_PATH", report_dir / "lineup_quality_report.json")

    # freshness
    monkeypatch.setattr("scripts.feature_freshness_report.SNAPSHOT_PATH", data_dir / "prediction_snapshots.csv")
    monkeypatch.setattr("scripts.feature_freshness_report.MARKET_ODDS_PATH", data_dir / "market_odds_history.csv")
    monkeypatch.setattr("scripts.feature_freshness_report.DAILY_CONTEXT_PATH", data_dir / "daily_game_context.csv")
    monkeypatch.setattr("scripts.feature_freshness_report.SAVANT_TOP3_CONTEXT_PATH", data_dir / "savant_top3_context.csv")
    monkeypatch.setattr("scripts.feature_freshness_report.WEATHER_CONTEXT_PATH", data_dir / "weather_context.csv")

    monkeypatch.setattr(
        "scripts.feature_freshness_report.SOURCE_CONFIG",
        {
            "odds": {
                "path": data_dir / "market_odds_history.csv",
                "time_columns": ["captured_at"],
            },
            "prediction_snapshots": {
                "path": data_dir / "prediction_snapshots.csv",
                "time_columns": ["snapshot_created_at"],
            },
            "daily_context": {
                "path": data_dir / "daily_game_context.csv",
                "time_columns": ["captured_at"],
            },
            "lineup": {
                "path": data_dir / "daily_game_context.csv",
                "time_columns": ["captured_at"],
            },
            "savant_top3": {
                "path": data_dir / "savant_top3_context.csv",
                "time_columns": ["captured_at"],
            },
            "weather": {
                "path": data_dir / "weather_context.csv",
                "time_columns": ["captured_at"],
            },
        },
    )

    # correctness
    monkeypatch.setattr("scripts.model_correctness_report.SNAPSHOT_PATH", data_dir / "prediction_snapshots.csv")
    monkeypatch.setattr("scripts.model_correctness_report.FINALIZED_PATH", data_dir / "finalized_games.csv")
    monkeypatch.setattr("scripts.model_correctness_report.FINALIZED_SNAPSHOT_OUTCOMES_PATH", data_dir / "finalized_snapshot_outcomes.csv")
    monkeypatch.setattr("scripts.model_correctness_report.LINEUP_QUALITY_CONTEXT_PATH", data_dir / "lineup_quality_context.csv")
    monkeypatch.setattr("scripts.model_correctness_report.UNDERDOG_REPORT_PATH", report_dir / "underdog_diagnostic_report.json")
    monkeypatch.setattr("scripts.model_correctness_report.CONFIDENCE_REPORT_PATH", report_dir / "confidence_bucket_guardrail_report.json")
    monkeypatch.setattr("scripts.model_correctness_report.FEATURE_FRESHNESS_REPORT_PATH", report_dir / "feature_freshness_report.json")
    monkeypatch.setattr("scripts.model_correctness_report.LINEUP_QUALITY_REPORT_PATH", report_dir / "lineup_quality_report.json")

    # slice gate
    monkeypatch.setattr("scripts.slice_promotion_gate.SAMPLE_STATE_PATH", data_dir / "sample_state.json")
    monkeypatch.setattr("scripts.slice_promotion_gate.UNDERDOG_REPORT_PATH", report_dir / "underdog_diagnostic_report.json")
    monkeypatch.setattr("scripts.slice_promotion_gate.CONFIDENCE_REPORT_PATH", report_dir / "confidence_bucket_guardrail_report.json")
    monkeypatch.setattr("scripts.slice_promotion_gate.MODEL_DECISION_GUARDRAIL_PATH", report_dir / "model_decision_guardrail_report.json")
    monkeypatch.setattr("scripts.slice_promotion_gate.CALIBRATION_DIAGNOSTICS_PATH", report_dir / "calibration_diagnostics_report.json")

    underdog = build_underdog_report()
    confidence = build_confidence_report()
    lineup_frame, lineup_report = build_lineup_quality()
    freshness = build_freshness_report()
    slice_gate = build_slice_gate_report()
    correctness = build_correctness_report()

    assert underdog["live_betting_allowed"] is False
    assert confidence["live_betting_allowed"] is False
    assert lineup_report["live_betting_allowed"] is False
    assert freshness["live_betting_allowed"] is False
    assert slice_gate["live_betting_allowed"] is False
    assert correctness["live_betting_allowed"] is False
    assert isinstance(lineup_frame, pd.DataFrame)
