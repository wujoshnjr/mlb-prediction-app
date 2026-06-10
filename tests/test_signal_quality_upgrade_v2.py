from __future__ import annotations

from pathlib import Path

from scripts.edge_sanity_guardrail import build_report as build_edge_report
from scripts.signal_quality_report import build_report as build_signal_report


def test_signal_quality_upgrade_missing_files_do_not_crash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    report_dir = tmp_path / "report"
    data_dir.mkdir()
    report_dir.mkdir()

    monkeypatch.setattr(
        "scripts.edge_sanity_guardrail.SNAPSHOT_PATH",
        data_dir / "prediction_snapshots.csv",
    )
    monkeypatch.setattr(
        "scripts.edge_sanity_guardrail.FINALIZED_PATH",
        data_dir / "finalized_games.csv",
    )
    monkeypatch.setattr(
        "scripts.edge_sanity_guardrail.FINALIZED_SNAPSHOT_OUTCOMES_PATH",
        data_dir / "finalized_snapshot_outcomes.csv",
    )
    monkeypatch.setattr(
        "scripts.edge_sanity_guardrail.REPORT_PATH",
        report_dir / "edge_sanity_guardrail_report.json",
    )

    monkeypatch.setattr(
        "scripts.signal_quality_report.EDGE_SANITY_PATH",
        report_dir / "edge_sanity_guardrail_report.json",
    )
    monkeypatch.setattr(
        "scripts.signal_quality_report.UNDERDOG_PATH",
        report_dir / "underdog_diagnostic_report.json",
    )
    monkeypatch.setattr(
        "scripts.signal_quality_report.CONFIDENCE_PATH",
        report_dir / "confidence_bucket_guardrail_report.json",
    )
    monkeypatch.setattr(
        "scripts.signal_quality_report.LINEUP_PATH",
        report_dir / "lineup_quality_report.json",
    )
    monkeypatch.setattr(
        "scripts.signal_quality_report.FRESHNESS_PATH",
        report_dir / "feature_freshness_report.json",
    )
    monkeypatch.setattr(
        "scripts.signal_quality_report.MODEL_CORRECTNESS_PATH",
        report_dir / "model_correctness_report.json",
    )
    monkeypatch.setattr(
        "scripts.signal_quality_report.SLICE_GATE_PATH",
        report_dir / "slice_promotion_gate_report.json",
    )
    monkeypatch.setattr(
        "scripts.signal_quality_report.SAMPLE_STATE_PATH",
        data_dir / "sample_state.json",
    )
    monkeypatch.setattr(
        "scripts.signal_quality_report.REPORT_PATH",
        report_dir / "signal_quality_report.json",
    )

    edge = build_edge_report()
    signal = build_signal_report()

    assert edge["live_betting_allowed"] is False
    assert edge["automated_wagering_allowed"] is False
    assert edge["production_model_replacement_allowed"] is False

    assert signal["live_betting_allowed"] is False
    assert signal["automated_wagering_allowed"] is False
    assert signal["production_model_replacement_allowed"] is False
    assert "cases" in signal
