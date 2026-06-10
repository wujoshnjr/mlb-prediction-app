from __future__ import annotations

from pathlib import Path

from scripts.product_experience_report import build_report


def test_product_experience_report_missing_files_do_not_crash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    report_dir = tmp_path / "report"
    data_dir.mkdir()
    report_dir.mkdir()

    monkeypatch.setattr("scripts.product_experience_report.SAMPLE_STATE_PATH", data_dir / "sample_state.json")
    monkeypatch.setattr("scripts.product_experience_report.MODEL_CORRECTNESS_PATH", report_dir / "model_correctness_report.json")
    monkeypatch.setattr("scripts.product_experience_report.EDGE_SANITY_PATH", report_dir / "edge_sanity_guardrail_report.json")
    monkeypatch.setattr("scripts.product_experience_report.SIGNAL_QUALITY_PATH", report_dir / "signal_quality_report.json")
    monkeypatch.setattr("scripts.product_experience_report.UNDERDOG_PATH", report_dir / "underdog_diagnostic_report.json")
    monkeypatch.setattr("scripts.product_experience_report.CONFIDENCE_PATH", report_dir / "confidence_bucket_guardrail_report.json")
    monkeypatch.setattr("scripts.product_experience_report.SLICE_GATE_PATH", report_dir / "slice_promotion_gate_report.json")
    monkeypatch.setattr("scripts.product_experience_report.LINEUP_PATH", report_dir / "lineup_quality_report.json")
    monkeypatch.setattr("scripts.product_experience_report.FRESHNESS_PATH", report_dir / "feature_freshness_report.json")
    monkeypatch.setattr("scripts.product_experience_report.READINESS_PATH", report_dir / "research_promotion_readiness_report.json")
    monkeypatch.setattr("scripts.product_experience_report.REPORT_PATH", report_dir / "product_experience_report.json")

    report = build_report()

    assert report["live_betting_allowed"] is False
    assert report["automated_wagering_allowed"] is False
    assert report["production_model_replacement_allowed"] is False
    assert "hero_metrics" in report
    assert "signal_case_cards" in report
    assert "evidence_cards" in report
    assert "risk_cards" in report
    assert "copy_blocks" in report
