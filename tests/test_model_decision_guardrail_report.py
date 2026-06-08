from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.model_decision_guardrail_report import build_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_model_decision_guardrail_blocks_high_confidence(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    data_dir = tmp_path / "data"

    predictions = []
    for i in range(40):
        predictions.append(
            {
                "game_id": str(i),
                "model_name": "xgboost_classifier",
                "predicted_prob": 0.75,
                "actual_home_win": 0 if i < 25 else 1,
                "confidence_bucket": "high_65_plus",
                "prediction_side": "home",
            }
        )
        predictions.append(
            {
                "game_id": str(i),
                "model_name": "market_no_vig_baseline",
                "predicted_prob": 0.52,
                "actual_home_win": 0 if i < 25 else 1,
                "confidence_bucket": "low_45_55",
                "prediction_side": "home",
            }
        )

    predictions_path = data_dir / "walk_forward_predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(predictions_path, index=False)

    _write_json(report_dir / "walk_forward_validation_report.json", {})
    _write_json(
        report_dir / "calibration_diagnostics_report.json",
        {
            "calibration_ready": False,
            "sample_count": 40,
            "overall": {"ece": 0.12},
        },
    )
    _write_json(report_dir / "model_comparison_report.json", {"recommended_challenger": "xgboost_classifier"})
    _write_json(report_dir / "prediction_trust_report.json", {"trust_counts": {}})
    _write_json(data_dir / "sample_state.json", {"clean_settled_snapshots": 40, "train_eligible_samples": 40})

    output_path = report_dir / "model_decision_guardrail_report.json"

    report = build_report(
        walk_forward_report_path=report_dir / "walk_forward_validation_report.json",
        walk_forward_predictions_path=predictions_path,
        calibration_report_path=report_dir / "calibration_diagnostics_report.json",
        model_comparison_path=report_dir / "model_comparison_report.json",
        prediction_trust_path=report_dir / "prediction_trust_report.json",
        sample_state_path=data_dir / "sample_state.json",
        output_path=output_path,
    )

    assert output_path.exists()
    assert report["decision"] == "NO_PROMOTION_SHADOW_ONLY"
    assert report["production_model_replacement_allowed"] is False
    assert report["probability_policy"]["block_high_confidence_language"] is True
    assert any(item["slice"] == "high_65_plus" for item in report["critical_findings"])


def test_model_decision_guardrail_missing_inputs_do_not_crash(tmp_path: Path) -> None:
    report = build_report(
        walk_forward_report_path=tmp_path / "missing_walk.json",
        walk_forward_predictions_path=tmp_path / "missing_predictions.csv",
        calibration_report_path=tmp_path / "missing_calibration.json",
        model_comparison_path=tmp_path / "missing_comparison.json",
        prediction_trust_path=tmp_path / "missing_trust.json",
        sample_state_path=tmp_path / "missing_sample_state.json",
        output_path=tmp_path / "report" / "model_decision_guardrail_report.json",
    )

    assert report["decision"] == "NO_PROMOTION_SHADOW_ONLY"
    assert report["production_model_replacement_allowed"] is False
    assert report["blockers"]
