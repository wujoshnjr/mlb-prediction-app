from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from scripts.research_promotion_readiness_report import build_report


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _input_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "sample_state": tmp_path / "data" / "sample_state.json",
        "training_status": tmp_path / "data" / "training_status.json",
        "finalized_linkage": tmp_path / "report" / "finalized_linkage_diagnostic_report.json",
        "model_lab": tmp_path / "report" / "model_lab_report.json",
        "feature_promotion": tmp_path / "report" / "feature_promotion_report.json",
        "walk_forward_validation": tmp_path / "report" / "walk_forward_validation_report.json",
        "calibration_diagnostics": tmp_path / "report" / "calibration_diagnostics_report.json",
        "prediction_trust": tmp_path / "report" / "prediction_trust_report.json",
        "model_comparison": tmp_path / "report" / "model_comparison_report.json",
        "model_decision_guardrail": tmp_path / "report" / "model_decision_guardrail_report.json",
        "shadow_ensemble_stack": tmp_path / "report" / "shadow_ensemble_stack_report.json",
        "data_contract": tmp_path / "report" / "data_contract_report.json",
        "pipeline_manifest": tmp_path / "report" / "pipeline_manifest.json",
    }


def _write_minimal_valid_inputs(tmp_path: Path) -> dict[str, Path]:
    paths = _input_paths(tmp_path)

    _write_json(
        paths["sample_state"],
        {
            "clean_settled_snapshots": 520,
            "train_eligible_samples": 520,
            "linked_games": 520,
        },
    )
    _write_json(paths["training_status"], {"trained": True, "sample_count": 520})
    _write_json(
        paths["finalized_linkage"],
        {
            "overlap_count_after": 520,
            "pending_not_final_count": 0,
            "api_not_found_or_failed_count": 0,
        },
    )
    _write_json(
        paths["model_lab"],
        {
            "sample_count": 520,
            "champion_candidate": "xgboost_classifier",
            "best_by_brier": "xgboost_classifier",
            "models": [
                {
                    "model_name": "xgboost_classifier",
                    "trained": True,
                    "brier": 0.22,
                    "logloss": 0.65,
                    "ece": 0.04,
                    "promotion_eligible": True,
                }
            ],
        },
    )
    _write_json(
        paths["feature_promotion"],
        {
            "candidate_shadow_count": 5,
            "ready_for_review_count": 0,
        },
    )
    _write_json(
        paths["walk_forward_validation"],
        {
            "walkforward_ready": True,
            "total_oos_predictions": 310,
            "model_vs_market": {
                "xgboost_classifier": {
                    "beats_market_brier": True,
                    "beats_market_logloss": False,
                    "delta_brier": -0.01,
                    "delta_logloss": 0.01,
                }
            },
        },
    )
    _write_json(
        paths["calibration_diagnostics"],
        {
            "calibration_ready": True,
            "sample_count": 520,
            "overall": {"ece": 0.04},
        },
    )
    _write_json(paths["prediction_trust"], {"trust_counts": {"A": 2, "B": 3}})
    _write_json(
        paths["model_comparison"],
        {
            "recommended_challenger": "xgboost_classifier",
            "recommended_champion": None,
        },
    )
    _write_json(
        paths["model_decision_guardrail"],
        {
            "status": "blocked",
            "decision": "NO_PROMOTION_SHADOW_ONLY",
            "recommended_challenger": "xgboost_classifier",
            "production_model_replacement_allowed": False,
            "probability_policy": {
                "official_probability_change_allowed": False,
                "shadow_probability_shrinkage_allowed": True,
                "recommended_default_alpha": 0.85,
                "recommended_max_display_confidence": 0.75,
                "block_high_confidence_language": False,
                "blocked_or_downgraded_slices": [],
            },
        },
    )
    _write_json(
        paths["shadow_ensemble_stack"],
        {
            "sample_count": 310,
            "recommended_shadow_ensemble": "weighted_average",
            "promotion_eligible": False,
        },
    )
    _write_json(paths["data_contract"], {"status": "ok"})
    _write_json(paths["pipeline_manifest"], {"tracked_file_count": 10})

    return paths


def test_missing_reports_do_not_crash(tmp_path: Path) -> None:
    output = tmp_path / "report" / "research_promotion_readiness_report.json"

    report = build_report(
        output_path=output,
        input_paths=_input_paths(tmp_path),
    )

    assert output.exists()
    assert report["status"] in {"insufficient_evidence", "blocked", "failed_safety"}
    assert report["research_promotion_allowed"] is False
    assert report["production_model_replacement_allowed"] is False
    assert report["live_betting_allowed"] is False
    assert report["automated_wagering_allowed"] is False


def test_sufficient_evidence_can_be_eligible_for_research_review(tmp_path: Path) -> None:
    output = tmp_path / "report" / "research_promotion_readiness_report.json"
    paths = _write_minimal_valid_inputs(tmp_path)

    report = build_report(output_path=output, input_paths=paths)

    assert output.exists()
    assert report["status"] == "eligible_for_research_review"
    assert report["research_promotion_allowed"] is True
    assert report["recommended_champion"] == "xgboost_classifier"
    assert report["production_model_replacement_allowed"] is False


def test_insufficient_samples_are_blocked(tmp_path: Path) -> None:
    output = tmp_path / "report" / "research_promotion_readiness_report.json"
    paths = _write_minimal_valid_inputs(tmp_path)

    _write_json(
        paths["sample_state"],
        {
            "clean_settled_snapshots": 153,
            "train_eligible_samples": 153,
            "linked_games": 153,
        },
    )

    report = build_report(output_path=output, input_paths=paths)

    assert report["status"] == "insufficient_evidence"
    assert report["research_promotion_allowed"] is False
    assert any("train eligible samples below threshold" in item for item in report["blockers"])


def test_unsafe_governance_flag_blocks(tmp_path: Path) -> None:
    output = tmp_path / "report" / "research_promotion_readiness_report.json"
    paths = _write_minimal_valid_inputs(tmp_path)

    _write_json(
        paths["model_lab"],
        {
            "sample_count": 520,
            "champion_candidate": "xgboost_classifier",
            "live_betting_allowed": True,
            "models": [],
        },
    )

    report = build_report(output_path=output, input_paths=paths)

    assert report["status"] == "failed_safety"
    assert report["research_promotion_allowed"] is False
    assert report["unsafe_governance_findings"]
