from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import artifact_rebuild_readiness_report as report_mod


def write_csv_rows(path: Path, rows: int) -> None:
    """Write a CSV file with a header and exactly `rows` data rows."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_a", "feature_b"])

        for index in range(rows):
            writer.writerow([index, index + 1])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def patch_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        report_mod,
        "TRAINING_SAMPLES_PATH",
        tmp_path / "data" / "training_samples.csv",
    )
    monkeypatch.setattr(
        report_mod,
        "TRAINING_STATUS_PATH",
        tmp_path / "data" / "training_status.json",
    )
    monkeypatch.setattr(
        report_mod,
        "MODEL_ARTIFACT_STATUS_PATH",
        tmp_path / "data" / "model_artifact_status.json",
    )
    monkeypatch.setattr(
        report_mod,
        "MODEL_ARTIFACT_STATUS_REPORT_PATH",
        tmp_path / "report" / "model_artifact_status_report.json",
    )
    monkeypatch.setattr(
        report_mod,
        "MODEL_STATUS_CONSISTENCY_REPORT_PATH",
        tmp_path / "report" / "model_status_consistency_report.json",
    )
    monkeypatch.setattr(
        report_mod,
        "MODEL_LAB_REPORT_PATH",
        tmp_path / "report" / "model_lab_report.json",
    )
    monkeypatch.setattr(
        report_mod,
        "WALK_FORWARD_VALIDATION_REPORT_PATH",
        tmp_path / "report" / "walk_forward_validation_report.json",
    )
    monkeypatch.setattr(
        report_mod,
        "REPORT_PATH",
        tmp_path / "report" / "artifact_rebuild_readiness_report.json",
    )


def write_ready_fixture(
    tmp_path: Path,
    *,
    sample_count: int = 300,
    consistency_mismatches: list[dict] | None = None,
    walk_ready: bool = False,
    model_oos: int = 0,
    artifact_valid: bool = True,
    artifact_reason: str = "",
    artifact_action: str = "",
) -> None:
    if consistency_mismatches is None:
        consistency_mismatches = []

    write_csv_rows(
        tmp_path / "data" / "training_samples.csv",
        sample_count,
    )

    write_json(
        tmp_path / "data" / "training_status.json",
        {
            "sample_count": sample_count,
            "trained": True,
        },
    )

    write_json(
        tmp_path / "data" / "model_artifact_status.json",
        {
            "valid": artifact_valid,
            "reason": artifact_reason,
            "action": artifact_action,
            "active_model_allowed": False,
        },
    )

    write_json(
        tmp_path / "report" / "model_artifact_status_report.json",
        {
            "status": "ok",
        },
    )

    write_json(
        tmp_path / "report" / "model_status_consistency_report.json",
        {
            "status": "ok" if not consistency_mismatches else "failed",
            "sample_count_consistent": len(consistency_mismatches) == 0,
            "mismatches": consistency_mismatches,
        },
    )

    write_json(
        tmp_path / "report" / "model_lab_report.json",
        {
            "sample_count": sample_count,
            "best_by_brier": "logistic_baseline",
            "best_by_logloss": "logistic_baseline",
            "best_by_ece": "xgboost_classifier",
        },
    )

    write_json(
        tmp_path / "report" / "walk_forward_validation_report.json",
        {
            "walkforward_ready": walk_ready,
            "total_oos_predictions": model_oos,
            "unique_oos_games": model_oos,
            "model_oos_counts": (
                {"logistic_baseline": model_oos}
                if model_oos > 0
                else {}
            ),
        },
    )


def test_missing_files_no_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    patch_paths(tmp_path, monkeypatch)

    result = report_mod.generate_report()

    assert result["status"] == "blocked"
    assert result["artifact_rebuild_allowed"] is False
    assert result["production_model_replacement_allowed"] is False
    assert result["live_betting_allowed"] is False
    assert result["automated_wagering_allowed"] is False
    assert len(result["blockers"]) > 0


def test_sample_count_below_300_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    write_ready_fixture(
        tmp_path,
        sample_count=299,
        consistency_mismatches=[],
        walk_ready=True,
        model_oos=300,
    )
    patch_paths(tmp_path, monkeypatch)

    result = report_mod.generate_report()

    assert result["training_samples_row_count"] == 299
    assert result["training_status_sample_count"] == 299
    assert result["artifact_rebuild_allowed"] is False
    assert any(
        blocker["code"] == "insufficient_training_samples_for_rebuild"
        for blocker in result["blockers"]
    )
    assert any(
        blocker["code"] == "insufficient_training_status_samples_for_rebuild"
        for blocker in result["blockers"]
    )


def test_training_samples_mismatch_training_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    write_ready_fixture(
        tmp_path,
        sample_count=300,
        consistency_mismatches=[],
        walk_ready=True,
        model_oos=300,
    )

    write_json(
        tmp_path / "data" / "training_status.json",
        {
            "sample_count": 301,
            "trained": True,
        },
    )

    patch_paths(tmp_path, monkeypatch)

    result = report_mod.generate_report()

    assert result["training_samples_row_count"] == 300
    assert result["training_status_sample_count"] == 301
    assert result["artifact_rebuild_allowed"] is False
    assert any(
        blocker["code"] == "training_samples_mismatch_training_status"
        for blocker in result["blockers"]
    )


def test_model_status_consistency_mismatches_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    write_ready_fixture(
        tmp_path,
        sample_count=300,
        consistency_mismatches=[
            {
                "field": "artifact_metadata_vs_training_status",
                "left": 109,
                "right": 300,
            }
        ],
        walk_ready=True,
        model_oos=300,
    )

    patch_paths(tmp_path, monkeypatch)

    result = report_mod.generate_report()

    assert result["model_status_consistency_mismatch_count"] == 1
    assert result["artifact_rebuild_allowed"] is False
    assert any(
        blocker["code"] == "model_status_consistency_mismatches_present"
        for blocker in result["blockers"]
    )


def test_artifact_invalid_sample_count_mismatch_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    write_ready_fixture(
        tmp_path,
        sample_count=300,
        consistency_mismatches=[],
        walk_ready=True,
        model_oos=300,
        artifact_valid=False,
        artifact_reason="sample_count_mismatch",
    )

    patch_paths(tmp_path, monkeypatch)

    result = report_mod.generate_report()

    assert result["artifact_quarantine_required"] is True
    assert "Keep current artifact quarantined" in " ".join(
        result["recommendations"]
    )


def test_exact_300_matching_no_mismatches_rebuild_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    write_ready_fixture(
        tmp_path,
        sample_count=300,
        consistency_mismatches=[],
        walk_ready=True,
        model_oos=300,
    )

    patch_paths(tmp_path, monkeypatch)

    result = report_mod.generate_report()

    assert result["training_samples_row_count"] == 300
    assert result["training_status_sample_count"] == 300
    assert result["model_status_consistency_mismatch_count"] == 0
    assert result["walk_forward_max_model_oos"] == 300
    assert result["artifact_rebuild_allowed"] is True
    assert result["status"] == "ok"
    assert result["artifact_rebuild_status"] == "ready_for_rebuild"


def test_500_samples_walk_forward_not_ready_no_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    write_ready_fixture(
        tmp_path,
        sample_count=500,
        consistency_mismatches=[],
        walk_ready=False,
        model_oos=350,
    )

    patch_paths(tmp_path, monkeypatch)

    result = report_mod.generate_report()

    assert result["artifact_rebuild_allowed"] is True
    assert result["walk_forward_ready"] is False
    assert result["promotion_candidate_allowed"] is False
    assert result["production_model_replacement_allowed"] is False


def test_500_samples_ready_walk_forward_promotion_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    write_ready_fixture(
        tmp_path,
        sample_count=500,
        consistency_mismatches=[],
        walk_ready=True,
        model_oos=300,
    )

    patch_paths(tmp_path, monkeypatch)

    result = report_mod.generate_report()

    assert result["artifact_rebuild_allowed"] is True
    assert result["walk_forward_ready"] is True
    assert result["walk_forward_max_model_oos"] == 300
    assert result["promotion_candidate_allowed"] is True
    assert result["production_model_replacement_allowed"] is False


def test_production_model_replacement_always_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    write_ready_fixture(
        tmp_path,
        sample_count=500,
        consistency_mismatches=[],
        walk_ready=True,
        model_oos=300,
    )

    patch_paths(tmp_path, monkeypatch)

    result = report_mod.generate_report()

    assert result["production_model_replacement_allowed"] is False


def test_live_betting_and_automated_wagering_always_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    write_ready_fixture(
        tmp_path,
        sample_count=300,
        consistency_mismatches=[],
        walk_ready=True,
        model_oos=300,
    )

    patch_paths(tmp_path, monkeypatch)

    result = report_mod.generate_report()

    assert result["live_betting_allowed"] is False
    assert result["automated_wagering_allowed"] is False


def test_json_safe_removes_nan_infinity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    patch_paths(tmp_path, monkeypatch)

    write_csv_rows(
        tmp_path / "data" / "training_samples.csv",
        300,
    )

    with (tmp_path / "data" / "training_status.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "sample_count": float("nan"),
                "extra": float("inf"),
                "list": [1, 2, float("-inf")],
                "nested": {"value": float("nan")},
            },
            handle,
        )

    write_json(
        tmp_path / "data" / "model_artifact_status.json",
        {
            "valid": True,
            "reason": "",
            "action": "",
        },
    )
    write_json(
        tmp_path / "report" / "model_artifact_status_report.json",
        {
            "status": "ok",
        },
    )
    write_json(
        tmp_path / "report" / "model_status_consistency_report.json",
        {
            "status": "ok",
            "sample_count_consistent": True,
            "mismatches": [],
        },
    )
    write_json(
        tmp_path / "report" / "model_lab_report.json",
        {
            "sample_count": 300,
        },
    )
    write_json(
        tmp_path / "report" / "walk_forward_validation_report.json",
        {
            "walkforward_ready": True,
            "total_oos_predictions": 300,
            "unique_oos_games": 300,
            "model_oos_counts": {"logistic_baseline": 300},
        },
    )

    result = report_mod.generate_report()

    assert result["training_status_sample_count"] is None

    report_path = tmp_path / "report" / "artifact_rebuild_readiness_report.json"
    assert report_path.exists()

    with report_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    dumped = json.dumps(loaded, allow_nan=False)

    assert "NaN" not in dumped
    assert "Infinity" not in dumped
