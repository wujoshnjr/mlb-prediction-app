from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import scripts.model_status_consistency_report as report


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def patch_paths(tmp_path, monkeypatch, *, artifact_name: str = "calibrator.pkl"):
    monkeypatch.setattr(
        report,
        "TRAINING_SAMPLES_PATH",
        tmp_path / "training_samples.csv",
    )
    monkeypatch.setattr(
        report,
        "TRAINING_STATUS_PATH",
        tmp_path / "training_status.json",
    )
    monkeypatch.setattr(
        report,
        "CALIBRATOR_PATH",
        tmp_path / artifact_name,
    )
    monkeypatch.setattr(
        report,
        "MODEL_DIR_ARTIFACT_PATH",
        tmp_path / "missing_model.joblib",
    )
    monkeypatch.setattr(
        report,
        "MODEL_ARTIFACT_STATUS_PATH",
        tmp_path / "model_artifact_status.json",
    )
    monkeypatch.setattr(
        report,
        "PREDICTION_REPORT_PATH",
        tmp_path / "prediction.json",
    )
    monkeypatch.setattr(
        report,
        "REPORT_PATH",
        tmp_path / "model_status_consistency_report.json",
    )


def write_training_samples(path: Path, count: int) -> None:
    rows = [["feature_a", "feature_b"]]
    rows.extend([[str(index), str(index + 1)] for index in range(count)])
    write_csv(path, rows)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def write_artifact(path: Path, training_sample_count: int) -> None:
    with path.open("wb") as handle:
        pickle.dump(
            {
                "metadata": {
                    "training_sample_count": training_sample_count,
                    "pipeline_version": "baseline_v2_clean",
                    "model_type": "test",
                }
            },
            handle,
        )


def write_consistent_fixture(
    tmp_path,
    *,
    sample_count: int = 3,
    active_model_allowed: bool = True,
) -> None:
    write_training_samples(tmp_path / "training_samples.csv", sample_count)

    write_json(
        tmp_path / "training_status.json",
        {
            "sample_count": sample_count,
            "trained": True,
            "reason": "",
        },
    )

    write_json(
        tmp_path / "model_artifact_status.json",
        {
            "training_sample_count": sample_count,
            "training_status_sample_count": sample_count,
            "training_samples_row_count": sample_count,
            "sample_count_consistent": True,
            "active_model_allowed": active_model_allowed,
        },
    )

    write_json(
        tmp_path / "prediction.json",
        {
            "model_governance": {
                "loaded_artifact_sample_count": sample_count,
                "active_model_allowed": active_model_allowed,
            }
        },
    )

    write_artifact(tmp_path / "calibrator.pkl", sample_count)


def test_missing_files_partial_no_crash(tmp_path, monkeypatch):
    patch_paths(tmp_path, monkeypatch)

    result = report.generate_report()

    assert result["status"] == "partial"
    assert result["mismatches"] == []
    assert result["live_betting_allowed"] is False
    assert result["automated_wagering_allowed"] is False
    assert result["production_model_replacement_allowed"] is False
    assert "manual baseline should remain active" in " ".join(
        result["recommendations"]
    )


def test_matching_counts_ok(tmp_path, monkeypatch):
    patch_paths(tmp_path, monkeypatch)
    write_consistent_fixture(tmp_path, sample_count=3, active_model_allowed=True)

    result = report.generate_report()

    assert result["status"] == "ok"
    assert result["mismatches"] == []
    assert result["sample_count_consistent"] is True
    assert result["active_model_allowed"] is True
    assert result["trained"] is True


def test_training_samples_vs_training_status_mismatch(tmp_path, monkeypatch):
    patch_paths(tmp_path, monkeypatch)

    write_training_samples(tmp_path / "training_samples.csv", 3)
    write_json(
        tmp_path / "training_status.json",
        {
            "sample_count": 5,
            "trained": True,
        },
    )

    result = report.generate_report()

    assert result["status"] == "failed"
    assert result["mismatches"][0]["field"] == "training_samples_vs_training_status"
    assert result["mismatches"][0]["left"] == 3
    assert result["mismatches"][0]["right"] == 5


def test_artifact_status_vs_training_status_mismatch(tmp_path, monkeypatch):
    patch_paths(tmp_path, monkeypatch)

    write_json(
        tmp_path / "training_status.json",
        {
            "sample_count": 100,
            "trained": True,
        },
    )
    write_json(
        tmp_path / "model_artifact_status.json",
        {
            "training_sample_count": 200,
        },
    )

    result = report.generate_report()

    assert result["status"] == "failed"
    assert any(
        mismatch["field"] == "artifact_status_vs_training_status"
        for mismatch in result["mismatches"]
    )


def test_artifact_metadata_vs_training_status_mismatch(tmp_path, monkeypatch):
    patch_paths(tmp_path, monkeypatch)

    write_json(
        tmp_path / "training_status.json",
        {
            "sample_count": 100,
            "trained": True,
        },
    )
    write_artifact(tmp_path / "calibrator.pkl", 200)

    result = report.generate_report()

    assert result["status"] == "failed"
    assert any(
        mismatch["field"] == "artifact_metadata_vs_training_status"
        for mismatch in result["mismatches"]
    )


def test_prediction_loaded_count_mismatch(tmp_path, monkeypatch):
    patch_paths(tmp_path, monkeypatch)

    write_json(
        tmp_path / "training_status.json",
        {
            "sample_count": 100,
            "trained": True,
        },
    )
    write_json(
        tmp_path / "prediction.json",
        {
            "model_governance": {
                "loaded_artifact_sample_count": 200,
            }
        },
    )

    result = report.generate_report()

    assert result["status"] == "failed"
    assert any(
        mismatch["field"]
        == "prediction_loaded_artifact_sample_count_vs_training_status"
        for mismatch in result["mismatches"]
    )


def test_prediction_model_governance_active_model_allowed_is_detected(
    tmp_path,
    monkeypatch,
):
    patch_paths(tmp_path, monkeypatch)

    write_training_samples(tmp_path / "training_samples.csv", 3)

    write_json(
        tmp_path / "training_status.json",
        {
            "sample_count": 3,
            "trained": True,
        },
    )

    write_json(
        tmp_path / "model_artifact_status.json",
        {
            "training_sample_count": 3,
            "training_status_sample_count": 3,
            "training_samples_row_count": 3,
            "sample_count_consistent": True,
            "active_model_allowed": False,
        },
    )

    write_json(
        tmp_path / "prediction.json",
        {
            "model_governance": {
                "loaded_artifact_sample_count": 3,
                "active_model_allowed": True,
            }
        },
    )

    write_artifact(tmp_path / "calibrator.pkl", 3)

    result = report.generate_report()

    assert result["status"] == "ok"
    assert result["active_model_allowed"] is True
    assert result["sample_count_consistent"] is True


def test_safety_flags_always_false(tmp_path, monkeypatch):
    patch_paths(tmp_path, monkeypatch)

    result = report.generate_report()

    assert result["live_betting_allowed"] is False
    assert result["automated_wagering_allowed"] is False
    assert result["production_model_replacement_allowed"] is False


def test_json_safe_removes_nan_and_infinity():
    payload = {
        "a": float("nan"),
        "b": float("inf"),
        "c": -float("inf"),
        "d": {"nested": float("nan")},
        "e": [1.0, float("nan")],
    }

    clean = report._json_safe(payload)
    dumped = json.dumps(clean, allow_nan=False)

    assert "NaN" not in dumped
    assert "Infinity" not in dumped
    assert clean["a"] is None
    assert clean["b"] is None
    assert clean["c"] is None
    assert clean["d"]["nested"] is None


def test_artifact_status_sample_count_consistent_false_triggers_mismatch(
    tmp_path,
    monkeypatch,
):
    patch_paths(tmp_path, monkeypatch)

    write_training_samples(tmp_path / "training_samples.csv", 2)

    write_json(
        tmp_path / "training_status.json",
        {
            "sample_count": 2,
            "trained": True,
        },
    )

    write_json(
        tmp_path / "model_artifact_status.json",
        {
            "training_sample_count": 2,
            "training_status_sample_count": 2,
            "training_samples_row_count": 2,
            "sample_count_consistent": False,
        },
    )

    write_json(
        tmp_path / "prediction.json",
        {
            "model_governance": {
                "loaded_artifact_sample_count": 2,
            }
        },
    )

    write_artifact(tmp_path / "calibrator.pkl", 2)

    result = report.generate_report()

    assert result["status"] == "failed"
    assert any(
        mismatch["field"] == "model_artifact_status_sample_count_consistent"
        for mismatch in result["mismatches"]
    )


def test_no_artifact_recommends_manual_baseline(tmp_path, monkeypatch):
    patch_paths(tmp_path, monkeypatch)

    write_training_samples(tmp_path / "training_samples.csv", 1)

    write_json(
        tmp_path / "training_status.json",
        {
            "sample_count": 1,
            "trained": True,
        },
    )

    write_json(
        tmp_path / "model_artifact_status.json",
        {
            "training_sample_count": 1,
            "training_status_sample_count": 1,
            "training_samples_row_count": 1,
            "sample_count_consistent": True,
        },
    )

    write_json(
        tmp_path / "prediction.json",
        {
            "model_governance": {
                "loaded_artifact_sample_count": 1,
            }
        },
    )

    result = report.generate_report()

    assert result["artifact_exists"] is False
    assert "manual baseline should remain active" in " ".join(
        result["recommendations"]
    )
