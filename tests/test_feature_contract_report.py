# tests/test_feature_contract_report.py
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import feature_contract_report


def test_feature_contract_report_writes_dict(tmp_path, monkeypatch):
    report_path = tmp_path / "feature_contract_report.json"
    monkeypatch.setattr(feature_contract_report, "REPORT_PATH", report_path)

    report = feature_contract_report.generate_report()

    assert isinstance(report, dict)
    assert report_path.exists()

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["status"] in {"ok", "warning", "error", "skipped"}
    assert "feature_schema_hash" in data
    assert "core_feature_count" in data
    assert isinstance(data["checks"], dict)
    assert data["live_betting_allowed"] is False
    assert data["automated_wagering_allowed"] is False
    assert data["production_model_replacement_allowed"] is False


def test_feature_contract_report_json_has_no_nan(tmp_path, monkeypatch):
    report_path = tmp_path / "feature_contract_report.json"
    monkeypatch.setattr(feature_contract_report, "REPORT_PATH", report_path)

    feature_contract_report.generate_report()
    payload = report_path.read_text(encoding="utf-8")

    assert "NaN" not in payload
    assert "Infinity" not in payload
    assert "-Infinity" not in payload
