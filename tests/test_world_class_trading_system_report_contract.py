from __future__ import annotations

import json
from pathlib import Path

import pytest


PREDICTION_PATH = Path("report/prediction.json")
REPORT_PATH = Path("report/world_class_trading_system_report.json")


def test_world_class_trading_system_report_contract() -> None:
    if not PREDICTION_PATH.exists():
        pytest.skip("report/prediction.json not found")

    from scripts.world_class_trading_system_report import build_report

    report = build_report()

    assert isinstance(report, dict)
    assert "generated_at" in report
    assert "status" in report
    assert "world_class_stage" in report
    assert "overall_score" in report
    assert "overall_grade" in report
    assert "layers" in report
    assert "recommendations" in report

    assert report.get("live_betting_allowed") is False
    assert report.get("shadow_live_allowed") is False
    assert report.get("production_allowed") is False

    layers = report.get("layers")
    assert isinstance(layers, dict)

    required_layers = {
        "data_trust",
        "research_quality",
        "risk_controls",
        "model_upgrade_path",
        "product_readiness",
    }

    assert required_layers.issubset(set(layers.keys()))

    for key in required_layers:
        layer = layers[key]
        assert isinstance(layer, dict)
        assert "status" in layer
        assert "score" in layer
        assert "grade" in layer
        assert "signals" in layer
        assert "blockers" in layer
        assert "warnings" in layer
        assert "recommendations" in layer
        assert isinstance(layer["signals"], list)

    risk_layer = layers["risk_controls"]
    assert risk_layer.get("status") in {
        "ready_for_next_review",
        "building",
        "not_ready",
        "blocked",
    }

    critical_blockers = report.get("critical_blockers")
    assert isinstance(critical_blockers, list)

    if critical_blockers:
        assert report.get("status") == "critical_blocked"

    assert REPORT_PATH.exists()

    with REPORT_PATH.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)

    assert saved.get("live_betting_allowed") is False
    assert saved.get("production_allowed") is False
