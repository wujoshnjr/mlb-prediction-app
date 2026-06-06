"""
test_promotion_gate_contract.py

Promotion gate must not allow production/live promotion when blocked or insufficient.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPORT_PATH = Path("report/promotion_gate_report.json")


def test_promotion_gate_contract() -> None:
    if not REPORT_PATH.exists():
        pytest.skip("report/promotion_gate_report.json not found")

    with REPORT_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    assert "promotion_allowed" in data, "missing promotion_allowed"
    assert "production_allowed" in data, "missing production_allowed"
    assert "blockers" in data, "missing blockers"

    status = data.get("status")
    if status in {"insufficient_samples", "blocked"}:
        assert data.get("promotion_allowed") is False
        assert data.get("production_allowed") is False

    # The current system should never enable production without explicit promotion.
    if data.get("production_allowed") is True:
        assert data.get("promotion_allowed") is True
        assert not data.get("blockers"), "production cannot be allowed with blockers"
