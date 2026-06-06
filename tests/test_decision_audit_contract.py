"""
test_decision_audit_contract.py

Decision audit report and CSV must expose the expected contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


REPORT_PATH = Path("report/decision_audit_report.json")
CSV_PATH = Path("report/decision_audit.csv")


def test_decision_audit_json_contract() -> None:
    if not REPORT_PATH.exists():
        pytest.skip("report/decision_audit_report.json not found")

    with REPORT_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    assert "status" in data, "missing status"
    assert "audit_count" in data or "rows" in data, "missing audit_count or rows"


def test_decision_audit_csv_contract() -> None:
    if not CSV_PATH.exists():
        pytest.skip("report/decision_audit.csv not found")

    frame = pd.read_csv(CSV_PATH)
    required = {
        "game_id",
        "recommendation",
        "data_quality_grade",
        "live_betting_allowed",
        "stake_multiplier",
    }
    missing = required - set(frame.columns)
    assert not missing, f"missing decision audit columns: {sorted(missing)}"

    if frame.empty:
        pytest.skip("decision_audit.csv has headers but no rows")

    live_values = frame["live_betting_allowed"].astype(str).str.lower()
    assert not live_values.isin({"true", "1", "yes"}).any(), (
        "decision audit must not allow live betting"
    )
