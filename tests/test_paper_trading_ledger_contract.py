"""
test_paper_trading_ledger_contract.py

Paper trading ledger must not contain live stake.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


LEDGER_PATH = Path("data/paper_trading_ledger.csv")
REPORT_PATH = Path("report/paper_trading_ledger_report.json")


def test_paper_trading_ledger_has_no_live_stake() -> None:
    if not LEDGER_PATH.exists():
        pytest.skip("data/paper_trading_ledger.csv not found")

    frame = pd.read_csv(LEDGER_PATH)

    if frame.empty:
        pytest.skip("paper trading ledger is empty")

    if "paper_stake_units" in frame.columns:
        paper_stakes = pd.to_numeric(frame["paper_stake_units"], errors="coerce").fillna(0.0)
        assert (paper_stakes >= 0).all(), "paper_stake_units must be non-negative"

    if "live_stake_units" in frame.columns:
        live_stakes = pd.to_numeric(frame["live_stake_units"], errors="coerce").fillna(0.0)
        assert (live_stakes == 0).all(), "live_stake_units must always be zero"

    assert "game_id" in frame.columns, "ledger missing game_id column"
    assert not frame["game_id"].isna().all(), "all ledger game_id values are empty"


def test_paper_trading_ledger_report_has_no_live_exposure() -> None:
    if not REPORT_PATH.exists():
        pytest.skip("report/paper_trading_ledger_report.json not found")

    with REPORT_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    assert "status" in data, "missing status"

    for key in ("live_stake_units", "live_exposure_units"):
        if key in data and data[key] is not None:
            assert float(data[key]) == 0.0, f"{key} must be zero"
