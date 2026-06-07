from __future__ import annotations

import json
from pathlib import Path

import pytest


SNAPSHOT_PATH = Path("data/prediction_snapshots.csv")
FINALIZED_PATH = Path("data/finalized_games.csv")
REPORT_PATH = Path("report/settled_prediction_link_report.json")


def test_settled_prediction_link_report_contract() -> None:
    if not SNAPSHOT_PATH.exists():
        pytest.skip("data/prediction_snapshots.csv not found")

    if not FINALIZED_PATH.exists():
        pytest.skip("data/finalized_games.csv not found")

    from scripts.settled_prediction_link_report import build_report

    report = build_report()

    assert isinstance(report, dict)
    assert report.get("status") in {"ok", "failed"}

    required_keys = [
        "generated_at",
        "status",
        "input_files",
        "snapshot_game_count",
        "finalized_game_count",
        "linked_game_count",
        "linked_snapshot_row_count",
        "link_rate",
        "errors",
        "warnings",
        "recommendations",
    ]

    for key in required_keys:
        assert key in report

    assert isinstance(report["input_files"], dict)
    assert isinstance(report["errors"], list)
    assert isinstance(report["warnings"], list)
    assert isinstance(report["recommendations"], list)

    assert report["snapshot_game_count"] >= 0
    assert report["finalized_game_count"] >= 0
    assert report["linked_game_count"] >= 0
    assert report["linked_snapshot_row_count"] >= 0
    assert 0.0 <= float(report["link_rate"]) <= 1.0

    assert REPORT_PATH.exists()

    with REPORT_PATH.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)

    assert saved.get("linked_game_count") == report.get("linked_game_count")
