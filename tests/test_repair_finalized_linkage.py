from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from scripts.repair_finalized_linkage import build_report


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self.payload


def _live_payload(
    *,
    home: str = "Dodgers",
    away: str = "Giants",
    home_runs: int = 5,
    away_runs: int = 3,
    status: str = "Final",
    official_date: str = "2026-05-27",
) -> dict[str, Any]:
    return {
        "gameData": {
            "status": {"abstractGameState": status, "detailedState": status},
            "datetime": {"officialDate": official_date},
            "teams": {
                "home": {"name": home, "teamName": home, "abbreviation": home[:3].upper()},
                "away": {"name": away, "teamName": away, "abbreviation": away[:3].upper()},
            },
        },
        "liveData": {
            "linescore": {
                "teams": {
                    "home": {"runs": home_runs},
                    "away": {"runs": away_runs},
                }
            }
        },
    }


def _schedule_payload() -> dict[str, Any]:
    return {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 888,
                        "officialDate": "2026-05-27",
                        "status": {"abstractGameState": "Final"},
                        "teams": {
                            "home": {
                                "score": 7,
                                "team": {"name": "Los Angeles Dodgers"},
                            },
                            "away": {
                                "score": 4,
                                "team": {"name": "San Francisco Giants"},
                            },
                        },
                    }
                ]
            }
        ]
    }


def _write_snapshots(path: Path, *, game_id: str = "999") -> None:
    pd.DataFrame(
        [
            {
                "game_id": game_id,
                "game_date": "2026-05-27",
                "snapshot_created_at": "2026-05-27T12:00:00Z",
                "pipeline_version": "baseline_v2_clean",
                "snapshot_valid": "true",
                "home_team": "Dodgers",
                "away_team": "Giants",
            }
        ]
    ).to_csv(path, index=False)


def test_direct_live_feed_final_appends_snapshot_game_id(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "prediction_snapshots.csv"
    finalized_path = tmp_path / "finalized_games.csv"
    report_path = tmp_path / "report" / "finalized_linkage_diagnostic_report.json"

    _write_snapshots(snapshot_path, game_id="999")
    pd.DataFrame(columns=["game_id", "home_win"]).to_csv(finalized_path, index=False)

    def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(_live_payload(home="Los Angeles Dodgers", away="San Francisco Giants"))

    report = build_report(
        snapshot_path=snapshot_path,
        finalized_path=finalized_path,
        report_path=report_path,
        request_get=fake_get,
        sleep_seconds=0,
    )

    assert report_path.exists()
    assert report["api_final_written_count"] == 1
    assert report["overlap_count_after"] == 1

    finalized = pd.read_csv(finalized_path, dtype=str)
    assert finalized.iloc[-1]["game_id"] == "999"
    assert finalized.iloc[-1]["home_win"] == "1"


def test_schedule_match_repairs_when_direct_feed_fails(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "prediction_snapshots.csv"
    finalized_path = tmp_path / "finalized_games.csv"
    report_path = tmp_path / "report" / "finalized_linkage_diagnostic_report.json"

    _write_snapshots(snapshot_path, game_id="snapshot-123")
    pd.DataFrame(columns=["game_id", "home_win"]).to_csv(finalized_path, index=False)

    def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        if "schedule" in url:
            return FakeResponse(_schedule_payload())
        return FakeResponse({}, status_code=404)

    report = build_report(
        snapshot_path=snapshot_path,
        finalized_path=finalized_path,
        report_path=report_path,
        request_get=fake_get,
        sleep_seconds=0,
    )

    assert report["schedule_matched_count"] == 1
    assert report["overlap_count_after"] == 1

    finalized = pd.read_csv(finalized_path, dtype=str)
    assert finalized.iloc[-1]["game_id"] == "snapshot-123"
    assert finalized.iloc[-1]["home_win"] == "1"


def test_missing_files_produce_structured_report(tmp_path: Path) -> None:
    report_path = tmp_path / "report" / "finalized_linkage_diagnostic_report.json"

    report = build_report(
        snapshot_path=tmp_path / "missing_snapshots.csv",
        finalized_path=tmp_path / "missing_finalized.csv",
        report_path=report_path,
        sleep_seconds=0,
    )

    assert report_path.exists()
    assert report["status"] == "partial"
    assert report["errors"]
