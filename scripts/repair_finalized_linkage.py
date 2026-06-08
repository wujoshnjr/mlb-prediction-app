from __future__ import annotations

import csv
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


DATA_DIR = Path("data")
REPORT_DIR = Path("report")

SNAPSHOT_PATH = DATA_DIR / "prediction_snapshots.csv"
FINALIZED_PATH = DATA_DIR / "finalized_games.csv"
REPORT_PATH = REPORT_DIR / "finalized_linkage_diagnostic_report.json"

MLB_LIVE_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

MAX_REPAIR_ATTEMPTS = 200

FINALIZED_FIELDNAMES = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_win",
    "status",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}

    if isinstance(value, list):
        return [_json_safe(child) for child in value]

    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _normalize_game_id(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()
    if not text:
        return ""

    try:
        parsed = float(text)
        if math.isfinite(parsed) and parsed.is_integer():
            return str(int(parsed))
    except Exception:
        pass

    return text


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        parsed = int(float(str(value).strip()))
        return parsed
    except Exception:
        return None


def _team_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("st.", "saint")
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _team_matches(left: Any, right: Any) -> bool:
    left_key = _team_key(left)
    right_key = _team_key(right)

    if not left_key or not right_key:
        return True

    if left_key == right_key:
        return True

    # Allow "Dodgers" to match "Los Angeles Dodgers".
    if len(left_key) >= 4 and left_key in right_key:
        return True

    if len(right_key) >= 4 and right_key in left_key:
        return True

    return False


def _read_csv(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    status = {
        "path": str(path),
        "exists": path.exists(),
        "rows": 0,
        "error": "",
    }

    if not path.exists():
        status["error"] = "file_missing"
        return pd.DataFrame(), status

    try:
        frame = pd.read_csv(path, dtype=str)
    except Exception as exc:
        status["error"] = str(exc)
        return pd.DataFrame(), status

    status["rows"] = int(len(frame))
    return frame, status


def _prepare_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "game_id" not in frame.columns:
        return pd.DataFrame()

    result = frame.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    result = result[result["game_id"] != ""].copy()

    if "snapshot_valid" in result.columns:
        valid = result["snapshot_valid"].astype(str).str.strip().str.lower()
        result = result[valid.isin({"true", "1", "yes", "y", "valid", "ok"})].copy()

    if "pipeline_version" in result.columns:
        preferred = result[
            result["pipeline_version"].astype(str).str.strip() == "baseline_v2_clean"
        ].copy()
        if not preferred.empty:
            result = preferred

    if "snapshot_created_at" in result.columns:
        result["_sort_time"] = pd.to_datetime(
            result["snapshot_created_at"],
            errors="coerce",
            utc=True,
        )
        result = result.sort_values(
            ["game_id", "_sort_time"],
            kind="mergesort",
            na_position="first",
        )
        result = result.groupby("game_id", as_index=False).tail(1)
        result = result.sort_values(
            ["_sort_time", "game_id"],
            kind="mergesort",
            na_position="last",
        )
    else:
        result = result.drop_duplicates("game_id", keep="last")

    return result.reset_index(drop=True)


def _prepare_finalized(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "game_id" not in frame.columns:
        return pd.DataFrame()

    result = frame.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    result = result[result["game_id"] != ""].copy()

    return result.drop_duplicates("game_id", keep="last").reset_index(drop=True)


def _finalized_ids(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "game_id" not in frame.columns:
        return set()

    return {
        _normalize_game_id(value)
        for value in frame["game_id"].tolist()
        if _normalize_game_id(value)
    }


def _extract_live_game(
    payload: Dict[str, Any],
    *,
    snapshot_game_id: str,
    api_game_id: Optional[str] = None,
) -> Dict[str, Any]:
    status_data = payload.get("gameData", {}).get("status", {})
    abstract_state = str(status_data.get("abstractGameState", "") or "").strip()
    detailed_state = str(status_data.get("detailedState", "") or "").strip()

    game_data = payload.get("gameData", {})
    teams = game_data.get("teams", {})
    home_team = teams.get("home", {}) or {}
    away_team = teams.get("away", {}) or {}

    linescore = payload.get("liveData", {}).get("linescore", {})
    score_data = linescore.get("teams", {})
    home_score = _safe_int((score_data.get("home", {}) or {}).get("runs"))
    away_score = _safe_int((score_data.get("away", {}) or {}).get("runs"))

    game_date = str(
        (game_data.get("datetime", {}) or {}).get("officialDate")
        or ""
    ).strip()

    home_name = str(
        home_team.get("name")
        or home_team.get("teamName")
        or home_team.get("abbreviation")
        or ""
    ).strip()

    away_name = str(
        away_team.get("name")
        or away_team.get("teamName")
        or away_team.get("abbreviation")
        or ""
    ).strip()

    is_final = abstract_state.lower() == "final"

    if not is_final:
        return {
            "final": False,
            "status": abstract_state or detailed_state or "unknown",
            "reason": "game_not_final",
            "api_game_id": api_game_id,
        }

    if home_score is None or away_score is None:
        return {
            "final": False,
            "status": abstract_state or detailed_state or "Final",
            "reason": "final_but_score_missing",
            "api_game_id": api_game_id,
        }

    return {
        "final": True,
        "status": "Final",
        "game": {
            "game_id": snapshot_game_id,
            "game_date": game_date,
            "home_team": home_name,
            "away_team": away_name,
            "home_score": home_score,
            "away_score": away_score,
            "home_win": 1 if home_score > away_score else 0,
            "status": "Final",
        },
        "api_game_id": api_game_id,
        "api_home_team": home_name,
        "api_away_team": away_name,
    }


def _teams_compatible(snapshot: Dict[str, Any], live_result: Dict[str, Any]) -> bool:
    if not live_result.get("final"):
        return False

    game = live_result.get("game") or {}

    return (
        _team_matches(snapshot.get("home_team"), game.get("home_team"))
        and _team_matches(snapshot.get("away_team"), game.get("away_team"))
    )


def _fetch_live_result(
    game_id: str,
    *,
    snapshot: Dict[str, Any],
    request_get: Callable[..., Any],
) -> Dict[str, Any]:
    url = MLB_LIVE_FEED_URL.format(game_id=game_id)

    try:
        response = request_get(url, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {
            "method": "direct_live_feed",
            "snapshot_game_id": game_id,
            "final": False,
            "status": "request_failed",
            "reason": str(exc),
        }

    parsed = _extract_live_game(
        payload,
        snapshot_game_id=game_id,
        api_game_id=game_id,
    )
    parsed["method"] = "direct_live_feed"
    parsed["snapshot_game_id"] = game_id

    if parsed.get("final") and not _teams_compatible(snapshot, parsed):
        parsed["final"] = False
        parsed["reason"] = "direct_game_id_team_mismatch"

    return parsed


def _schedule_games_for_date(
    game_date: str,
    *,
    request_get: Callable[..., Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    status = {
        "date": game_date,
        "error": "",
        "game_count": 0,
    }

    if not game_date:
        status["error"] = "missing_game_date"
        return [], status

    try:
        response = request_get(
            MLB_SCHEDULE_URL,
            params={
                "sportId": 1,
                "date": game_date,
                "hydrate": "team,linescore",
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        status["error"] = str(exc)
        return [], status

    games: List[Dict[str, Any]] = []
    for date_block in payload.get("dates", []) or []:
        for game in date_block.get("games", []) or []:
            if isinstance(game, dict):
                games.append(game)

    status["game_count"] = len(games)
    return games, status


def _schedule_game_to_finalized(
    schedule_game: Dict[str, Any],
    *,
    snapshot_game_id: str,
) -> Dict[str, Any]:
    status_data = schedule_game.get("status", {}) or {}
    abstract_state = str(status_data.get("abstractGameState", "") or "").strip()

    teams = schedule_game.get("teams", {}) or {}
    home = teams.get("home", {}) or {}
    away = teams.get("away", {}) or {}

    home_team_data = home.get("team", {}) or {}
    away_team_data = away.get("team", {}) or {}

    home_score = _safe_int(home.get("score"))
    away_score = _safe_int(away.get("score"))

    if home_score is None:
        home_score = _safe_int(
            ((schedule_game.get("linescore", {}) or {}).get("teams", {}) or {})
            .get("home", {})
            .get("runs")
        )

    if away_score is None:
        away_score = _safe_int(
            ((schedule_game.get("linescore", {}) or {}).get("teams", {}) or {})
            .get("away", {})
            .get("runs")
        )

    if abstract_state.lower() != "final":
        return {
            "final": False,
            "status": abstract_state or "unknown",
            "reason": "schedule_game_not_final",
        }

    if home_score is None or away_score is None:
        return {
            "final": False,
            "status": abstract_state or "Final",
            "reason": "schedule_final_but_score_missing",
        }

    return {
        "final": True,
        "status": "Final",
        "game": {
            "game_id": snapshot_game_id,
            "game_date": str(schedule_game.get("officialDate") or "").strip(),
            "home_team": str(home_team_data.get("name") or home_team_data.get("teamName") or "").strip(),
            "away_team": str(away_team_data.get("name") or away_team_data.get("teamName") or "").strip(),
            "home_score": home_score,
            "away_score": away_score,
            "home_win": 1 if home_score > away_score else 0,
            "status": "Final",
        },
        "api_game_id": _normalize_game_id(schedule_game.get("gamePk")),
        "api_home_team": str(home_team_data.get("name") or "").strip(),
        "api_away_team": str(away_team_data.get("name") or "").strip(),
    }


def _find_schedule_match(
    snapshot: Dict[str, Any],
    *,
    schedule_cache: Dict[str, Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    request_get: Callable[..., Any],
) -> Dict[str, Any]:
    snapshot_game_id = _normalize_game_id(snapshot.get("game_id"))
    game_date = str(snapshot.get("game_date") or "").strip()

    if not game_date:
        return {
            "method": "schedule_match",
            "snapshot_game_id": snapshot_game_id,
            "final": False,
            "status": "not_checked",
            "reason": "missing_game_date",
        }

    if game_date not in schedule_cache:
        schedule_cache[game_date] = _schedule_games_for_date(
            game_date,
            request_get=request_get,
        )

    schedule_games, schedule_status = schedule_cache[game_date]
    if schedule_status.get("error"):
        return {
            "method": "schedule_match",
            "snapshot_game_id": snapshot_game_id,
            "final": False,
            "status": "request_failed",
            "reason": schedule_status.get("error"),
            "schedule_status": schedule_status,
        }

    for schedule_game in schedule_games:
        teams = schedule_game.get("teams", {}) or {}
        home_team = ((teams.get("home", {}) or {}).get("team", {}) or {}).get("name")
        away_team = ((teams.get("away", {}) or {}).get("team", {}) or {}).get("name")

        if (
            _team_matches(snapshot.get("home_team"), home_team)
            and _team_matches(snapshot.get("away_team"), away_team)
        ):
            parsed = _schedule_game_to_finalized(
                schedule_game,
                snapshot_game_id=snapshot_game_id,
            )
            parsed["method"] = "schedule_match"
            parsed["snapshot_game_id"] = snapshot_game_id
            return parsed

    return {
        "method": "schedule_match",
        "snapshot_game_id": snapshot_game_id,
        "final": False,
        "status": "not_found",
        "reason": "no_schedule_team_match",
        "schedule_game_count": len(schedule_games),
    }


def _append_finalized_rows(path: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Safely upsert finalized rows using the canonical finalized_games schema.

    This intentionally rewrites the CSV instead of raw-appending because older
    files may only contain a subset of columns such as game_id/home_win. Rewriting
    prevents malformed mixed-width CSV rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame(columns=FINALIZED_FIELDNAMES)

    if path.exists() and path.stat().st_size > 0:
        try:
            existing = pd.read_csv(path, dtype=str)
        except Exception:
            existing = pd.DataFrame(columns=FINALIZED_FIELDNAMES)

    for column in FINALIZED_FIELDNAMES:
        if column not in existing.columns:
            existing[column] = ""

    existing = existing[FINALIZED_FIELDNAMES].copy()

    if not existing.empty:
        existing["game_id"] = existing["game_id"].apply(_normalize_game_id)
        existing = existing[existing["game_id"] != ""].copy()
        existing = existing.drop_duplicates("game_id", keep="last")

    existing_ids = _finalized_ids(existing)

    normalized_rows: List[Dict[str, Any]] = []
    seen = set(existing_ids)

    for row in rows:
        game_id = _normalize_game_id(row.get("game_id"))
        if not game_id or game_id in seen:
            continue

        home_score = _safe_int(row.get("home_score"))
        away_score = _safe_int(row.get("away_score"))
        if home_score is None or away_score is None:
            continue

        home_win = _safe_int(row.get("home_win"))
        if home_win not in {0, 1}:
            home_win = 1 if home_score > away_score else 0

        normalized_rows.append(
            {
                "game_id": game_id,
                "game_date": str(row.get("game_date") or "").strip(),
                "home_team": str(row.get("home_team") or "").strip(),
                "away_team": str(row.get("away_team") or "").strip(),
                "home_score": str(home_score),
                "away_score": str(away_score),
                "home_win": str(home_win),
                "status": "Final",
            }
        )
        seen.add(game_id)

    if normalized_rows:
        combined = pd.concat(
            [existing, pd.DataFrame(normalized_rows)],
            ignore_index=True,
        )
    else:
        combined = existing.copy()

    for column in FINALIZED_FIELDNAMES:
        if column not in combined.columns:
            combined[column] = ""

    combined = combined[FINALIZED_FIELDNAMES].copy()
    combined["game_id"] = combined["game_id"].apply(_normalize_game_id)
    combined = combined[combined["game_id"] != ""].copy()
    combined = combined.drop_duplicates("game_id", keep="last")

    combined.to_csv(path, index=False)

    return {
        "path": str(path),
        "received": len(rows),
        "inserted": len(normalized_rows),
        "duplicates_or_invalid_skipped": max(0, len(rows) - len(normalized_rows)),
        "total_rows_after": int(len(combined)),
    }
    

def build_report(
    *,
    snapshot_path: Path = SNAPSHOT_PATH,
    finalized_path: Path = FINALIZED_PATH,
    report_path: Path = REPORT_PATH,
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    request_get: Callable[..., Any] = requests.get,
    sleep_seconds: float = 0.10,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    snapshots_raw, snapshot_status = _read_csv(snapshot_path)
    finalized_raw, finalized_status = _read_csv(finalized_path)

    if snapshot_status["error"]:
        errors.append(f"prediction_snapshots unavailable: {snapshot_status['error']}")

    snapshots = _prepare_snapshots(snapshots_raw)
    finalized = _prepare_finalized(finalized_raw)

    snapshot_ids = _finalized_ids(snapshots)
    finalized_ids_before = _finalized_ids(finalized)
    overlap_before = snapshot_ids & finalized_ids_before

    missing_ids = sorted(snapshot_ids - finalized_ids_before)

    snapshot_date_range = {
        "min": "",
        "max": "",
    }
    if not snapshots.empty and "game_date" in snapshots.columns:
        dates = snapshots["game_date"].dropna().astype(str)
        dates = dates[dates.str.strip() != ""]
        if not dates.empty:
            snapshot_date_range = {"min": str(dates.min()), "max": str(dates.max())}

    finalized_date_range = {
        "min": "",
        "max": "",
    }
    if not finalized.empty and "game_date" in finalized.columns:
        dates = finalized["game_date"].dropna().astype(str)
        dates = dates[dates.str.strip() != ""]
        if not dates.empty:
            finalized_date_range = {"min": str(dates.min()), "max": str(dates.max())}

    if snapshots.empty or "game_id" not in snapshots.columns:
        repair_candidates = pd.DataFrame()
    else:
        repair_candidates = snapshots[
            snapshots["game_id"].isin(missing_ids)
        ].copy()

    rows_to_append: List[Dict[str, Any]] = []
    api_check_sample: List[Dict[str, Any]] = []
    schedule_cache: Dict[str, Tuple[List[Dict[str, Any]], Dict[str, Any]]] = {}

    attempted = 0
    pending_not_final_count = 0
    api_final_but_not_written_count = 0
    api_final_written_count = 0
    api_not_found_or_failed_count = 0
    schedule_matched_count = 0
    direct_matched_count = 0

    for _, snapshot_row in repair_candidates.head(max_repair_attempts).iterrows():
        snapshot = snapshot_row.to_dict()
        snapshot_game_id = _normalize_game_id(snapshot.get("game_id"))
        if not snapshot_game_id:
            continue

        attempted += 1

        direct = _fetch_live_result(
            snapshot_game_id,
            snapshot=snapshot,
            request_get=request_get,
        )

        selected = direct

        if not direct.get("final"):
            schedule_result = _find_schedule_match(
                snapshot,
                schedule_cache=schedule_cache,
                request_get=request_get,
            )

            if schedule_result.get("final"):
                selected = schedule_result
                schedule_matched_count += 1
            else:
                selected = direct

        if selected.get("final"):
            game = selected.get("game") or {}
            if game:
                rows_to_append.append(game)

                if selected.get("method") == "direct_live_feed":
                    direct_matched_count += 1
            else:
                api_final_but_not_written_count += 1
        else:
            reason = str(selected.get("reason") or "").lower()
            status = str(selected.get("status") or "").lower()

            if "not_final" in reason or status in {"preview", "scheduled", "in progress", "live"}:
                pending_not_final_count += 1
            else:
                api_not_found_or_failed_count += 1

        if len(api_check_sample) < 25:
            api_check_sample.append(
                {
                    "snapshot_game_id": snapshot_game_id,
                    "snapshot_game_date": snapshot.get("game_date", ""),
                    "snapshot_home_team": snapshot.get("home_team", ""),
                    "snapshot_away_team": snapshot.get("away_team", ""),
                    "direct": {
                        "final": direct.get("final"),
                        "status": direct.get("status"),
                        "reason": direct.get("reason"),
                        "method": direct.get("method"),
                        "api_game_id": direct.get("api_game_id"),
                    },
                    "selected": {
                        "final": selected.get("final"),
                        "status": selected.get("status"),
                        "reason": selected.get("reason"),
                        "method": selected.get("method"),
                        "api_game_id": selected.get("api_game_id"),
                    },
                }
            )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    append_summary = _append_finalized_rows(finalized_path, rows_to_append)

    api_final_written_count = int(append_summary.get("inserted", 0) or 0)
    api_final_but_not_written_count += max(
        0,
        len(rows_to_append) - api_final_written_count,
    )

    finalized_after_raw, finalized_after_status = _read_csv(finalized_path)
    finalized_after = _prepare_finalized(finalized_after_raw)
    finalized_ids_after = _finalized_ids(finalized_after)
    overlap_after = snapshot_ids & finalized_ids_after

    if snapshot_ids and not overlap_after:
        warnings.append(
            "No prediction snapshot game_id currently links to finalized_games.csv after repair."
        )

    if missing_ids and attempted == 0:
        warnings.append("Missing finalized snapshot game_ids exist, but no repair candidates were attempted.")

    status = "ok" if not errors else "partial"

    report = {
        "generated_at": _utc_now(),
        "status": status,
        "snapshot_game_id_count": len(snapshot_ids),
        "finalized_game_id_count_before": len(finalized_ids_before),
        "finalized_game_id_count_after": len(finalized_ids_after),
        "overlap_count_before": len(overlap_before),
        "overlap_count_after": len(overlap_after),
        "missing_finalized_game_id_count_before": len(missing_ids),
        "missing_finalized_game_ids_sample": missing_ids[:50],
        "snapshot_date_range": snapshot_date_range,
        "finalized_date_range": finalized_date_range,
        "repair_attempted": attempted,
        "max_repair_attempts": max_repair_attempts,
        "api_check_sample": api_check_sample,
        "pending_not_final_count": pending_not_final_count,
        "api_final_written_count": api_final_written_count,
        "api_final_but_not_written_count": api_final_but_not_written_count,
        "api_not_found_or_failed_count": api_not_found_or_failed_count,
        "direct_matched_count": direct_matched_count,
        "schedule_matched_count": schedule_matched_count,
        "append_summary": append_summary,
        "input_files": {
            "prediction_snapshots": snapshot_status,
            "finalized_games_before": finalized_status,
            "finalized_games_after": finalized_after_status,
        },
        "errors": errors,
        "warnings": warnings,
        "recommendations": [
            "If overlap_count_after remains zero, inspect api_check_sample for not_found, team_mismatch, or pending statuses.",
            "If games are pending, wait for final status before training/evaluation.",
            "If schedule_match works but direct_live_feed fails, keep using snapshot game_id as finalized_games.game_id so training joins remain stable.",
            "Do not use snapshot home_win/home_score as trusted outcomes; finalized_games.csv remains the trusted outcome source.",
        ],
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }

    _write_json(report_path, report)
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
