from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


ODDS_FETCH_DIAGNOSTIC_PATH = Path("report/odds_fetch_diagnostic.json")


TEAM_ABBREV_MAP = {
    "la dodgers": "los angeles dodgers",
    "lad dodgers": "los angeles dodgers",
    "ny yankees": "new york yankees",
    "nyy yankees": "new york yankees",
    "ny mets": "new york mets",
    "nym mets": "new york mets",
    "chi cubs": "chicago cubs",
    "chc cubs": "chicago cubs",
    "chi white sox": "chicago white sox",
    "cws white sox": "chicago white sox",
    "sf giants": "san francisco giants",
    "sfg giants": "san francisco giants",
    "sd padres": "san diego padres",
    "sdp padres": "san diego padres",
    "tb rays": "tampa bay rays",
    "tbr rays": "tampa bay rays",
    "kc royals": "kansas city royals",
    "kcr royals": "kansas city royals",
    "az diamondbacks": "arizona diamondbacks",
    "ari diamondbacks": "arizona diamondbacks",
    "dc nationals": "washington nationals",
    "ws nationals": "washington nationals",
    "wsh nationals": "washington nationals",
    "stl cardinals": "st louis cardinals",
    "st louis cardinals": "st louis cardinals",
    "la angels": "los angeles angels",
    "laa angels": "los angeles angels",
}

SAMPLE_CANDIDATE_COLUMNS = [
    "game_id",
    "game_date",
    "commence_time",
    "captured_at",
    "market",
    "sportsbook",
    "bookmaker",
    "home_team",
    "away_team",
    "team",
    "side",
    "outcome_name",
    "price",
    "odds",
    "american_odds",
    "is_closing_snapshot",
]


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy/NaN values into JSON-safe values."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return value

    if isinstance(value, (int, str, bool)):
        return value

    if hasattr(value, "item"):
        try:
            item = value.item()
            if isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
                return ""
            return item
        except Exception:
            pass

    return str(value)


def _safe_read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """Read JSON file safely."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            return None, f"File {path} does not contain a JSON object"

        return data, ""
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON in {path}: {exc}"
    except Exception as exc:
        return None, f"Error reading {path}: {exc}"


def _odds_fetch_diagnostic_summary() -> Dict[str, Any]:
    data, error = _safe_read_json(ODDS_FETCH_DIAGNOSTIC_PATH)

    if data is None:
        return {
            "exists": False,
            "status": "",
            "selected_attempt": "",
            "final_event_count": 0,
            "final_usable_row_count": 0,
            "attempts": [],
            "recommendations": [],
            "error": error,
        }

    attempts = data.get("attempts", [])
    if not isinstance(attempts, list):
        attempts = []

    recommendations = data.get("recommendations", [])
    if not isinstance(recommendations, list):
        recommendations = []

    return {
        "exists": True,
        "status": data.get("status", ""),
        "selected_attempt": data.get("selected_attempt", ""),
        "final_event_count": data.get("final_event_count", 0),
        "final_usable_row_count": data.get("final_usable_row_count", 0),
        "attempts": attempts,
        "recommendations": recommendations,
        "error": "",
    }


def _safe_read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    """Read CSV safely."""
    try:
        frame = pd.read_csv(path)
        return frame, ""
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except Exception as exc:
        return None, f"Error reading CSV {path}: {exc}"


def _normalize_team_name(value: Any) -> str:
    """Normalize team name for loose matching."""
    if value is None:
        return ""

    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return ""

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(mlb|baseball)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return TEAM_ABBREV_MAP.get(text, text)


def _string_key(value: Any) -> str:
    """Return a clean lowercase string key."""
    if value is None:
        return ""

    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return ""

    return text


def _parse_datetime(value: Any) -> str:
    """Return YYYY-MM-DD-like string when possible."""
    if value is None:
        return ""

    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""

    return text[:10]


def _column_exists(frame: pd.DataFrame, column: str) -> bool:
    return isinstance(frame, pd.DataFrame) and column in frame.columns


def _safe_count(frame: Optional[pd.DataFrame]) -> int:
    if frame is None:
        return 0
    return int(len(frame))


def _filter_moneyline(frame: pd.DataFrame) -> pd.DataFrame:
    """Return likely moneyline rows."""
    if frame is None or frame.empty:
        return pd.DataFrame()

    if not _column_exists(frame, "market"):
        return frame.iloc[0:0].copy()

    market_text = frame["market"].astype(str).str.lower().str.strip()
    mask = (
        market_text.eq("moneyline")
        | market_text.eq("ml")
        | market_text.str.contains("moneyline", na=False)
    )

    return frame[mask].copy()


def _extract_prediction_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract prediction rows from supported prediction report shapes."""
    for key in (
        "predictions",
        "today_predictions",
        "games",
        "recommendations",
        "paper_bets",
    ):
        rows = report.get(key, [])
        if isinstance(rows, list) and rows:
            return [item for item in rows if isinstance(item, dict)]

    return []


def _row_team_values(row: pd.Series) -> List[str]:
    values: List[str] = []

    for column in ("home_team", "away_team", "team", "side", "outcome_name"):
        if column in row.index:
            normalized = _normalize_team_name(row.get(column))
            if normalized:
                values.append(normalized)

    return values


def _find_candidate_odds_for_prediction(
    prediction: Dict[str, Any],
    odds_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Find possible odds rows for a prediction using game_id and team names."""
    if odds_frame is None or odds_frame.empty:
        return pd.DataFrame()

    pred_game_id = _string_key(prediction.get("game_id"))
    pred_home = _normalize_team_name(prediction.get("home_team"))
    pred_away = _normalize_team_name(prediction.get("away_team"))
    pred_date = _parse_datetime(
        prediction.get("game_date")
        or prediction.get("start_time")
        or prediction.get("commence_time")
    )

    mask = pd.Series(False, index=odds_frame.index)

    if pred_game_id and _column_exists(odds_frame, "game_id"):
        odds_game_ids = odds_frame["game_id"].map(_string_key)
        mask |= odds_game_ids.eq(pred_game_id)

    if pred_home and pred_away:
        if _column_exists(odds_frame, "home_team") and _column_exists(odds_frame, "away_team"):
            odds_home = odds_frame["home_team"].map(_normalize_team_name)
            odds_away = odds_frame["away_team"].map(_normalize_team_name)

            direct_match = odds_home.eq(pred_home) & odds_away.eq(pred_away)
            swapped_match = odds_home.eq(pred_away) & odds_away.eq(pred_home)

            mask |= direct_match | swapped_match

        if _column_exists(odds_frame, "team"):
            odds_team = odds_frame["team"].map(_normalize_team_name)
            mask |= odds_team.isin([pred_home, pred_away])

        if _column_exists(odds_frame, "outcome_name"):
            outcome_team = odds_frame["outcome_name"].map(_normalize_team_name)
            mask |= outcome_team.isin([pred_home, pred_away])

    candidates = odds_frame[mask].copy()

    if candidates.empty:
        return candidates

    date_columns = [
        column
        for column in ("game_date", "commence_time", "captured_at")
        if _column_exists(candidates, column)
    ]

    if pred_date and date_columns:
        date_mask = pd.Series(False, index=candidates.index)

        for column in date_columns:
            date_mask |= candidates[column].astype(str).str[:10].eq(pred_date)

        if date_mask.any():
            candidates = candidates[date_mask].copy()

    return candidates


def _candidate_by_game_id_count(
    prediction: Dict[str, Any],
    candidates: pd.DataFrame,
) -> int:
    if candidates.empty or not _column_exists(candidates, "game_id"):
        return 0

    pred_game_id = _string_key(prediction.get("game_id"))
    if not pred_game_id:
        return 0

    return int(candidates["game_id"].map(_string_key).eq(pred_game_id).sum())


def _candidate_by_team_name_count(
    prediction: Dict[str, Any],
    candidates: pd.DataFrame,
) -> int:
    if candidates.empty:
        return 0

    pred_home = _normalize_team_name(prediction.get("home_team"))
    pred_away = _normalize_team_name(prediction.get("away_team"))
    expected = {value for value in (pred_home, pred_away) if value}

    if not expected:
        return 0

    count = 0
    for _, row in candidates.iterrows():
        values = set(_row_team_values(row))
        if values.intersection(expected):
            count += 1

    return int(count)


def _has_price_column(frame: pd.DataFrame) -> Optional[str]:
    for column in ("price", "odds", "american_odds"):
        if _column_exists(frame, column):
            return column
    return None


def _summarize_candidate_odds(
    prediction: Dict[str, Any],
    candidates: pd.DataFrame,
) -> Dict[str, Any]:
    """Summarize candidate odds rows."""
    if candidates is None or candidates.empty:
        return {
            "candidate_count": 0,
            "candidate_by_game_id_count": 0,
            "candidate_by_team_name_count": 0,
            "candidate_moneyline_count": 0,
            "candidate_closing_count": 0,
            "candidate_sportsbooks": [],
            "candidate_markets": [],
            "has_home_price": False,
            "has_away_price": False,
            "sample_candidates": [],
        }

    moneyline = _filter_moneyline(candidates)

    sportsbooks = set()
    for column in ("sportsbook", "bookmaker"):
        if _column_exists(candidates, column):
            sportsbooks.update(
                _string_key(value)
                for value in candidates[column].tolist()
                if _string_key(value)
            )

    markets = []
    if _column_exists(candidates, "market"):
        markets = sorted(
            {
                _string_key(value)
                for value in candidates["market"].tolist()
                if _string_key(value)
            }
        )

    closing_count = 0
    if _column_exists(candidates, "is_closing_snapshot"):
        closing_text = candidates["is_closing_snapshot"].astype(str).str.lower()
        closing_count = int(closing_text.isin(["true", "1", "yes"]).sum())

    price_column = _has_price_column(candidates)
    pred_home = _normalize_team_name(prediction.get("home_team"))
    pred_away = _normalize_team_name(prediction.get("away_team"))

    has_home_price = False
    has_away_price = False

    if price_column:
        for _, row in candidates.iterrows():
            price_value = _json_safe(row.get(price_column))
            if price_value == "":
                continue

            values = set(_row_team_values(row))

            if pred_home and pred_home in values:
                has_home_price = True
            if pred_away and pred_away in values:
                has_away_price = True

            side = _string_key(row.get("side"))
            if side in {"home", "h"}:
                has_home_price = True
            if side in {"away", "a"}:
                has_away_price = True

    sample_candidates: List[Dict[str, Any]] = []
    for _, row in candidates.head(5).iterrows():
        sample: Dict[str, Any] = {}

        for column in SAMPLE_CANDIDATE_COLUMNS:
            if column in row.index:
                sample[column] = _json_safe(row.get(column))

        sample_candidates.append(sample)

    return {
        "candidate_count": int(len(candidates)),
        "candidate_by_game_id_count": _candidate_by_game_id_count(prediction, candidates),
        "candidate_by_team_name_count": _candidate_by_team_name_count(prediction, candidates),
        "candidate_moneyline_count": int(len(moneyline)),
        "candidate_closing_count": int(closing_count),
        "candidate_sportsbooks": sorted(sportsbooks),
        "candidate_markets": markets,
        "has_home_price": bool(has_home_price),
        "has_away_price": bool(has_away_price),
        "sample_candidates": sample_candidates,
    }


def _determine_likely_issue(
    odds_status: str,
    market_odds_file_exists: bool,
    market_odds_has_rows: bool,
    candidate_summary: Dict[str, Any],
) -> str:
    if odds_status == "OK":
        return "odds_ok"

    if not market_odds_file_exists:
        return "market_odds_file_missing"

    if not market_odds_has_rows:
        return "market_odds_file_empty"

    if candidate_summary["candidate_count"] == 0:
        return "no_candidate_odds_found"

    if candidate_summary["candidate_moneyline_count"] == 0:
        return "candidate_exists_but_no_moneyline"

    if not candidate_summary["has_home_price"] or not candidate_summary["has_away_price"]:
        return "incomplete_home_away_prices"

    if odds_status == "UNAVAILABLE" and candidate_summary["candidate_moneyline_count"] > 0:
        return "matching_logic_too_strict_or_schema_mismatch"

    return "unknown"


def _odds_quality_counts(predictions: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "OK": 0,
        "SUSPICIOUS": 0,
        "UNAVAILABLE": 0,
        "MISSING": 0,
    }

    for prediction in predictions:
        status = _string_key(prediction.get("odds_quality_status")).upper()
        if not status:
            status = "MISSING"

        counts[status] = counts.get(status, 0) + 1

    return counts


def _build_recommendations(
    report: Dict[str, Any],
    issue_counter: Counter,
) -> List[str]:
    recommendations: List[str] = []

    market = report["market_odds_history"]

    if not market["exists"]:
        recommendations.append(
            "Check odds fetch workflow and ODDS_API_KEY; market_odds_history.csv is missing or unreadable."
        )
    elif market["total_rows"] == 0:
        recommendations.append(
            "market_odds_history.csv exists but is empty. Check whether odds fetching runs before prediction."
        )
    elif market["moneyline_rows"] == 0:
        recommendations.append(
            "Odds were fetched but no moneyline market rows were stored. Check odds parser market mapping."
        )

    if issue_counter.get("matching_logic_too_strict_or_schema_mismatch", 0) > 0:
        recommendations.append(
            "Relax prediction.py odds matching by adding team-name fallback or schema alias mapping."
        )

    if issue_counter.get("no_candidate_odds_found", 0) > 0:
        recommendations.append(
            "No candidate odds found for some games. Verify team-name normalization, game_id alignment, and odds fetch timing."
        )

    if issue_counter.get("incomplete_home_away_prices", 0) > 0:
        recommendations.append(
            "Candidate odds exist but home/away prices are incomplete. Check outcome side extraction in odds parser."
        )

    if issue_counter.get("provider_returned_zero_events", 0) > 0:
        recommendations.append(
            "The Odds API returned zero events. Check ODDS_API_KEY, subscription, quota headers, baseball_mlb availability, and date window."
        )

    if issue_counter.get("parser_dropped_all_provider_events", 0) > 0:
        recommendations.append(
            "Provider returned events but parser produced zero usable rows. Inspect bookmaker market keys and h2h outcome parsing."
        )

    if issue_counter.get("schedule_matching_failed_after_fetch", 0) > 0:
        recommendations.append(
            "Usable odds rows exist but prediction games remain unavailable. Inspect model.attach_schedule_game_ids team/date matching."
        )

    if issue_counter.get("market_odds_history_not_persisted", 0) > 0:
        recommendations.append(
            "Odds are fetched but not persisted to data/market_odds_history.csv; inspect prediction.py market history writer."
        )

    if not recommendations:
        recommendations.append(
            "No obvious odds matching issue detected by diagnostic. Inspect prediction.py gating logic next."
        )

    return recommendations


def build_odds_matching_diagnostic(
    prediction_path: str = "report/prediction.json",
    market_odds_path: str = "data/market_odds_history.csv",
    context_path: str = "data/daily_game_context.csv",
    snapshots_path: str = "data/prediction_snapshots.csv",
    output_path: str = "report/odds_matching_diagnostic.json",
) -> Dict[str, Any]:
    """Build and write odds matching diagnostic report."""
    generated_at = datetime.now(timezone.utc).isoformat()

    diagnostic: Dict[str, Any] = {
        "generated_at": generated_at,
        "prediction_report": {
            "exists": False,
            "error": "",
            "scheduled_game_count": 0,
            "prediction_count": 0,
            "generated_at": "",
            "report_date": "",
        },
        "market_odds_history": {
            "exists": False,
            "error": "",
            "total_rows": 0,
            "moneyline_rows": 0,
            "closing_rows": 0,
            "columns": [],
        },
        "daily_context": {
            "exists": False,
            "error": "",
            "total_rows": 0,
        },
        "prediction_snapshots": {
            "exists": False,
            "error": "",
            "total_rows": 0,
        },
        "odds_fetch_diagnostic": _odds_fetch_diagnostic_summary(),
        "odds_quality_counts": {
            "OK": 0,
            "SUSPICIOUS": 0,
            "UNAVAILABLE": 0,
            "MISSING": 0,
        },
        "candidate_match_summary": {},
        "top_likely_issues": [],
        "unavailable_games_sample": [],
        "prediction_diagnostics": [],
        "recommendations": [],
    }

    prediction_report, prediction_error = _safe_read_json(Path(prediction_path))
    diagnostic["prediction_report"]["exists"] = prediction_report is not None
    diagnostic["prediction_report"]["error"] = prediction_error

    predictions: List[Dict[str, Any]] = []
    if prediction_report is not None:
        predictions = _extract_prediction_rows(prediction_report)
        diagnostic["prediction_report"]["prediction_count"] = int(len(predictions))
        diagnostic["prediction_report"]["scheduled_game_count"] = int(
            prediction_report.get("scheduled_game_count") or 0
        )
        diagnostic["prediction_report"]["generated_at"] = _json_safe(
            prediction_report.get("generated_at")
        )
        diagnostic["prediction_report"]["report_date"] = _json_safe(
            prediction_report.get("report_date")
            or prediction_report.get("date")
        )
        diagnostic["odds_quality_counts"] = _odds_quality_counts(predictions)

    odds_frame, odds_error = _safe_read_csv(Path(market_odds_path))
    market_odds_file_exists = odds_frame is not None
    market_odds_has_rows = odds_frame is not None and not odds_frame.empty

    diagnostic["market_odds_history"]["exists"] = market_odds_file_exists
    diagnostic["market_odds_history"]["error"] = odds_error

    if odds_frame is not None:
        diagnostic["market_odds_history"]["total_rows"] = int(len(odds_frame))
        diagnostic["market_odds_history"]["columns"] = list(odds_frame.columns)

        moneyline = _filter_moneyline(odds_frame)
        diagnostic["market_odds_history"]["moneyline_rows"] = int(len(moneyline))

        if _column_exists(odds_frame, "is_closing_snapshot"):
            closing_text = odds_frame["is_closing_snapshot"].astype(str).str.lower()
            diagnostic["market_odds_history"]["closing_rows"] = int(
                closing_text.isin(["true", "1", "yes"]).sum()
            )

    context_frame, context_error = _safe_read_csv(Path(context_path))
    diagnostic["daily_context"]["exists"] = context_frame is not None
    diagnostic["daily_context"]["error"] = context_error
    diagnostic["daily_context"]["total_rows"] = _safe_count(context_frame)

    snapshots_frame, snapshots_error = _safe_read_csv(Path(snapshots_path))
    diagnostic["prediction_snapshots"]["exists"] = snapshots_frame is not None
    diagnostic["prediction_snapshots"]["error"] = snapshots_error
    diagnostic["prediction_snapshots"]["total_rows"] = _safe_count(snapshots_frame)

    prediction_diagnostics: List[Dict[str, Any]] = []

    for prediction in predictions:
        odds_status = _string_key(prediction.get("odds_quality_status")).upper()
        if not odds_status:
            odds_status = "MISSING"

        candidates = pd.DataFrame()
        if odds_frame is not None and not odds_frame.empty:
            candidates = _find_candidate_odds_for_prediction(
                prediction,
                odds_frame,
            )

        candidate_summary = _summarize_candidate_odds(
            prediction,
            candidates,
        )

        likely_issue = _determine_likely_issue(
            odds_status=odds_status,
            market_odds_file_exists=market_odds_file_exists,
            market_odds_has_rows=market_odds_has_rows,
            candidate_summary=candidate_summary,
        )

        item: Dict[str, Any] = {
            "game_id": _string_key(prediction.get("game_id")),
            "game_date": _json_safe(prediction.get("game_date")),
            "home_team": _json_safe(prediction.get("home_team")),
            "away_team": _json_safe(prediction.get("away_team")),
            "normalized_home_team": _normalize_team_name(prediction.get("home_team")),
            "normalized_away_team": _normalize_team_name(prediction.get("away_team")),
            "odds_quality_status": odds_status,
            "suspicious_odds_reason": _json_safe(
                prediction.get("suspicious_odds_reason")
            ),
            "moneyline_gate_status": _json_safe(
                prediction.get("moneyline_gate_status")
            ),
            "candidate_count": candidate_summary["candidate_count"],
            "candidate_by_game_id_count": candidate_summary["candidate_by_game_id_count"],
            "candidate_by_team_name_count": candidate_summary["candidate_by_team_name_count"],
            "candidate_moneyline_count": candidate_summary["candidate_moneyline_count"],
            "candidate_closing_count": candidate_summary["candidate_closing_count"],
            "candidate_sportsbooks": candidate_summary["candidate_sportsbooks"],
            "candidate_markets": candidate_summary["candidate_markets"],
            "has_home_price": candidate_summary["has_home_price"],
            "has_away_price": candidate_summary["has_away_price"],
            "likely_issue": likely_issue,
            "sample_candidates": candidate_summary["sample_candidates"],
        }

        prediction_diagnostics.append(item)

    diagnostic["prediction_diagnostics"] = prediction_diagnostics

    issue_counter = Counter(
        item["likely_issue"]
        for item in prediction_diagnostics
    )

    odds_fetch_summary = diagnostic.get("odds_fetch_diagnostic", {})
    if isinstance(odds_fetch_summary, dict) and odds_fetch_summary.get("exists"):
        try:
            final_event_count = int(odds_fetch_summary.get("final_event_count") or 0)
        except Exception:
            final_event_count = 0

        try:
            final_usable_row_count = int(
                odds_fetch_summary.get("final_usable_row_count") or 0
            )
        except Exception:
            final_usable_row_count = 0

        if final_event_count == 0:
            issue_counter["provider_returned_zero_events"] += 1

        elif final_event_count > 0 and final_usable_row_count == 0:
            issue_counter["parser_dropped_all_provider_events"] += 1

        elif (
            final_usable_row_count > 0
            and diagnostic["odds_quality_counts"].get("OK", 0) == 0
        ):
            issue_counter["schedule_matching_failed_after_fetch"] += 1

        if final_usable_row_count > 0 and not market_odds_has_rows:
            issue_counter["market_odds_history_not_persisted"] += 1

    diagnostic["top_likely_issues"] = [
        {"issue": issue, "count": int(count)}
        for issue, count in issue_counter.most_common()
    ]

    diagnostic["unavailable_games_sample"] = [
        item
        for item in prediction_diagnostics
        if item["odds_quality_status"] == "UNAVAILABLE"
    ][:5]

    if prediction_diagnostics:
        diagnostic["candidate_match_summary"] = {
            "games_with_candidates": int(
                sum(1 for item in prediction_diagnostics if item["candidate_count"] > 0)
            ),
            "games_with_moneyline_candidates": int(
                sum(1 for item in prediction_diagnostics if item["candidate_moneyline_count"] > 0)
            ),
            "games_with_complete_home_away_prices": int(
                sum(
                    1
                    for item in prediction_diagnostics
                    if item["has_home_price"] and item["has_away_price"]
                )
            ),
            "games_without_candidates": int(
                sum(1 for item in prediction_diagnostics if item["candidate_count"] == 0)
            ),
        }

    diagnostic["recommendations"] = _build_recommendations(
        diagnostic,
        issue_counter,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            diagnostic,
            file,
            indent=2,
            ensure_ascii=True,
            default=str,
        )

    return diagnostic


if __name__ == "__main__":
    result = build_odds_matching_diagnostic()

    summary = {
        "generated_at": result["generated_at"],
        "prediction_report": result["prediction_report"],
        "market_odds_history": {
            "exists": result["market_odds_history"]["exists"],
            "total_rows": result["market_odds_history"]["total_rows"],
            "moneyline_rows": result["market_odds_history"]["moneyline_rows"],
            "closing_rows": result["market_odds_history"]["closing_rows"],
        },
        "odds_fetch_diagnostic": result.get("odds_fetch_diagnostic", {}),
        "odds_quality_counts": result["odds_quality_counts"],
        "candidate_match_summary": result["candidate_match_summary"],
        "top_likely_issues": result["top_likely_issues"][:5],
        "recommendations": result["recommendations"],
        "full_report_written_to": "report/odds_matching_diagnostic.json",
    }

    print(json.dumps(summary, indent=2, ensure_ascii=True, default=str))
