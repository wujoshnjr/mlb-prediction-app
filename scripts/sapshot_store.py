# scripts/snapshot_store.py
"""Clean pregame snapshot and settlement storage for baseline_v2_clean.

This store is forward-collected only:
- A game is eligible only when observed before its scheduled start time.
- Only the first valid pregame snapshot per pipeline version and game is kept.
- Settlement writes final results only; it never recomputes pregame features.
- Legacy historical/backfilled rows are not treated as clean training samples.

Runtime strings and comments are ASCII-only for safe browser-based editing.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from scripts.feature_schema import EXPECTED_FEATURES

try:
    import config
except ImportError:
    config = None  # type: ignore[assignment]


PIPELINE_VERSION = str(
    getattr(config, "PIPELINE_VERSION", "baseline_v2_clean")
)
SNAPSHOT_POLICY = str(
    getattr(config, "SNAPSHOT_POLICY", "first_seen_pregame")
)
BETTING_MODE = str(
    getattr(config, "BETTING_MODE", "paper_trading")
)
SNAPSHOT_STORE_FILE = Path(
    str(getattr(config, "SNAPSHOT_STORE_FILE", "data/prediction_snapshots.csv"))
)

PREGAME_STATUSES = {
    "preview",
    "scheduled",
    "pre-game",
    "pre game",
    "pregame",
    "warmup",
    "warm up",
}

BASE_COLUMNS = [
    "snapshot_id",
    "pipeline_version",
    "snapshot_policy",
    "betting_mode",
    "snapshot_created_at",
    "snapshot_valid",
    "snapshot_invalid_reason",
    "game_id",
    "game_date",
    "start_time",
    "game_status",
    "home_team",
    "away_team",
    "rating_source",
    "model_source",
    "premarket_model_home_prob",
    "displayed_home_win_pct",
    "predicted_home_win_pct",
    "manual_no_odds_pred",
    "home_moneyline_odds",
    "away_moneyline_odds",
    "market_no_vig_home_prob",
    "model_edge_home",
    "spread_line",
    "total_line",
    "odds_quality_status",
    "suspicious_odds_reason",
    "odds_source",
    "bookmaker_quotes_json",
    "market_adjustment_applied",
    "recommendation_status",
    "moneyline_recommendation",
    "spread_recommendation",
    "total_recommendation",
    "home_kelly_fraction",
    "away_kelly_fraction",
    "over_prob",
    "home_cover_prob",
    "away_cover_prob",
    "nrfi_prob",
    "nrfi_recommendation",
    "settled_at",
    "home_win",
    "home_score",
    "away_score",
    "closing_home_odds",
    "closing_away_odds",
    "closing_spread_line",
    "closing_total_line",
    "clv_home_moneyline",
]

FEATURE_COLUMNS = [
    feature for feature in EXPECTED_FEATURES if feature not in BASE_COLUMNS
]
SNAPSHOT_COLUMNS = BASE_COLUMNS + FEATURE_COLUMNS


def utc_now() -> datetime:
    """Return current time as an aware UTC datetime."""
    return datetime.now(timezone.utc)


def to_utc_iso(value: datetime | None = None) -> str:
    """Return an unambiguous UTC timestamp with a Z suffix."""
    timestamp = (value or utc_now()).astimezone(timezone.utc)
    return timestamp.isoformat().replace("+00:00", "Z")


def parse_utc_datetime(value: Any) -> datetime | None:
    """Parse API timestamp values and normalize them to aware UTC datetimes."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def normalize_game_id(value: Any) -> str:
    """Normalize game identifiers so CSV and API values match reliably."""
    if value is None:
        return ""

    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "null"}:
        return ""
    return text


def stringify(value: Any) -> str:
    """Convert scalar values to stable CSV-safe strings."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def as_decimal_odds(value: Any) -> float | None:
    """Return valid decimal odds or None."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 1.0 else None


def as_probability(value: Any) -> float | None:
    """Return a numeric probability within [0, 1], or None."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if 0.0 <= parsed <= 1.0 else None


def as_float_or_none(value: Any) -> float | None:
    """Return a finite numeric value, including negative values, or None."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(parsed) or parsed in (float("inf"), float("-inf")):
        return None

    return parsed


def compute_market_no_vig_home_prob(
    home_odds: Any,
    away_odds: Any,
) -> float | None:
    """Compute two-way no-vig home probability from decimal moneyline odds."""
    home_decimal = as_decimal_odds(home_odds)
    away_decimal = as_decimal_odds(away_odds)
    if home_decimal is None or away_decimal is None:
        return None

    home_raw = 1.0 / home_decimal
    away_raw = 1.0 / away_decimal
    denominator = home_raw + away_raw
    if denominator <= 0:
        return None

    return home_raw / denominator


def snapshot_key(pipeline_version: str, game_id: Any) -> str:
    """Return canonical unique identifier for a clean first-seen snapshot."""
    return f"{pipeline_version}:{normalize_game_id(game_id)}"


def read_all_snapshot_rows(
    path: Path = SNAPSHOT_STORE_FILE,
) -> list[dict[str, str]]:
    """Read all stored rows without filtering."""
    if not path.exists():
        return []

    try:
        with path.open("r", newline="", encoding="utf-8") as file_obj:
            return [dict(row) for row in csv.DictReader(file_obj)]
    except OSError as exc:
        print(f"Unable to read snapshot store {path}: {exc}")
        return []


def write_snapshot_rows(
    rows: list[dict[str, str]],
    path: Path = SNAPSHOT_STORE_FILE,
) -> None:
    """Atomically rewrite the CSV using the canonical column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=SNAPSHOT_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: row.get(column, "") for column in SNAPSHOT_COLUMNS}
            )

    temporary_path.replace(path)


def validate_pregame_prediction(
    prediction: dict[str, Any],
    snapshot_created_at: datetime | None = None,
) -> tuple[bool, str]:
    """Confirm a prediction was observed before first pitch."""
    observed_at = (snapshot_created_at or utc_now()).astimezone(timezone.utc)

    if not normalize_game_id(prediction.get("game_id")):
        return False, "missing_game_id"

    start_time = parse_utc_datetime(
        prediction.get("start_time")
        or prediction.get("game_datetime")
    )
    if start_time is None:
        return False, "missing_or_invalid_start_time"

    if observed_at >= start_time:
        return False, "snapshot_created_after_game_start"

    status = str(
        prediction.get("game_status")
        or prediction.get("status")
        or ""
    ).strip().lower()
    if status not in PREGAME_STATUSES:
        return False, f"not_confirmed_pregame_status:{status or 'missing'}"

    return True, ""


def build_snapshot_row(
    prediction: dict[str, Any],
    snapshot_created_at: datetime | None = None,
) -> dict[str, str]:
    """Flatten one runtime prediction into the canonical clean schema."""
    observed_at = (snapshot_created_at or utc_now()).astimezone(timezone.utc)
    valid, invalid_reason = validate_pregame_prediction(
        prediction,
        snapshot_created_at=observed_at,
    )
    game_id = normalize_game_id(prediction.get("game_id"))

    market_probability = as_probability(
        prediction.get("market_no_vig_home_prob")
    )
    if market_probability is None:
        market_probability = compute_market_no_vig_home_prob(
            prediction.get("home_moneyline_odds"),
            prediction.get("away_moneyline_odds"),
        )

    premarket_probability = as_probability(
        prediction.get("premarket_model_home_prob")
    )
    displayed_probability = as_probability(
        prediction.get("displayed_home_win_pct")
    )
    if displayed_probability is None:
        displayed_probability = as_probability(
            prediction.get("predicted_home_win_pct")
        )

      model_edge = as_float_or_none(prediction.get("model_edge_home"))
    if model_edge is None and (
        premarket_probability is not None and market_probability is not None
    ):
        model_edge = premarket_probability - market_probability

    row: dict[str, str] = {column: "" for column in SNAPSHOT_COLUMNS}
    row.update(
        {
            "snapshot_id": snapshot_key(PIPELINE_VERSION, game_id),
            "pipeline_version": PIPELINE_VERSION,
            "snapshot_policy": SNAPSHOT_POLICY,
            "betting_mode": stringify(
                prediction.get("betting_mode") or BETTING_MODE
            ),
            "snapshot_created_at": to_utc_iso(observed_at),
            "snapshot_valid": "true" if valid else "false",
            "snapshot_invalid_reason": invalid_reason,
            "game_id": game_id,
            "game_date": stringify(prediction.get("game_date")),
            "start_time": stringify(
                prediction.get("start_time")
                or prediction.get("game_datetime")
            ),
            "game_status": stringify(
                prediction.get("game_status")
                or prediction.get("status")
            ),
            "home_team": stringify(prediction.get("home_team")),
            "away_team": stringify(prediction.get("away_team")),
            "rating_source": stringify(prediction.get("rating_source")),
            "model_source": stringify(prediction.get("model_source")),
            "premarket_model_home_prob": stringify(premarket_probability),
            "displayed_home_win_pct": stringify(displayed_probability),
            "predicted_home_win_pct": stringify(displayed_probability),
            "manual_no_odds_pred": stringify(
                prediction.get("manual_no_odds_pred")
            ),
            "home_moneyline_odds": stringify(
                prediction.get("home_moneyline_odds")
            ),
            "away_moneyline_odds": stringify(
                prediction.get("away_moneyline_odds")
            ),
            "market_no_vig_home_prob": stringify(market_probability),
            "model_edge_home": stringify(model_edge),
            "spread_line": stringify(prediction.get("spread_line")),
            "total_line": stringify(prediction.get("total_line")),
            "odds_quality_status": stringify(
                prediction.get("odds_quality_status") or "UNAVAILABLE"
            ),
            "suspicious_odds_reason": stringify(
                prediction.get("suspicious_odds_reason")
            ),
            "odds_source": stringify(prediction.get("odds_source")),
            "bookmaker_quotes_json": json.dumps(
                prediction.get("bookmaker_quotes", []),
                separators=(",", ":"),
                ensure_ascii=True,
            ),
            "market_adjustment_applied": stringify(
                prediction.get("market_adjustment_applied")
            ),
            "recommendation_status": stringify(
                prediction.get("recommendation_status") or "TRACKING_ONLY"
            ),
            "moneyline_recommendation": stringify(
                prediction.get("moneyline_recommendation") or "NO BET"
            ),
            "spread_recommendation": stringify(
                prediction.get("spread_recommendation") or "NO BET"
            ),
            "total_recommendation": stringify(
                prediction.get("total_recommendation") or "NO BET"
            ),
            "home_kelly_fraction": stringify(
                prediction.get("home_kelly_fraction")
            ),
            "away_kelly_fraction": stringify(
                prediction.get("away_kelly_fraction")
            ),
            "over_prob": stringify(prediction.get("over_prob")),
            "home_cover_prob": stringify(prediction.get("home_cover_prob")),
            "away_cover_prob": stringify(prediction.get("away_cover_prob")),
            "nrfi_prob": stringify(prediction.get("nrfi_prob")),
            "nrfi_recommendation": stringify(
                prediction.get("nrfi_recommendation") or "NO DATA"
            ),
        }
    )

    features = prediction.get("features", {})
    if not isinstance(features, dict):
        features = {}

    for feature in FEATURE_COLUMNS:
        row[feature] = stringify(features.get(feature))

    return row


def append_first_seen_pregame_snapshots(
    predictions: Iterable[dict[str, Any]],
    path: Path = SNAPSHOT_STORE_FILE,
    snapshot_created_at: datetime | None = None,
) -> dict[str, Any]:
    """Persist only the first valid pregame snapshot for each game."""
    observed_at = (snapshot_created_at or utc_now()).astimezone(timezone.utc)
    rows = read_all_snapshot_rows(path)
    existing_keys = {
        snapshot_key(
            row.get("pipeline_version", ""),
            row.get("game_id", ""),
        )
        for row in rows
    }

    total_games = 0
    inserted = 0
    duplicates = 0
    skipped: dict[str, int] = {}

    for prediction in predictions:
        total_games += 1
        if not isinstance(prediction, dict):
            skipped["malformed_prediction"] = (
                skipped.get("malformed_prediction", 0) + 1
            )
            continue

        row = build_snapshot_row(
            prediction,
            snapshot_created_at=observed_at,
        )
        if row["snapshot_valid"] != "true":
            reason = row["snapshot_invalid_reason"] or "invalid"
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        if row["snapshot_id"] in existing_keys:
            duplicates += 1
            continue

        rows.append(row)
        existing_keys.add(row["snapshot_id"])
        inserted += 1

    if inserted:
        write_snapshot_rows(rows, path)

    return {
        "pipeline_version": PIPELINE_VERSION,
        "snapshot_policy": SNAPSHOT_POLICY,
        "total_games": total_games,
        "inserted": inserted,
        "duplicates": duplicates,
        "skipped": skipped,
        "stored_rows": len(rows),
    }


def settle_snapshots(
    final_games: Iterable[dict[str, Any]],
    path: Path = SNAPSHOT_STORE_FILE,
    settled_at: datetime | None = None,
) -> dict[str, Any]:
    """Backfill results only, preserving all original pregame values."""
    rows = read_all_snapshot_rows(path)
    if not rows:
        return {
            "pipeline_version": PIPELINE_VERSION,
            "final_games": 0,
            "updated": 0,
            "unmatched": 0,
            "stored_rows": 0,
        }

    results: dict[str, dict[str, Any]] = {}
    for result in final_games:
        if not isinstance(result, dict):
            continue
        game_id = normalize_game_id(result.get("game_id"))
        if game_id:
            results[game_id] = result

    matched_ids: set[str] = set()
    updated = 0
    settlement_time = to_utc_iso(settled_at)

    for row in rows:
        if row.get("pipeline_version") != PIPELINE_VERSION:
            continue
        if row.get("snapshot_valid", "").strip().lower() != "true":
            continue

        game_id = normalize_game_id(row.get("game_id"))
        result = results.get(game_id)
        if result is None:
            continue

        matched_ids.add(game_id)
        if row.get("home_win", "").strip():
            continue

        try:
            home_score = int(float(result.get("home_score")))
            away_score = int(float(result.get("away_score")))
        except (TypeError, ValueError):
            continue

        home_win_value = result.get("home_win")
        if str(home_win_value).strip() not in {"0", "1"}:
            home_win_value = 1 if home_score > away_score else 0

        row["settled_at"] = settlement_time
        row["home_win"] = str(home_win_value)
        row["home_score"] = str(home_score)
        row["away_score"] = str(away_score)
        updated += 1

    if updated:
        write_snapshot_rows(rows, path)

    return {
        "pipeline_version": PIPELINE_VERSION,
        "final_games": len(results),
        "updated": updated,
        "unmatched": len(set(results) - matched_ids),
        "stored_rows": len(rows),
    }


def read_snapshot_rows(
    pipeline_version: str = PIPELINE_VERSION,
    valid_only: bool = True,
    settled_only: bool = False,
    path: Path = SNAPSHOT_STORE_FILE,
) -> pd.DataFrame:
    """Read snapshot data for training or performance statistics."""
    if not path.exists():
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    frame = pd.read_csv(path, dtype=str)
    if frame.empty:
        return frame

    if "pipeline_version" in frame.columns:
        frame = frame[frame["pipeline_version"] == pipeline_version]

    if valid_only and "snapshot_valid" in frame.columns:
        frame = frame[
            frame["snapshot_valid"].astype(str).str.lower() == "true"
        ]

    if settled_only and "home_win" in frame.columns:
        frame = frame[
            frame["home_win"].notna()
            & (frame["home_win"].astype(str).str.strip() != "")
        ]

    numeric_columns = [
        "premarket_model_home_prob",
        "displayed_home_win_pct",
        "predicted_home_win_pct",
        "manual_no_odds_pred",
        "home_moneyline_odds",
        "away_moneyline_odds",
        "market_no_vig_home_prob",
        "model_edge_home",
        "spread_line",
        "total_line",
        "home_kelly_fraction",
        "away_kelly_fraction",
        "over_prob",
        "home_cover_prob",
        "away_cover_prob",
        "nrfi_prob",
        "home_win",
        "home_score",
        "away_score",
    ] + FEATURE_COLUMNS

    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame
