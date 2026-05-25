# scripts/validate_predictions.py
"""Validate an already-generated prediction report.

This script must not fetch data or generate new predictions. It only verifies
the output written by prediction.py and fails CI when the report contains
strong signals of broken upstream data or implausible prediction distributions.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPORT = Path("report/prediction.json")
VALIDATION_ERRORS_FILE = Path("report/validation_errors.txt")
ELO_FILE = Path("data/elo_ratings.json")

MIN_GAMES_FOR_DAILY_MEAN_CHECK = 5
MIN_ALLOWED_MEAN_HOME_PROB = 0.40
MAX_ALLOWED_MEAN_HOME_PROB = 0.60
MAX_ALLOWED_ELO_RANGE = 400.0
EXTREME_HIGH_PROB = 0.80
EXTREME_LOW_PROB = 0.20

CRITICAL_ERROR_TOKENS = (
    "traceback",
    "嚴重錯誤",
    "严重错误",
    "critical",
    "fatal",
    "nameerror",
    "truth value of a series is ambiguous",
    "not enough values to unpack",
    "lag_features error",
)


def _write_failure(errors: list[str], pipeline_errors: list[str]) -> None:
    VALIDATION_ERRORS_FILE.parent.mkdir(parents=True, exist_ok=True)
    output = list(errors)
    if pipeline_errors:
        output.extend(["", "Pipeline errors:"])
        output.extend(f"- {item}" for item in pipeline_errors)
    VALIDATION_ERRORS_FILE.write_text("\n".join(output), encoding="utf-8")


def _fail(errors: list[str], pipeline_errors: list[str] | None = None) -> None:
    pipeline_errors = pipeline_errors or []
    print("❌ 驗證失敗:")
    for error in errors:
        print(f"  - {error}")
    if pipeline_errors:
        print("pipeline errors 摘要:")
        for error in pipeline_errors[-10:]:
            print(f"  - {error}")
    _write_failure(errors, pipeline_errors)
    raise SystemExit(1)


def _as_finite_probability(value: Any, label: str, errors: list[str]) -> float | None:
    try:
        prob = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} 缺少有效數值: {value!r}")
        return None
    if not math.isfinite(prob) or prob < 0.0 or prob > 1.0:
        errors.append(f"{label} 超出 [0, 1] 範圍: {prob}")
        return None
    return prob


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except FileNotFoundError:
        raise SystemExit(f"❌ {path} 不存在")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"❌ {path} JSON 無法解析: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"❌ {path} 內容必須是 JSON object")
    return value


def validate() -> None:
    data = _read_json(REPORT)
    predictions = data.get("today_predictions", [])
    pipeline_errors = [str(item) for item in data.get("errors", []) if item is not None]
    schedule_fetch_ok = data.get("schedule_fetch_ok")
    scheduled_game_count = data.get("scheduled_game_count")

    if not isinstance(predictions, list):
        _fail(["today_predictions 必須為 list"], pipeline_errors)

    if not predictions:
        if schedule_fetch_ok is True and scheduled_game_count == 0:
            print("✅ 賽程來源確認今日無比賽，無預測屬正常情況")
            return
        _fail([
            "預測為空，但 report 未明確確認 schedule_fetch_ok=true 且 "
            "scheduled_game_count=0；不得將 API/管線失敗誤判為休賽日"
        ], pipeline_errors)

    validation_errors: list[str] = []
    if schedule_fetch_ok is not True:
        validation_errors.append("已有預測，但 schedule_fetch_ok 不是 true")
    if scheduled_game_count is not None and scheduled_game_count < len(predictions):
        validation_errors.append("預測場數大於排程場數")

    critical_pipeline_errors = [
        error for error in pipeline_errors
        if any(token in error.lower() for token in CRITICAL_ERROR_TOKENS)
    ]
    if critical_pipeline_errors:
        validation_errors.append(f"存在 {len(critical_pipeline_errors)} 條關鍵 pipeline 錯誤")

    home_probs: list[float] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            validation_errors.append(f"第 {index + 1} 場預測不是 object")
            continue
        probability = _as_finite_probability(
            prediction.get("predicted_home_win_pct"),
            f"第 {index + 1} 場 predicted_home_win_pct",
            validation_errors,
        )
        if probability is not None:
            home_probs.append(probability)

    if len(home_probs) != len(predictions):
        _fail(validation_errors, pipeline_errors)

    mean_home = float(np.mean(home_probs))
    min_home = float(np.min(home_probs))
    max_home = float(np.max(home_probs))
    extreme_games = sum(prob > EXTREME_HIGH_PROB or prob < EXTREME_LOW_PROB for prob in home_probs)

    print(
        "Home probability diagnostics: "
        f"games={len(home_probs)}, mean={mean_home:.3f}, "
        f"min={min_home:.3f}, max={max_home:.3f}, extreme_games={extreme_games}"
    )

    if len(home_probs) >= MIN_GAMES_FOR_DAILY_MEAN_CHECK:
        if mean_home > MAX_ALLOWED_MEAN_HOME_PROB or mean_home < MIN_ALLOWED_MEAN_HOME_PROB:
            validation_errors.append(f"Avg home prob {mean_home:.3f} 異常")
    else:
        print(f"僅 {len(home_probs)} 場比賽，跳過 daily mean 檢查")

    if ELO_FILE.exists():
        try:
            elo = _read_json(ELO_FILE)
            numeric_ratings = [float(value) for value in elo.values()]
            if numeric_ratings:
                elo_min = min(numeric_ratings)
                elo_max = max(numeric_ratings)
                elo_range = elo_max - elo_min
                print(
                    "Elo diagnostics: "
                    f"teams={len(numeric_ratings)}, min={elo_min:.1f}, "
                    f"max={elo_max:.1f}, range={elo_range:.1f}"
                )
                if elo_range > MAX_ALLOWED_ELO_RANGE:
                    validation_errors.append(f"Elo range {elo_range:.1f} 過大")
        except (TypeError, ValueError) as exc:
            validation_errors.append(f"Elo rating 檔案含非數值資料: {exc}")

    model_sources = Counter(
        str(prediction.get("model_source", "unknown"))
        for prediction in predictions if isinstance(prediction, dict)
    )
    print(f"Model source distribution: {dict(model_sources)}")
    if model_sources and model_sources.get("manual", 0) == len(predictions):
        print("ℹ️ 所有比賽目前使用手工預測 baseline")

    valid_nrfi = [
        prediction for prediction in predictions
        if isinstance(prediction, dict)
        and prediction.get("nrfi_source") in {"ml", "manual"}
        and prediction.get("nrfi_prob") is not None
    ]
    nrfi_sources = Counter(
        str(prediction.get("nrfi_source", "unknown"))
        for prediction in predictions if isinstance(prediction, dict)
    )
    print(f"NRFI source distribution: {dict(nrfi_sources)}")

    nrfi_probs: list[float] = []
    for index, prediction in enumerate(valid_nrfi):
        probability = _as_finite_probability(
            prediction.get("nrfi_prob"),
            f"有效 NRFI 第 {index + 1} 場 nrfi_prob",
            validation_errors,
        )
        if probability is not None:
            nrfi_probs.append(probability)
    if len(nrfi_probs) > 2 and all(abs(prob - 0.5) < 0.01 for prob in nrfi_probs):
        validation_errors.append("NRFI 機率單一（全部≈0.5）")

    if validation_errors:
        _fail(validation_errors, pipeline_errors)
    if VALIDATION_ERRORS_FILE.exists():
        VALIDATION_ERRORS_FILE.unlink()
    print("✅ 預測輸出通過自動驗證")


if __name__ == "__main__":
    validate()
