from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


REPORT_DIR = Path("report")
OUTPUT_PATH = REPORT_DIR / "pipeline_manifest.json"

TRACKED_FILES = [
    "report/prediction.json",
    "report/evaluation_clv_diagnostic.json",
    "report/market_edge_research.json",
    "report/feature_availability_diagnostic.json",
    "report/feature_zero_root_cause_diagnostic.json",
    "report/feature_grade_report.json",
    "report/baseline_comparison_report.json",
    "report/clv_by_edge_bucket.json",
    "report/clv_by_side.json",
    "report/clv_by_odds_range.json",
    "report/clv_by_lineup_status.json",
    "report/calibration_report.json",
    "report/walkforward_evaluation.json",
    "report/rolling_walkforward_evaluation.json",
    "report/rolling_walkforward_predictions.csv",
    "report/lineup_starter_slice_report.json",
    "report/market_close_report.json",
    "report/research_quality_report.json",
    "report/settle_reliability_report.json",
    "report/model_registry_report.json",
    "report/promotion_gate_report.json",
    "report/decision_audit_report.json",
    "report/decision_audit.csv",
    "report/paper_trading_ledger_report.json",
    "report/risk_exposure_report.json",
    "report/artifact_retention_manifest.json",
    "report/data_contract_report.json",
    "report/pipeline_manifest.json",
    "report/index.html",
    "report/walkforward_pr:contentReference[oaicite:4]{index=4}on 爆掉的大錯，但工程級 manifest 最好要自我追蹤。

---

# 修改 3：清掉重複的 `_as_int`

## 檔案：`scripts/clv_slice_report.py`

目前 `_as_int()` 被定義了兩次，雖然不會直接造成 crash，但應該清掉。fileciteturn663file0L48-L59

你現在有：

```python
def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    parsed = _as_float(value)
    if parsed is None:
        return default
    return int(parsed)


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    parsed = _as_float(value)
    if parsed is None:
        return default
    return int(parsed)
    "data/model_registry.json",
    "data/paper_trading_ledger.csv",
    "data/training_status.json",
    "data/prediction_snapshots.csv",
    "data/market_odds_history.csv",
    "data/finalized_games.csv",
    "data/daily_game_context.csv",
    "data/weather_context.csv",
    "data/top3_player_context.csv",
    "data/savant_top3_context.csv",
    "data/pitcher_advanced_context.csv",
    "data/team_form_context.csv",
    "data/context_feature_bridge.csv",
    "data/projected_lineup_context.csv",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_summary(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"json_valid": False, "json_error": str(exc)}

    if not isinstance(data, dict):
        return {"json_valid": True, "json_type": type(data).__name__}

    summary: Dict[str, Any] = {
        "json_valid": True,
        "json_type": "dict",
    }

    for key in ("generated_at", "status", "error_count", "warning_count"):
        if key in data:
            summary[key] = data.get(key)

    predictions = data.get("predictions") or data.get("today_predictions") or data.get("games")
    if isinstance(predictions, list):
        summary["prediction_count"] = len(predictions)

    slices = data.get("slices")
    if isinstance(slices, list):
        summary["slice_count"] = len(slices)

    bins = data.get("bins")
    if isinstance(bins, list):
        summary["bin_count"] = len(bins)

    return summary


def _csv_row_count(path: Path) -> Optional[int]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            rows = sum(1 for _ in reader)
    except Exception:
        return None

    return max(0, rows - 1)


def _file_record(path_text: str) -> Dict[str, Any]:
    path = Path(path_text)
    record: Dict[str, Any] = {
        "path": path_text,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        "sha256": _sha256(path),
    }

    if path.suffix.lower() == ".json" and path.exists():
        record.update(_json_summary(path))

    if path.suffix.lower() == ".csv" and path.exists():
        record["row_count"] = _csv_row_count(path)

    return record


def build_manifest() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = [_file_record(path) for path in TRACKED_FILES]
    missing = [item["path"] for item in files if not item["exists"]]

    report = {
        "generated_at": _utc_now(),
        "status": "ok" if not missing else "partial",
        "tracked_file_count": len(files),
        "missing_file_count": len(missing),
        "missing_files": missing,
        "files": files,
        "recommendations": []
        if not missing
        else [
            "Some tracked files are missing; check whether they are optional, not generated yet, or failed upstream."
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    manifest = build_manifest()
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "tracked_file_count": manifest["tracked_file_count"],
                "missing_file_count": manifest["missing_file_count"],
                "output_path": str(OUTPUT_PATH),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
