from __future__ import annotations

import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPORT_DIR = Path("report")
DATA_DIR = Path("data")
OUTPUT_PATH = REPORT_DIR / "artifact_retention_manifest.json"

SCAN_PATTERNS = [
    "report/*.json",
    "report/*.csv",
    "report/index.html",
    "data/*.json",
    "data/*.csv",
    "data/*.pkl",
]

LARGE_FILE_BYTES = 5 * 1024 * 1024


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


def _modified_at(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _classify(path: Path, size_bytes: int) -> tuple[str, str, str]:
    path_text = str(path).replace("\\", "/").lower()
    suffix = path.suffix.lower()

    if size_bytes > LARGE_FILE_BYTES:
        return "generated_large", "external_storage", "large generated artifact; avoid committing to git"

    if suffix == ".pkl":
        return "critical", "github_artifact", "model artifact; prefer artifact or registry storage"

    if path_text in {
        "report/prediction.json",
        "report/data_contract_report.json",
        "report/pipeline_manifest.json",
        "data/training_status.json",
    }:
        return "critical", "git", "critical audit/report file"

    if suffix in {".json", ".csv"}:
        return "important", "git", "small report/data file can be committed"

    if suffix in {".html"}:
        return "important", "github_artifact", "dashboard output"

    return "optional", "ignore", "optional generated file"


def _file_record(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    size = path.stat().st_size if exists and path.is_file() else 0
    retention_class, storage, notes = _classify(path, size)

    return {
        "path": str(path).replace("\\", "/"),
        "exists": exists,
        "size_bytes": size,
        "sha256": _sha256(path),
        "modified_at": _modified_at(path),
        "retention_class": retention_class,
        "recommended_storage": storage,
        "notes": notes,
    }


def build_manifest() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    discovered: List[Path] = []
    for pattern in SCAN_PATTERNS:
        discovered.extend(Path(match) for match in glob.glob(pattern))

    unique_paths = sorted(set(discovered), key=lambda item: str(item))
    records = [_file_record(path) for path in unique_paths]

    large_files = [
        record["path"]
        for record in records
        if record["retention_class"] == "generated_large"
    ]

    report = {
        "generated_at": _utc_now(),
        "status": "ok",
        "input_files": {"scan_patterns": SCAN_PATTERNS},
        "file_count": len(records),
        "large_file_count": len(large_files),
        "large_files": large_files,
        "files": records,
        "errors": [],
        "warnings": [],
        "recommendations": [
            "Store large generated data outside git when possible.",
            "Keep critical report JSONs in git or workflow artifacts for auditability.",
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    report = build_manifest()
    print(
        json.dumps(
            {
                "status": report["status"],
                "file_count": report["file_count"],
                "large_file_count": report["large_file_count"],
                "output_path": str(OUTPUT_PATH),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
