from __future__ import annotations

import ast
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(".")
REPORT_DIR = Path("report")
OUTPUT_PATH = REPORT_DIR / "repo_anomaly_report.json"

SCAN_EXTENSIONS = {".py", ".yml", ".yaml", ".json", ".html", ".md", ".txt"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", ".venv", "venv"}
MAX_FILE_SIZE_BYTES = 2_500_000
SELF_PATH = Path("scripts/repo_anomaly_report.py")

CRITICAL_PATHS = [
    "main.py",
    "prediction.py",
    "requirements.txt",
    ".github/workflows/fetch_data.yml",
    ".github/workflows/artifact_quarantine.yml",
    "scripts/model_eval_report.py",
    "scripts/baseline_comparison_report.py",
    "scripts/walkforward_evaluation.py",
    "scripts/feature_priority_report.py",
    "scripts/artifact_quarantine_report.py",
    "scripts/report_health_gate.py",
    "scripts/html_report_builder.py",
    "scripts/data_contract_validator.py",
    "scripts/pipeline_manifest.py",
]

QUALITY_STATUS_REPORTS = {
    "report/model_eval_report.json",
    "report/prediction_collapse_report.json",
    "report/baseline_comparison_report.json",
    "report/walkforward_evaluation.json",
    "report/feature_priority_report.json",
    "report/artifact_quarantine_report.json",
}

SAFETY_FLAGS = (
    "live_betting_allowed",
    "automated_wagering_allowed",
    "production_model_replacement_allowed",
)

SUSPICIOUS_TEXT_PATTERNS = {
    "conversation_artifact_marker": r"Skipped \d+ messages?|Citation Marker|Resource uri:",
    "copied_markdown_url_in_code": r"\]\(https?://",
    "placeholder_ellipsis": r"TODO|FIXME|pass  #|\.\.\.",
    "raw_tilde_separator": r"~{8,}",
    "potential_secret_literal": r"(?i)(api[_-]?key|secret|token|password)\s*=\s*['\"][^'\"]{8,}",
}

MODEL_QUALITY_BLOCK_STATUSES = {"failed", "blocked", "warning", "quarantined", "needs_review", "insufficient_samples"}
PIPELINE_FAILURE_STATUSES = {"error", "fatal"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(child) for child in value]
    return value if isinstance(value, str) else str(value)


def iter_scan_files() -> list[Path]:
    paths: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.parts)
        if parts & SKIP_DIRS:
            continue
        if path.suffix.lower() not in SCAN_EXTENSIONS:
            continue
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            continue
        paths.append(path)
    return sorted(paths, key=lambda item: str(item))


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None


def check_critical_paths() -> list[dict[str, Any]]:
    issues = []
    for item in CRITICAL_PATHS:
        path = Path(item)
        if not path.exists():
            issues.append({"severity": "error", "path": item, "issue": "critical_path_missing"})
    return issues


def check_python_syntax(paths: list[Path]) -> list[dict[str, Any]]:
    issues = []
    for path in paths:
        if path.suffix.lower() != ".py":
            continue
        text = read_text(path)
        if text is None:
            issues.append({"severity": "error", "path": str(path), "issue": "python_unreadable"})
            continue
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            issues.append(
                {
                    "severity": "error",
                    "path": str(path),
                    "issue": "python_syntax_error",
                    "line": exc.lineno,
                    "message": exc.msg,
                }
            )
    return issues


def check_json_files(paths: list[Path]) -> list[dict[str, Any]]:
    issues = []
    for path in paths:
        if path.suffix.lower() != ".json":
            continue
        text = read_text(path)
        if text is None:
            issues.append({"severity": "error", "path": str(path), "issue": "json_unreadable"})
            continue
        try:
            json.loads(text)
        except Exception as exc:
            issues.append({"severity": "error", "path": str(path), "issue": "json_invalid", "message": str(exc)})
    return issues


def check_safety_flags(paths: list[Path]) -> list[dict[str, Any]]:
    issues = []
    for path in paths:
        if path.suffix.lower() != ".json" or (not str(path).startswith("report") and not str(path).startswith("data")):
            continue
        text = read_text(path)
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for flag in SAFETY_FLAGS:
            if flag in payload and payload.get(flag) is not False:
                issues.append({"severity": "error", "path": str(path), "issue": "unsafe_safety_flag", "flag": flag, "value": payload.get(flag)})
    return issues


def check_quality_statuses(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues = []
    model_quality_blocks = []
    for report_path in QUALITY_STATUS_REPORTS:
        path = Path(report_path)
        if not path.exists():
            issues.append({"severity": "warning", "path": report_path, "issue": "quality_report_missing"})
            continue
        text = read_text(path)
        if not text:
            issues.append({"severity": "warning", "path": report_path, "issue": "quality_report_unreadable"})
            continue
        try:
            payload = json.loads(text)
        except Exception as exc:
            issues.append({"severity": "error", "path": report_path, "issue": "quality_report_invalid_json", "message": str(exc)})
            continue
        if not isinstance(payload, dict):
            issues.append({"severity": "warning", "path": report_path, "issue": "quality_report_not_object"})
            continue
        status = str(payload.get("status", "")).lower()
        if status in PIPELINE_FAILURE_STATUSES:
            issues.append({"severity": "error", "path": report_path, "issue": "pipeline_failure_status", "status": status})
        if status in MODEL_QUALITY_BLOCK_STATUSES or payload.get("promotion_allowed") is False or payload.get("do_not_promote") is True:
            model_quality_blocks.append(
                {
                    "path": report_path,
                    "status": status or None,
                    "promotion_allowed": payload.get("promotion_allowed"),
                    "do_not_promote": payload.get("do_not_promote"),
                }
            )
    return issues, model_quality_blocks


def _match_assignment_value(line: str) -> str | None:
    match = re.search(r"=\s*['\"]([^'\"]+)['\"]", line)
    return match.group(1) if match else None


def _is_env_var_placeholder(value: str | None) -> bool:
    if not value:
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]{3,}", value))


def check_text_patterns(paths: list[Path]) -> list[dict[str, Any]]:
    issues = []
    for path in paths:
        if path == SELF_PATH:
            continue
        text = read_text(path)
        if not text:
            continue
        lines = text.splitlines()
        for pattern_name, pattern in SUSPICIOUS_TEXT_PATTERNS.items():
            for match in re.finditer(pattern, text):
                line_no = text.count("\n", 0, match.start()) + 1
                line_text = lines[line_no - 1] if 0 <= line_no - 1 < len(lines) else ""
                if pattern_name == "potential_secret_literal" and _is_env_var_placeholder(_match_assignment_value(line_text)):
                    continue
                severity = "warning"
                if pattern_name == "potential_secret_literal":
                    severity = "error"
                if pattern_name == "placeholder_ellipsis" and path.suffix.lower() in {".md", ".txt"}:
                    continue
                issues.append({"severity": severity, "path": str(path), "issue": pattern_name, "line": line_no})
                break
    return issues


def check_workflows(paths: list[Path]) -> list[dict[str, Any]]:
    issues = []
    workflow_dir = Path(".github/workflows")
    if not workflow_dir.exists():
        return [{"severity": "error", "path": str(workflow_dir), "issue": "workflow_dir_missing"}]
    for path in sorted(workflow_dir.glob("*.yml")) + sorted(workflow_dir.glob("*.yaml")):
        text = read_text(path) or ""
        if "workflow_run:" in text:
            issues.append({"severity": "warning", "path": str(path), "issue": "workflow_run_trigger_present"})
        if "contents: write" in text and "workflow_dispatch" not in text and "github-actions[bot]" in text:
            issues.append({"severity": "warning", "path": str(path), "issue": "write_permission_without_manual_trigger_review"})
        if "git push" in text and "[skip ci]" not in text:
            issues.append({"severity": "warning", "path": str(path), "issue": "git_push_without_skip_ci_commit_message"})
    return issues


def check_html_builder() -> list[dict[str, Any]]:
    issues = []
    builder = Path("scripts/html_report_builder.py")
    text = read_text(builder) or ""
    required_tokens = ["Executive Snapshot", "Governance & Safety", "Model Quality", "Feature Roadmap", "Risk Disclosure"]
    for token in required_tokens:
        if token not in text:
            issues.append({"severity": "warning", "path": str(builder), "issue": "html_builder_missing_section", "section": token})
    if "viewport" not in text:
        issues.append({"severity": "warning", "path": str(builder), "issue": "html_builder_missing_viewport"})
    return issues


def build_report() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    paths = iter_scan_files()
    issues: list[dict[str, Any]] = []
    issues.extend(check_critical_paths())
    issues.extend(check_python_syntax(paths))
    issues.extend(check_json_files(paths))
    issues.extend(check_safety_flags(paths))
    quality_issues, model_quality_blocks = check_quality_statuses(paths)
    issues.extend(quality_issues)
    issues.extend(check_text_patterns(paths))
    issues.extend(check_workflows(paths))
    issues.extend(check_html_builder())

    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    status = "failed" if error_count else "warning" if warning_count or model_quality_blocks else "ok"

    report = {
        "generated_at": utc_now(),
        "report_type": "repo_anomaly_report",
        "status": status,
        "scanned_file_count": len(paths),
        "error_count": error_count,
        "warning_count": warning_count,
        "model_quality_block_count": len(model_quality_blocks),
        "issues": issues,
        "model_quality_blocks": model_quality_blocks,
        "recommendations": [],
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }
    if error_count:
        report["recommendations"].append("Fix repository anomaly errors before treating the pipeline as stable.")
    if warning_count:
        report["recommendations"].append("Review warnings; some may be expected for model-quality or optional-report states.")
    if model_quality_blocks:
        report["recommendations"].append("Model-quality blocks should keep promotion and betting locked, but should not be confused with pipeline crashes.")
    if not report["recommendations"]:
        report["recommendations"].append("No repository anomaly issues detected by this scanner.")

    OUTPUT_PATH.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report


def main() -> int:
    report = build_report()
    print(
        json.dumps(
            {
                "status": report["status"],
                "error_count": report["error_count"],
                "warning_count": report["warning_count"],
                "output_path": str(OUTPUT_PATH),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
