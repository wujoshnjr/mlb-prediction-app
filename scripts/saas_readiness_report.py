from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPORT_DIR = Path("report")
DATA_DIR = Path("data")
DOCS_DIR = Path("docs")

OUTPUT_PATH = REPORT_DIR / "saas_readiness_report.json"

README_PATH = Path("README.md")

REQUIRED_DOCS = {
    "saas_roadmap": DOCS_DIR / "SAAS_ROADMAP.md",
    "api_design": DOCS_DIR / "API_DESIGN.md",
    "risk_policy": DOCS_DIR / "RISK_POLICY.md",
    "data_sources": DOCS_DIR / "DATA_SOURCES.md",
    "evaluation_method": DOCS_DIR / "EVALUATION_METHOD.md",
    "deployment_render": DOCS_DIR / "DEPLOYMENT_RENDER.md",
    "github_actions_workflow": DOCS_DIR / "GITHUB_ACTIONS_WORKFLOW.md",
    "no_automated_wagering_policy": DOCS_DIR / "NO_AUTOMATED_WAGERING_POLICY.md",
    "b2b_product_spec": DOCS_DIR / "B2B_PRODUCT_SPEC.md",
}

INPUT_JSON_FILES = {
    "world_class_trading_system": REPORT_DIR / "world_class_trading_system_report.json",
    "promotion_gate": REPORT_DIR / "promotion_gate_report.json",
    "data_contract": REPORT_DIR / "data_contract_report.json",
    "pipeline_manifest": REPORT_DIR / "pipeline_manifest.json",
    "sample_state": DATA_DIR / "sample_state.json",
    "training_status": DATA_DIR / "training_status.json",
}

DISABLED_MODULES = [
    "real_money_betting",
    "automated_wagering",
    "sportsbook_execution",
    "user_fund_custody",
    "commercial_api_distribution_until_license_review",
]

PLANNED_MODULES = [
    "multi_tenant_accounts",
    "api_key_auth",
    "rate_limit",
    "usage_tracking",
    "billing",
    "developer_portal",
    "webhook",
    "white_label_widgets",
    "postgres_data_warehouse",
    "enterprise_audit_logs",
    "data_license_registry",
]

IMPLEMENTED_MODULES = [
    "paper_trading_research_dashboard",
    "prediction_snapshot_pipeline",
    "finalized_outcome_linking",
    "market_comparison_reports",
    "clv_tracking",
    "promotion_gate",
    "data_contract_validation",
    "pipeline_manifest",
    "world_class_control_tower",
    "sample_state_tracking",
    "saas_readiness_tracking",
]

UNSAFE_TRUE_FIELDS = {
    "live_betting_allowed",
    "automated_wagering_allowed",
    "user_funds_handled",
    "sportsbook_execution_enabled",
    "real_money_betting_enabled",
    "production_live_enabled",
    "api_commercialization_ready",
    "billing_ready",
    "multi_tenant_ready",
    "b2b_api_ready",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}

    if isinstance(value, list):
        return [_json_safe(child) for child in value]

    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _json_safe(payload)
    path.write_text(
        json.dumps(
            safe_payload,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    status = {
        "path": str(path),
        "exists": path.exists(),
        "error": "",
        "type": None,
    }

    if not path.exists():
        status["error"] = "file_missing"
        return None, status

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status["error"] = str(exc)
        return None, status

    status["type"] = type(payload).__name__

    if not isinstance(payload, dict):
        status["error"] = "json_not_object"
        return None, status

    return payload, status


def _read_text(path: Path) -> Tuple[str, Dict[str, Any]]:
    status = {
        "path": str(path),
        "exists": path.exists(),
        "error": "",
        "characters": 0,
    }

    if not path.exists():
        status["error"] = "file_missing"
        return "", status

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        status["error"] = str(exc)
        return "", status

    status["characters"] = len(text)
    return text, status


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None

        parsed = float(value)
        if not math.isfinite(parsed):
            return None

        return parsed
    except Exception:
        return None


def _to_int(value: Any, default: int = 0) -> int:
    parsed = _to_float(value)
    return int(parsed) if parsed is not None else default


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "enabled"}

    return bool(value)


def _find_unsafe_true_flags(
    payload: Any,
    *,
    prefix: str = "",
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    if isinstance(payload, dict):
        for key, value in payload.items():
            current_path = f"{prefix}.{key}" if prefix else str(key)

            if key in UNSAFE_TRUE_FIELDS and _is_true(value):
                findings.append(
                    {
                        "path": current_path,
                        "field": key,
                        "value": value,
                    }
                )

            findings.extend(_find_unsafe_true_flags(value, prefix=current_path))

    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(
                _find_unsafe_true_flags(value, prefix=f"{prefix}[{index}]")
            )

    return findings


def _score_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(max(0.0, min(100.0, numerator / denominator * 100.0)), 2)


def _doc_keyword_score(text: str, keywords: List[str]) -> float:
    if not text:
        return 0.0

    normalized = text.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in normalized)
    return _score_ratio(hits, len(keywords))


def _load_inputs() -> Tuple[Dict[str, Optional[Dict[str, Any]]], Dict[str, Dict[str, Any]], Dict[str, str]]:
    reports: Dict[str, Optional[Dict[str, Any]]] = {}
    input_files: Dict[str, Dict[str, Any]] = {}
    text_files: Dict[str, str] = {}

    readme_text, readme_status = _read_text(README_PATH)
    text_files["README.md"] = readme_text
    input_files["README.md"] = readme_status

    for name, path in REQUIRED_DOCS.items():
        text, status = _read_text(path)
        text_files[name] = text
        input_files[name] = status

    for name, path in INPUT_JSON_FILES.items():
        payload, status = _read_json(path)
        reports[name] = payload
        input_files[name] = status

    return reports, input_files, text_files


def _documentation_score(input_files: Dict[str, Dict[str, Any]], text_files: Dict[str, str]) -> float:
    required_items = ["README.md", *REQUIRED_DOCS.keys()]
    existing_count = sum(1 for key in required_items if input_files.get(key, {}).get("exists"))

    existence_score = _score_ratio(existing_count, len(required_items))

    keyword_requirements = [
        "paper-trading",
        "research",
        "not a real-money betting",
        "live betting",
        "automated wagering",
        "CLV",
        "OOS",
        "promotion gate",
        "data quality",
        "SaaS",
        "B2B",
    ]

    combined_text = "\n".join(text_files.values())
    keyword_score = _doc_keyword_score(combined_text, keyword_requirements)

    return round(existence_score * 0.65 + keyword_score * 0.35, 2)


def _governance_score(
    reports: Dict[str, Optional[Dict[str, Any]]],
    input_files: Dict[str, Dict[str, Any]],
    text_files: Dict[str, str],
) -> Tuple[float, List[Dict[str, Any]]]:
    unsafe_findings: List[Dict[str, Any]] = []

    for name in ("world_class_trading_system", "promotion_gate", "data_contract"):
        report = reports.get(name)
        if isinstance(report, dict):
            for finding in _find_unsafe_true_flags(report):
                finding["source"] = name
                unsafe_findings.append(finding)

    base_score = 100.0

    if unsafe_findings:
        base_score = 0.0

    required_governance_docs = [
        "risk_policy",
        "no_automated_wagering_policy",
        "evaluation_method",
    ]
    missing_governance_docs = [
        key
        for key in required_governance_docs
        if not input_files.get(key, {}).get("exists")
    ]

    base_score -= len(missing_governance_docs) * 12.0

    combined_text = "\n".join(
        text_files.get(key, "")
        for key in ["risk_policy", "no_automated_wagering_policy", "README.md"]
    ).lower()

    required_phrases = [
        "no automated wagering",
        "no user funds",
        "no guaranteed profit",
        "paper",
        "live betting",
    ]

    missing_phrases = [
        phrase
        for phrase in required_phrases
        if phrase not in combined_text
    ]

    base_score -= len(missing_phrases) * 6.0

    return round(max(0.0, min(100.0, base_score)), 2), unsafe_findings


def _evidence_score(reports: Dict[str, Optional[Dict[str, Any]]]) -> Tuple[float, List[str]]:
    warnings: List[str] = []
    sample_state = reports.get("sample_state") or {}
    training_status = reports.get("training_status") or {}

    clean_settled = _to_int(sample_state.get("clean_settled_snapshots"))
    walkforward = _to_int(sample_state.get("walkforward_predictions"))
    trained = bool(sample_state.get("trained", training_status.get("trained", False)))

    score = 0.0

    score += min(40.0, clean_settled / 500.0 * 40.0)
    score += min(30.0, walkforward / 300.0 * 30.0)

    if trained:
        score += 10.0
    else:
        warnings.append("Model is not trained yet according to sample_state/training_status.")

    for key, points in (
        ("promotion_gate", 5.0),
        ("data_contract", 5.0),
        ("pipeline_manifest", 5.0),
        ("world_class_trading_system", 5.0),
    ):
        if reports.get(key):
            score += points
        else:
            warnings.append(f"{key} report is missing or unreadable.")

    if clean_settled < 500:
        warnings.append(f"Clean settled samples below SaaS evidence threshold: {clean_settled} < 500.")

    if walkforward < 300:
        warnings.append(f"Rolling OOS predictions below SaaS evidence threshold: {walkforward} < 300.")

    return round(max(0.0, min(100.0, score)), 2), warnings


def _security_readiness_score(input_files: Dict[str, Dict[str, Any]], text_files: Dict[str, str]) -> Tuple[float, List[str]]:
    warnings: List[str] = []

    security_docs = [
        "api_design",
        "risk_policy",
        "no_automated_wagering_policy",
        "b2b_product_spec",
    ]

    doc_score = _score_ratio(
        sum(1 for key in security_docs if input_files.get(key, {}).get("exists")),
        len(security_docs),
    )

    combined_text = "\n".join(text_files.values()).lower()

    planned_security_topics = [
        "api key",
        "rate limit",
        "usage tracking",
        "audit log",
        "rbac",
        "multi-tenant",
        "billing",
        "developer portal",
    ]

    topic_score = _doc_keyword_score(combined_text, planned_security_topics)

    score = round(doc_score * 0.45 + topic_score * 0.25, 2)

    warnings.extend(
        [
            "API key authentication is planned but not implemented.",
            "Rate limiting is planned but not implemented.",
            "RBAC / multi-tenant organization isolation is planned but not implemented.",
            "Enterprise audit logs are planned but not implemented.",
            "Billing and usage metering are planned but not implemented.",
        ]
    )

    return round(max(0.0, min(100.0, score)), 2), warnings


def _product_readiness_score(
    reports: Dict[str, Optional[Dict[str, Any]]],
    input_files: Dict[str, Dict[str, Any]],
) -> Tuple[float, List[str]]:
    warnings: List[str] = []

    score = 0.0

    if input_files.get("README.md", {}).get("exists"):
        score += 15.0

    if all(input_files.get(key, {}).get("exists") for key in REQUIRED_DOCS):
        score += 25.0
    else:
        warnings.append("Some SaaS/B2B documentation files are missing.")

    for key, points in (
        ("world_class_trading_system", 15.0),
        ("promotion_gate", 10.0),
        ("data_contract", 10.0),
        ("pipeline_manifest", 10.0),
        ("sample_state", 10.0),
        ("training_status", 5.0),
    ):
        if reports.get(key):
            score += points
        else:
            warnings.append(f"{key} is not available for product readiness scoring.")

    warnings.extend(
        [
            "Commercial API is not ready.",
            "Billing is not ready.",
            "Multi-tenant account system is not ready.",
            "White-label widgets are not ready.",
        ]
    )

    return round(max(0.0, min(100.0, score)), 2), warnings


def build_report() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    reports, input_files, text_files = _load_inputs()

    blockers: List[str] = []
    warnings: List[str] = []
    recommendations: List[str] = []

    documentation_score = _documentation_score(input_files, text_files)
    governance_score, unsafe_findings = _governance_score(reports, input_files, text_files)
    evidence_score, evidence_warnings = _evidence_score(reports)
    security_score, security_warnings = _security_readiness_score(input_files, text_files)
    product_score, product_warnings = _product_readiness_score(reports, input_files)

    warnings.extend(evidence_warnings)
    warnings.extend(security_warnings)
    warnings.extend(product_warnings)

    if unsafe_findings:
        blockers.append("Unsafe governance flags detected in upstream reports.")

    missing_docs = [
        str(REQUIRED_DOCS[key])
        for key, status in input_files.items()
        if key in REQUIRED_DOCS and not status.get("exists")
    ]
    for path in missing_docs:
        warnings.append(f"Missing required SaaS documentation file: {path}")

    implemented_modules = IMPLEMENTED_MODULES.copy()
    planned_modules = PLANNED_MODULES.copy()
    disabled_modules = DISABLED_MODULES.copy()

    live_betting_allowed = False
    automated_wagering_allowed = False
    user_funds_handled = False
    api_commercialization_ready = False
    billing_ready = False
    multi_tenant_ready = False
    b2b_api_ready = False

    for finding in unsafe_findings:
        field = finding.get("field")
        if field == "live_betting_allowed":
            live_betting_allowed = True
        elif field == "automated_wagering_allowed":
            automated_wagering_allowed = True
        elif field == "user_funds_handled":
            user_funds_handled = True
        elif field == "api_commercialization_ready":
            api_commercialization_ready = True
        elif field == "billing_ready":
            billing_ready = True
        elif field == "multi_tenant_ready":
            multi_tenant_ready = True
        elif field == "b2b_api_ready":
            b2b_api_ready = True

    if live_betting_allowed or automated_wagering_allowed or user_funds_handled:
        status = "failed"
    elif blockers:
        status = "failed"
    elif (
        documentation_score >= 95
        and governance_score >= 95
        and evidence_score >= 80
        and security_score >= 70
        and product_score >= 80
    ):
        status = "ok"
    else:
        status = "partial"

    recommendations.extend(
        [
            "Keep the current product positioned as a paper-trading research dashboard until OOS and sample evidence are stronger.",
            "Do not commercialize API access until data-source licensing and redistribution rights are reviewed.",
            "Implement API key authentication, rate limiting, usage tracking, RBAC, and audit logs before any B2B API launch.",
            "Keep real-money betting, sportsbook execution, automated wagering, and user-fund custody permanently disabled.",
            "Use sample_state.json as the canonical sample-count source for training, promotion, dashboard, and SaaS readiness.",
        ]
    )

    report = {
        "generated_at": _utc_now(),
        "status": status,
        "current_product_stage": "research_platform",
        "target_product_stage": "sports_intelligence_saas_b2b_api",
        "implemented_modules": implemented_modules,
        "planned_modules": planned_modules,
        "disabled_modules": disabled_modules,
        "live_betting_allowed": live_betting_allowed,
        "automated_wagering_allowed": automated_wagering_allowed,
        "user_funds_handled": user_funds_handled,
        "api_commercialization_ready": api_commercialization_ready,
        "billing_ready": billing_ready,
        "multi_tenant_ready": multi_tenant_ready,
        "b2b_api_ready": b2b_api_ready,
        "documentation_score": documentation_score,
        "governance_score": governance_score,
        "evidence_score": evidence_score,
        "security_readiness_score": security_score,
        "product_readiness_score": product_score,
        "unsafe_governance_findings": unsafe_findings,
        "blockers": blockers,
        "warnings": sorted(set(warnings)),
        "recommendations": recommendations,
        "input_files": input_files,
    }

    _write_json(OUTPUT_PATH, report)
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
