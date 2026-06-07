from __future__ import annotations

import json
from pathlib import Path

from scripts.saas_readiness_report import build_report


REPORT_PATH = Path("report/saas_readiness_report.json")


def test_saas_readiness_report_contract() -> None:
    report = build_report()

    assert isinstance(report, dict)

    required_keys = {
        "generated_at",
        "status",
        "current_product_stage",
        "target_product_stage",
        "implemented_modules",
        "planned_modules",
        "disabled_modules",
        "live_betting_allowed",
        "automated_wagering_allowed",
        "user_funds_handled",
        "api_commercialization_ready",
        "billing_ready",
        "multi_tenant_ready",
        "b2b_api_ready",
        "documentation_score",
        "governance_score",
        "evidence_score",
        "security_readiness_score",
        "product_readiness_score",
        "blockers",
        "warnings",
        "recommendations",
        "input_files",
    }

    assert required_keys.issubset(set(report.keys()))

    assert report["status"] in {"ok", "partial", "failed"}

    assert report["live_betting_allowed"] is False
    assert report["automated_wagering_allowed"] is False
    assert report["user_funds_handled"] is False

    assert report["api_commercialization_ready"] is False
    assert report["billing_ready"] is False
    assert report["multi_tenant_ready"] is False
    assert report["b2b_api_ready"] is False

    disabled_modules = report.get("disabled_modules")
    assert isinstance(disabled_modules, list)
    assert "real_money_betting" in disabled_modules
    assert "automated_wagering" in disabled_modules
    assert "sportsbook_execution" in disabled_modules
    assert "user_fund_custody" in disabled_modules

    planned_modules = report.get("planned_modules")
    assert isinstance(planned_modules, list)
    assert "multi_tenant_accounts" in planned_modules
    assert "api_key_auth" in planned_modules
    assert "billing" in planned_modules

    for key in (
        "documentation_score",
        "governance_score",
        "evidence_score",
        "security_readiness_score",
        "product_readiness_score",
    ):
        assert isinstance(report[key], (int, float))
        assert 0.0 <= float(report[key]) <= 100.0

    assert REPORT_PATH.exists()

    with REPORT_PATH.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)

    assert saved["live_betting_allowed"] is False
    assert saved["automated_wagering_allowed"] is False
    assert saved["user_funds_handled"] is False
