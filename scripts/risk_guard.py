from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple


class LiveBetRiskGuard:
    def __init__(
        self,
        market_research_report_path: str = "report/market_edge_research.json",
    ) -> None:
        self.report_path = Path(market_research_report_path)
        self.blacklisted_buckets: set[str] = set()
        self.report: Dict[str, Any] = {}
        self._load_market_blacklists()

    def _load_market_blacklists(self) -> None:
        if not self.report_path.exists():
            self.blacklisted_buckets.add("edge_8%+")
            return

        try:
            self.report = json.loads(self.report_path.read_text(encoding="utf-8"))
        except Exception:
            self.blacklisted_buckets.add("edge_8%+")
            return

        edge_buckets = self.report.get("edge_buckets", {})
        if isinstance(edge_buckets, dict):
            for bucket_name, stats in edge_buckets.items():
                if not isinstance(stats, dict):
                    continue
                avg_clv = stats.get("avg_clv")
                beaten_market = bool(
                    stats.get("beaten_market_brier")
                    and stats.get("beaten_market_logloss")
                )
                if avg_clv is None or float(avg_clv) <= 0 or not beaten_market:
                    self.blacklisted_buckets.add(f"edge_{bucket_name}")

        health_buckets = self.report.get("data_health_buckets", {})
        if isinstance(health_buckets, dict):
            for bucket_name, stats in health_buckets.items():
                if not isinstance(stats, dict):
                    continue
                avg_clv = stats.get("avg_clv")
                if avg_clv is None or float(avg_clv) <= 0:
                    self.blacklisted_buckets.add(f"health_{bucket_name}")

    @staticmethod
    def edge_bucket(edge: float) -> str:
        edge_abs = abs(float(edge))
        if edge_abs >= 0.08:
            return "8%+"
        if edge_abs >= 0.05:
            return "5-8%"
        if edge_abs >= 0.03:
            return "3-5%"
        return "below_threshold"

    def validate_bet_candidate(
        self,
        *,
        context: Dict[str, Any],
        model_prob: float,
        market_no_vig_prob: float,
        features: Dict[str, Any],
    ) -> Tuple[bool, str]:
        if model_prob is None or market_no_vig_prob is None:
            return False, "REJECT: probability unavailable"

        if not math.isfinite(float(model_prob)) or not math.isfinite(
            float(market_no_vig_prob)
        ):
            return False, "REJECT: probability non-finite"

        edge = abs(float(model_prob) - float(market_no_vig_prob))
        bucket = self.edge_bucket(edge)

        if bucket == "below_threshold":
            return False, "REJECT: edge below threshold"

        if f"edge_{bucket}" in self.blacklisted_buckets:
            return False, f"REJECT: historical negative CLV bucket {bucket}"

        if context.get("lineup_status") != "confirmed":
            return False, "REJECT: lineup not confirmed"

        critical_features = [
            "statcast_woba_diff",
            "top3_woba_diff",
            "sp_fip_diff",
        ]

        for feature in critical_features:
            value = features.get(feature)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return False, f"REJECT: critical feature invalid {feature}"
            if numeric == 0.0:
                return False, f"REJECT: critical feature zero {feature}"

        feature_health_flags = context.get("feature_health_flags") or []
        if isinstance(feature_health_flags, list):
            if "savant_top3_unavailable" in feature_health_flags:
                return False, "REJECT: savant top3 unavailable"
            if "statcast_core_zero" in feature_health_flags:
                return False, "REJECT: statcast core zero"

        if "health_degraded" in self.blacklisted_buckets:
            return False, "REJECT: degraded data health bucket blacklisted"

        return True, "PASS: live bet candidate approved"
