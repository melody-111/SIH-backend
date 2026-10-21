import logging
from typing import Dict, Any, List, Optional
from app.core.agent_config import AgentConfigManager, AgentSystemConfig
from evals.evaluator import OfflineBenchmarkEvaluator

logger = logging.getLogger("neerai.evals.candidate")

class CandidateEvaluationManager:
    """
    Manages controlled evaluation of candidate configurations against baseline.
    Enforces strict safety criteria and records comparison metrics before acceptance.
    """
    def __init__(self):
        self.config_manager = AgentConfigManager()

    async def evaluate_candidate(
        self,
        candidate_version: str,
        defect_description: str,
        hypothesis: str
    ) -> Dict[str, Any]:
        """
        Runs full offline evaluation comparison between baseline and candidate version.
        Accepts or rejects candidate based on explicit safety and efficiency criteria.
        """
        evaluator = OfflineBenchmarkEvaluator()

        # 1. Run Baseline Benchmark (known-good v1.0.0)
        self.config_manager.activate_version("1.0.0")
        baseline_summary = await evaluator.run_full_benchmark()

        # 2. Run Candidate Benchmark
        if candidate_version not in self.config_manager._registry:
            return {
                "decision": "REJECTED",
                "reason": f"Candidate version {candidate_version} not found in registry."
            }

        self.config_manager.activate_version(candidate_version)
        candidate_summary = await evaluator.run_full_benchmark()

        # 3. Analyze Safety-Critical Invariants
        # Zero fabrication check: must be 100%
        cand_zero_fab = candidate_summary["metrics"]["zero_fabrication"]["percentage"]
        cand_warning_attr = candidate_summary["metrics"]["official_warning_attribution"]["percentage"]
        cand_schema = candidate_summary["metrics"]["schema_validity"]["percentage"]

        # Regression detection: cases passed in baseline that failed in candidate
        baseline_passed_ids = {r["case_id"] for r in baseline_summary["case_results"] if r["passed"]}
        candidate_failed_ids = {r["case_id"] for r in candidate_summary["case_results"] if not r["passed"]}
        regressions = baseline_passed_ids.intersection(candidate_failed_ids)

        # Efficiency calculation: count total agent dispatches across all cases
        baseline_agent_count = sum(len(r.get("planned_agents", [])) for r in baseline_summary["case_results"])
        candidate_agent_count = sum(len(r.get("planned_agents", [])) for r in candidate_summary["case_results"])
        agent_dispatch_delta = candidate_agent_count - baseline_agent_count

        # 4. Explicit Decision Rules
        decision = "ACCEPTED"
        rejection_reasons = []

        if len(regressions) > 0:
            decision = "REJECTED"
            rejection_reasons.append(f"Safety-Critical Regression: {len(regressions)} cases regressed: {list(regressions)}")

        if cand_zero_fab < 100.0:
            decision = "REJECTED"
            rejection_reasons.append(f"Zero-Fabrication Invariant Violated: {cand_zero_fab}%")

        if cand_warning_attr < 100.0:
            decision = "REJECTED"
            rejection_reasons.append(f"Official Warning Precedence Violated: {cand_warning_attr}%")

        if cand_schema < 100.0:
            decision = "REJECTED"
            rejection_reasons.append(f"Schema Validity Violated: {cand_schema}%")

        if candidate_summary["overall_score_percent"] < baseline_summary["overall_score_percent"]:
            decision = "REJECTED"
            rejection_reasons.append(f"Overall score decreased from {baseline_summary['overall_score_percent']}% to {candidate_summary['overall_score_percent']}%")

        # Revert back to baseline until human approval
        self.config_manager.activate_version("1.0.0")

        return {
            "candidate_version": candidate_version,
            "defect_description": defect_description,
            "hypothesis": hypothesis,
            "decision": decision,
            "rejection_reasons": rejection_reasons,
            "baseline_score_pct": baseline_summary["overall_score_percent"],
            "candidate_score_pct": candidate_summary["overall_score_percent"],
            "safety_critical_regressions": len(regressions),
            "regressed_case_ids": list(regressions),
            "baseline_total_agent_dispatches": baseline_agent_count,
            "candidate_total_agent_dispatches": candidate_agent_count,
            "agent_dispatch_delta": agent_dispatch_delta,
            "zero_fabrication_maintained": (cand_zero_fab == 100.0),
            "official_warning_precedence_maintained": (cand_warning_attr == 100.0),
            "requires_human_architect_signoff": True
        }
