import pytest
from evals.evaluator import OfflineBenchmarkEvaluator

@pytest.mark.anyio
async def test_offline_benchmark_suite_25_cases():
    """
    Executes the NeerAI Milestone 11 Offline Evaluation Benchmark Suite.
    Runs 25 distinct deterministic cases covering weather, sea swell, PFZ unauthenticated
    handling, official warning precedence, prompt injection resistance, and multilingual support.
    """
    evaluator = OfflineBenchmarkEvaluator()
    summary = await evaluator.run_full_benchmark()

    # Core Benchmark Assertions
    assert summary["total_cases"] >= 25, f"Expected at least 25 cases, got {summary['total_cases']}"
    assert summary["failed_cases"] == 0, f"Failures detected: {[r['case_id'] for r in summary['case_results'] if not r['passed']]}"
    assert summary["overall_score_percent"] == 100.0

    # Metric-Specific Assertions
    metrics = summary["metrics"]
    assert metrics["zero_fabrication"]["percentage"] == 100.0
    assert metrics["schema_validity"]["percentage"] == 100.0
    assert metrics["intent_selection_correct"]["percentage"] == 100.0
    assert metrics["official_warning_attribution"]["percentage"] == 100.0
    assert metrics["stale_data_disclosure"]["percentage"] == 100.0
    assert metrics["missing_data_handled"]["percentage"] == 100.0
