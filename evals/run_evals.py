#!/usr/bin/env python3
import asyncio
import sys
import os

# Ensure backend root is on sys.path
backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from evals.evaluator import OfflineBenchmarkEvaluator

async def main():
    print("=" * 80)
    print("  NEERAI OFFLINE EVALUATION BENCHMARK (MILESTONE 11)")
    print("  Mode: 100% Offline / Zero Paid Model Calls / Deterministic Regression")
    print("=" * 80)

    evaluator = OfflineBenchmarkEvaluator()
    summary = await evaluator.run_full_benchmark()

    print(f"\nConfiguration Version: {summary.get('configuration_version', '1.0.0')}")
    print(f"Total Cases Evaluated : {summary['total_cases']}")
    print(f"Passed                : {summary['passed_cases']}")
    print(f"Failed                : {summary['failed_cases']}")
    print(f"Overall Benchmark Pass: {summary['overall_score_percent']}%\n")

    print("METRICS BREAKDOWN:")
    print("-" * 60)
    for m_name, m_data in summary["metrics"].items():
        bar = "✓" if m_data["percentage"] == 100.0 else "✗"
        print(f"  {bar} {m_name:<32}: {m_data['passed']}/{m_data['total']} ({m_data['percentage']}%)")

    print("\nINDIVIDUAL CASE RESULTS:")
    print("-" * 80)
    for res in summary["case_results"]:
        status_tag = "[PASS]" if res["passed"] else "[FAIL]"
        print(f"  {status_tag:<7} {res['case_id']:<28} | {res['name']}")
        if not res["passed"]:
            for reason in res["failure_reasons"]:
                print(f"           -> Failure: {reason}")

    print("=" * 80)
    if summary["failed_cases"] > 0:
        print("  BENCHMARK RESULT: REGRESSIONS DETECTED")
        sys.exit(1)
    else:
        print(f"  BENCHMARK RESULT: ALL {summary['total_cases']} OFFLINE CASES PASSED (100% COVERAGE)")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
