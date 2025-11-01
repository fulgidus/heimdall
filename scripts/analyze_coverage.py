#!/usr/bin/env python3
"""
Coverage Analysis Script

Runs pytest with coverage for each service and generates a comprehensive
coverage report.
"""

import json
import subprocess
import sys
from pathlib import Path


def run_coverage_for_service(service_path: Path) -> dict:
    """Run pytest with coverage for a specific service."""

    print(f"\n{'='*60}")
    print(f"Running coverage for {service_path.name}...")
    print(f"{'='*60}")

    # Check if service has tests
    tests_dir = service_path / "tests"
    if not tests_dir.exists():
        print(f"⚠️  No tests directory found for {service_path.name}")
        return {
            "service": service_path.name,
            "coverage": 0.0,
            "return_code": 0,
            "status": "no_tests",
        }

    src_dir = service_path / "src"
    if not src_dir.exists():
        print(f"⚠️  No src directory found for {service_path.name}")
        return {"service": service_path.name, "coverage": 0.0, "return_code": 0, "status": "no_src"}

    # Run pytest with coverage
    coverage_json_path = service_path / "coverage.json"
    html_cov_path = service_path / "htmlcov"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tests_dir),
            f"--cov={src_dir}",
            "--cov-report=json:" + str(coverage_json_path),
            "--cov-report=term",
            f"--cov-report=html:{html_cov_path}",
            "-v",
            "--tb=short",
        ],
        capture_output=True,
        text=True,
        cwd=str(service_path),
    )

    # Parse coverage JSON
    coverage_percent = 0.0
    status = "success" if result.returncode == 0 else "failed"

    if coverage_json_path.exists():
        try:
            with open(coverage_json_path) as f:
                coverage_data = json.load(f)
                coverage_percent = coverage_data.get("totals", {}).get("percent_covered", 0.0)
        except json.JSONDecodeError:
            print(f"⚠️  Failed to parse coverage JSON for {service_path.name}")
            status = "parse_error"

    return {
        "service": service_path.name,
        "coverage": coverage_percent,
        "return_code": result.returncode,
        "status": status,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def generate_summary_report(results: list[dict]) -> None:
    """Generate a summary report of coverage across all services."""

    print("\n" + "=" * 80)
    print("COVERAGE SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Service':<30} {'Coverage':<15} {'Status':<15}")
    print("-" * 80)

    total_coverage = 0
    services_with_tests = 0

    for result in results:
        status_emoji = {
            "success": "✅" if result["coverage"] >= 80 else "⚠️ ",
            "failed": "❌",
            "no_tests": "➖",
            "no_src": "➖",
            "parse_error": "⚠️ ",
        }.get(result["status"], "❓")

        coverage_str = f"{result['coverage']:.1f}%" if result["coverage"] > 0 else "N/A"
        status_str = result["status"].replace("_", " ").title()

        print(f"{status_emoji} {result['service']:<28} {coverage_str:<15} {status_str:<15}")

        if result["status"] == "success" and result["coverage"] > 0:
            total_coverage += result["coverage"]
            services_with_tests += 1

    print("-" * 80)

    if services_with_tests > 0:
        avg_coverage = total_coverage / services_with_tests
        avg_status = "✅" if avg_coverage >= 80 else "⚠️ "
        print(f"{avg_status} Average Coverage: {avg_coverage:.1f}%")
    else:
        print("⚠️  No services with test coverage data")

    print("=" * 80)


def main():
    """Main entry point."""
    services_dir = Path("services")

    if not services_dir.exists():
        print("❌ Error: services/ directory not found")
        print("   Please run this script from the project root")
        return 1

    # Find all services with src directories
    services = [
        d
        for d in services_dir.iterdir()
        if d.is_dir() and (d / "src").exists() and not d.name.startswith(".")
    ]

    if not services:
        print("❌ Error: No services found with src/ directories")
        return 1

    print(f"Found {len(services)} services to analyze:")
    for service in services:
        print(f"  • {service.name}")

    results: list[dict] = []

    for service in services:
        result = run_coverage_for_service(service)
        results.append(result)

    # Generate summary report
    generate_summary_report(results)

    # Save JSON report
    report_file = Path("coverage-report.json")

    services_with_tests = [r for r in results if r["status"] == "success" and r["coverage"] > 0]
    avg_coverage = (
        sum(r["coverage"] for r in services_with_tests) / len(services_with_tests)
        if services_with_tests
        else 0.0
    )

    report_data = {
        "average_coverage": avg_coverage,
        "total_services": len(results),
        "services_with_tests": len(services_with_tests),
        "services": [
            {"name": r["service"], "coverage": r["coverage"], "status": r["status"]}
            for r in results
        ],
    }

    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n📊 Detailed report saved to: {report_file}")

    # Return exit code based on coverage threshold
    if avg_coverage >= 80:
        print("\n✅ Coverage threshold met (≥80%)")
        return 0
    else:
        print(f"\n⚠️  Coverage below threshold: {avg_coverage:.1f}% < 80%")
        return 1


if __name__ == "__main__":
    sys.exit(main())
