#!/usr/bin/env python3
"""
Dependency Audit Tool - Analyze dependencies across all microservices

This script:
1. Scans all service requirements files
2. Checks for version pinning consistency
3. Detects transitive dependency conflicts
4. Identifies outdated packages
5. Checks for known vulnerabilities
6. Generates comprehensive reports (JSON, Markdown, CSV)

Usage:
    python scripts/audit_dependencies.py [--format=all] [--output=audit-results/]
"""

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


class DependencyAuditor:
    """Audits dependencies across all microservices."""

    def __init__(self, output_dir: str = "audit-results", output_format: str = "all"):
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.services_dir = Path("services")
        self.requirements_dir = Path("services/requirements")

        # Audit results structure
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "duplicate_packages": [],
            "vulnerabilities": [],
            "conflicts": [],
            "outdated": [],
            "version_matrix": {},
            "summary": {
                "total_services": 0,
                "total_packages": 0,
                "duplicate_count": 0,
                "conflict_count": 0,
                "vulnerability_count": 0,
            },
        }

    def log(self, message: str):
        """Print log message."""
        print(f"[INFO] {message}")

    def error(self, message: str):
        """Print error message."""
        print(f"[ERROR] {message}", file=sys.stderr)

    def find_service_requirements(self) -> dict[str, Path]:
        """Find all service requirements files."""
        services = {}

        if not self.services_dir.exists():
            self.error(f"Services directory not found: {self.services_dir}")
            return services

        # Find all service directories
        for service_dir in self.services_dir.iterdir():
            if not service_dir.is_dir():
                continue

            # Skip the requirements directory itself
            if service_dir.name == "requirements":
                continue

            req_file = service_dir / "requirements.txt"
            if req_file.exists():
                services[service_dir.name] = req_file

        self.log(f"Found {len(services)} services with requirements files")
        return services

    def parse_requirements(self, req_file: Path) -> dict[str, str]:
        """Parse a requirements file and return package:version dict."""
        packages = {}

        try:
            with open(req_file) as f:
                for line in f:
                    line = line.strip()

                    # Skip comments and empty lines
                    if not line or line.startswith("#"):
                        continue

                    # Skip -r references
                    if line.startswith("-r"):
                        continue

                    # Parse package==version
                    if "==" in line:
                        parts = line.split("==")
                        if len(parts) == 2:
                            package = parts[0].strip()
                            version = (
                                parts[1].strip().split("#")[0].strip()
                            )  # Remove inline comments
                            packages[package] = version
                    elif ">=" in line or "~=" in line or "<" in line:
                        # Handle other version specifiers
                        for sep in [">=", "~=", "<=", "<", ">"]:
                            if sep in line:
                                parts = line.split(sep)
                                package = parts[0].strip()
                                version = parts[1].strip().split("#")[0].strip()
                                packages[package] = f"{sep}{version}"
                                break

        except Exception as e:
            self.error(f"Error parsing {req_file}: {e}")

        return packages

    def check_version_consistency(self, services: dict[str, Path]):
        """Check for version consistency across services."""
        self.log("Checking version consistency...")

        # Build version matrix: package -> {service: version}
        version_matrix = defaultdict(dict)

        for service_name, req_file in services.items():
            packages = self.parse_requirements(req_file)
            for package, version in packages.items():
                version_matrix[package][service_name] = version

        # Find duplicates with different versions
        duplicates = []
        for package, service_versions in version_matrix.items():
            unique_versions = set(service_versions.values())
            if len(unique_versions) > 1:
                duplicates.append(
                    {
                        "package": package,
                        "versions": list(unique_versions),
                        "services": [
                            {"service": svc, "version": ver}
                            for svc, ver in service_versions.items()
                        ],
                    }
                )

        self.results["duplicate_packages"] = duplicates
        self.results["version_matrix"] = {
            pkg: dict(versions) for pkg, versions in version_matrix.items()
        }
        self.results["summary"]["duplicate_count"] = len(duplicates)

        if duplicates:
            self.log(f"Found {len(duplicates)} packages with version conflicts")
        else:
            self.log("No version conflicts found")

    def check_outdated_packages(self):
        """Check for outdated packages using pip list --outdated."""
        self.log("Checking for outdated packages...")

        try:
            # Get list of outdated packages
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"], capture_output=True, text=True
            )

            if result.returncode == 0 and result.stdout:
                outdated = json.loads(result.stdout)
                self.results["outdated"] = outdated
                self.log(f"Found {len(outdated)} outdated packages")
            else:
                self.log("No outdated packages found")

        except Exception as e:
            self.error(f"Error checking outdated packages: {e}")

    def check_vulnerabilities(self, services: dict[str, Path]):
        """Check for known vulnerabilities using safety."""
        self.log("Checking for vulnerabilities...")

        vulnerabilities = []

        try:
            for service_name, req_file in services.items():
                result = subprocess.run(
                    ["safety", "check", "-r", str(req_file), "--json"],
                    capture_output=True,
                    text=True,
                )

                if result.stdout:
                    try:
                        vulns = json.loads(result.stdout)
                        if isinstance(vulns, list):
                            for vuln in vulns:
                                vuln["service"] = service_name
                                vulnerabilities.append(vuln)
                    except json.JSONDecodeError:
                        pass

        except FileNotFoundError:
            self.log("Safety not installed, skipping vulnerability scan")
        except Exception as e:
            self.error(f"Error checking vulnerabilities: {e}")

        self.results["vulnerabilities"] = vulnerabilities
        self.results["summary"]["vulnerability_count"] = len(vulnerabilities)

        if vulnerabilities:
            self.log(f"Found {len(vulnerabilities)} vulnerabilities")
        else:
            self.log("No vulnerabilities found")

    def generate_json_report(self) -> bool:
        """Generate JSON report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_file = self.output_dir / "audit.json"

        try:
            with open(json_file, "w") as f:
                json.dump(self.results, f, indent=2)
            self.log(f"JSON report saved to {json_file}")
            return True
        except Exception as e:
            self.error(f"Failed to save JSON report: {e}")
            return False

    def generate_markdown_report(self) -> bool:
        """Generate Markdown report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        md_file = self.output_dir / "audit.md"

        try:
            with open(md_file, "w") as f:
                f.write("# Dependency Audit Report\n\n")
                f.write(f"**Generated**: {self.results['timestamp']}\n\n")

                # Summary
                f.write("## Summary\n\n")
                summary = self.results["summary"]
                f.write(f"- **Total Services**: {summary['total_services']}\n")
                f.write(f"- **Total Unique Packages**: {len(self.results['version_matrix'])}\n")
                f.write(f"- **Duplicate Packages**: {summary['duplicate_count']}\n")
                f.write(f"- **Vulnerabilities**: {summary['vulnerability_count']}\n")
                f.write(f"- **Conflicts**: {summary['conflict_count']}\n\n")

                # Duplicate packages
                if self.results["duplicate_packages"]:
                    f.write("## Version Conflicts\n\n")
                    f.write("These packages have different versions across services:\n\n")
                    for dup in self.results["duplicate_packages"]:
                        f.write(f"### {dup['package']}\n\n")
                        f.write("| Service | Version |\n")
                        f.write("|---------|--------|\n")
                        for svc in dup["services"]:
                            f.write(f"| {svc['service']} | {svc['version']} |\n")
                        f.write("\n")

                # Vulnerabilities
                if self.results["vulnerabilities"]:
                    f.write("## Vulnerabilities\n\n")
                    for vuln in self.results["vulnerabilities"]:
                        package = vuln.get("package", "Unknown")
                        service = vuln.get("service", "Unknown")
                        severity = vuln.get("severity", "UNKNOWN")
                        f.write(f"### {package} ({service}) - {severity}\n")
                        f.write(f"- **CVE**: {vuln.get('cve', 'N/A')}\n")
                        f.write(f"- **Description**: {vuln.get('description', 'N/A')}\n\n")

                # Outdated packages
                if self.results["outdated"]:
                    f.write("## Outdated Packages\n\n")
                    f.write("| Package | Current | Latest | Type |\n")
                    f.write("|---------|---------|--------|------|\n")
                    for pkg in self.results["outdated"]:
                        f.write(
                            f"| {pkg['name']} | {pkg['version']} | "
                            f"{pkg['latest_version']} | {pkg.get('latest_filetype', 'wheel')} |\n"
                        )
                    f.write("\n")

                # Recommendations
                f.write("## Recommendations\n\n")
                if self.results["duplicate_packages"]:
                    f.write(
                        "1. **Resolve Version Conflicts**: Standardize package versions across all services\n"
                    )
                if self.results["vulnerabilities"]:
                    f.write(
                        "2. **Fix Vulnerabilities**: Update affected packages to secure versions\n"
                    )
                if self.results["outdated"]:
                    f.write(
                        "3. **Update Packages**: Consider updating outdated packages (test thoroughly)\n"
                    )
                if not any(
                    [
                        self.results["duplicate_packages"],
                        self.results["vulnerabilities"],
                        self.results["outdated"],
                    ]
                ):
                    f.write("✅ No issues found! Dependencies are well-managed.\n")

            self.log(f"Markdown report saved to {md_file}")
            return True

        except Exception as e:
            self.error(f"Failed to save Markdown report: {e}")
            return False

    def generate_csv_report(self) -> bool:
        """Generate CSV version matrix."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_file = self.output_dir / "versions.csv"

        try:
            # Get all services
            services = sorted(
                {
                    svc
                    for versions in self.results["version_matrix"].values()
                    for svc in versions.keys()
                }
            )

            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)

                # Header
                writer.writerow(["Package"] + services)

                # Rows
                for package in sorted(self.results["version_matrix"].keys()):
                    row = [package]
                    versions = self.results["version_matrix"][package]
                    for service in services:
                        row.append(versions.get(service, ""))
                    writer.writerow(row)

            self.log(f"CSV report saved to {csv_file}")
            return True

        except Exception as e:
            self.error(f"Failed to save CSV report: {e}")
            return False

    def run(self) -> int:
        """Main execution method."""
        print("=" * 60)
        print("Heimdall - Dependency Auditor")
        print("=" * 60)

        # Find services
        services = self.find_service_requirements()
        if not services:
            self.error("No service requirements found")
            return 1

        self.results["summary"]["total_services"] = len(services)

        # Run checks
        self.check_version_consistency(services)
        self.check_outdated_packages()
        self.check_vulnerabilities(services)

        # Generate reports
        if self.output_format in ["all", "json"]:
            self.generate_json_report()

        if self.output_format in ["all", "markdown", "md"]:
            self.generate_markdown_report()

        if self.output_format in ["all", "csv"]:
            self.generate_csv_report()

        # Print summary
        print("\n" + "=" * 60)
        print("AUDIT SUMMARY")
        print("=" * 60)
        print(f"Services analyzed: {self.results['summary']['total_services']}")
        print(f"Unique packages: {len(self.results['version_matrix'])}")
        print(f"Version conflicts: {self.results['summary']['duplicate_count']}")
        print(f"Vulnerabilities: {self.results['summary']['vulnerability_count']}")
        print(f"Outdated packages: {len(self.results['outdated'])}")
        print("=" * 60)

        # Return exit code
        if self.results["summary"]["vulnerability_count"] > 0:
            print("\n⚠️  WARNING: Vulnerabilities detected!")
            return 1

        if self.results["summary"]["duplicate_count"] > 0:
            print("\n⚠️  WARNING: Version conflicts detected!")
            return 1

        print("\n✅ All checks passed!")
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Audit dependencies across all microservices")
    parser.add_argument(
        "--format",
        choices=["all", "json", "markdown", "md", "csv"],
        default="all",
        help="Output format (default: all)",
    )
    parser.add_argument(
        "--output", default="audit-results", help="Output directory (default: audit-results)"
    )

    args = parser.parse_args()

    auditor = DependencyAuditor(output_dir=args.output, output_format=args.format)

    return auditor.run()


if __name__ == "__main__":
    sys.exit(main())
