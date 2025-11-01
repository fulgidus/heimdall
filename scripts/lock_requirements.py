#!/usr/bin/env python3
"""
Lock Requirements Script - Generate .lock files from .txt requirement files

This script:
1. Finds all .txt files in services/requirements/
2. Generates corresponding .lock files using pip-compile
3. Runs safety checks for vulnerabilities
4. Generates audit reports (JSON and Markdown)

Usage:
    python scripts/lock_requirements.py [--verbose] [--allow-unsafe]
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class LockRequirementsManager:
    """Manages requirement file locking and security auditing."""

    def __init__(self, verbose: bool = False, allow_unsafe: bool = False):
        self.verbose = verbose
        self.allow_unsafe = allow_unsafe
        self.requirements_dir = Path("services/requirements")
        self.audit_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "files_processed": [],
            "vulnerabilities": [],
            "total_packages": 0,
            "errors": [],
        }

    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[INFO] {message}")

    def error(self, message: str):
        """Print error message."""
        print(f"[ERROR] {message}", file=sys.stderr)

    def check_dependencies(self) -> bool:
        """Check if required tools are installed."""
        try:
            # Check pip-tools
            result = subprocess.run(["pip-compile", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                self.error("pip-tools not found. Install with: pip install pip-tools")
                return False

            # Check safety (optional)
            result = subprocess.run(["safety", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                self.log("safety not found. Install with: pip install safety")
                self.log("Continuing without security scanning...")

            return True
        except FileNotFoundError as e:
            self.error(f"Required tool not found: {e}")
            return False

    def find_requirement_files(self) -> list[Path]:
        """Find all .txt files in requirements directory."""
        if not self.requirements_dir.exists():
            self.error(f"Requirements directory not found: {self.requirements_dir}")
            return []

        txt_files = list(self.requirements_dir.glob("*.txt"))
        self.log(f"Found {len(txt_files)} requirement files")
        return txt_files

    def compile_requirements(self, req_file: Path) -> tuple[bool, Path]:
        """Compile a requirements file to .lock file."""
        lock_file = req_file.with_suffix(".lock")

        self.log(f"Compiling {req_file.name} -> {lock_file.name}")

        cmd = [
            "pip-compile",
            str(req_file),
            "--output-file",
            str(lock_file),
            "--resolver=backtracking",
            "--upgrade",
        ]

        if self.allow_unsafe:
            cmd.append("--allow-unsafe")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

            if result.returncode != 0:
                self.error(f"Failed to compile {req_file.name}")
                self.error(result.stderr)
                self.audit_results["errors"].append({"file": str(req_file), "error": result.stderr})
                return False, lock_file

            self.log(f"Successfully compiled {lock_file.name}")
            return True, lock_file

        except Exception as e:
            self.error(f"Exception during compilation: {e}")
            self.audit_results["errors"].append({"file": str(req_file), "error": str(e)})
            return False, lock_file

    def run_safety_check(self, lock_file: Path) -> dict:
        """Run safety check on a lock file."""
        audit_json = lock_file.with_suffix(".audit.json")

        self.log(f"Running safety check on {lock_file.name}")

        try:
            result = subprocess.run(
                ["safety", "check", "-r", str(lock_file), "--json"], capture_output=True, text=True
            )

            # Safety returns non-zero if vulnerabilities found
            if result.stdout:
                try:
                    audit_data = json.loads(result.stdout)
                    # Write audit to file
                    with open(audit_json, "w") as f:
                        json.dump(audit_data, f, indent=2)
                    self.log(f"Audit saved to {audit_json.name}")
                    return audit_data
                except json.JSONDecodeError:
                    self.log(f"Could not parse safety output for {lock_file.name}")
                    return {}
            else:
                self.log(f"No vulnerabilities found in {lock_file.name}")
                return {}

        except FileNotFoundError:
            self.log("Safety not installed, skipping security scan")
            return {}
        except Exception as e:
            self.error(f"Error running safety check: {e}")
            return {}

    def count_packages(self, lock_file: Path) -> int:
        """Count packages in a lock file."""
        if not lock_file.exists():
            return 0

        try:
            with open(lock_file) as f:
                # Count non-comment, non-empty lines
                count = sum(
                    1
                    for line in f
                    if line.strip()
                    and not line.strip().startswith("#")
                    and not line.strip().startswith("-")
                )
            return count
        except Exception:
            return 0

    def generate_report(self) -> bool:
        """Generate audit reports in JSON and Markdown formats."""
        # Save JSON report
        json_report = Path("services/requirements/audit_report.json")
        try:
            with open(json_report, "w") as f:
                json.dump(self.audit_results, f, indent=2)
            self.log(f"JSON report saved to {json_report}")
        except Exception as e:
            self.error(f"Failed to save JSON report: {e}")

        # Generate Markdown report
        md_report = Path("services/requirements/audit_report.md")
        try:
            with open(md_report, "w") as f:
                f.write("# Dependency Audit Report\n\n")
                f.write(f"**Generated**: {self.audit_results['timestamp']}\n\n")

                f.write("## Summary\n\n")
                f.write(f"- **Files Processed**: {len(self.audit_results['files_processed'])}\n")
                f.write(f"- **Total Packages Locked**: {self.audit_results['total_packages']}\n")
                f.write(
                    f"- **Vulnerabilities Found**: {len(self.audit_results['vulnerabilities'])}\n"
                )
                f.write(f"- **Errors**: {len(self.audit_results['errors'])}\n\n")

                if self.audit_results["files_processed"]:
                    f.write("## Files Processed\n\n")
                    for file_info in self.audit_results["files_processed"]:
                        f.write(
                            f"- `{file_info['file']}` → `{file_info['lock_file']}` "
                            f"({file_info['packages']} packages)\n"
                        )
                    f.write("\n")

                if self.audit_results["vulnerabilities"]:
                    f.write("## Vulnerabilities\n\n")
                    for vuln in self.audit_results["vulnerabilities"]:
                        f.write(
                            f"### {vuln.get('package', 'Unknown')} - {vuln.get('severity', 'UNKNOWN')}\n"
                        )
                        f.write(f"- **CVE**: {vuln.get('cve', 'N/A')}\n")
                        f.write(f"- **Affected Version**: {vuln.get('version', 'N/A')}\n")
                        f.write(f"- **Description**: {vuln.get('description', 'N/A')}\n\n")

                if self.audit_results["errors"]:
                    f.write("## Errors\n\n")
                    for error in self.audit_results["errors"]:
                        f.write(f"- `{error['file']}`: {error['error']}\n")
                    f.write("\n")

            self.log(f"Markdown report saved to {md_report}")
            return True

        except Exception as e:
            self.error(f"Failed to save Markdown report: {e}")
            return False

    def run(self) -> int:
        """Main execution method."""
        print("=" * 60)
        print("Heimdall - Lock Requirements Generator")
        print("=" * 60)

        # Check dependencies
        if not self.check_dependencies():
            return 1

        # Find requirement files
        req_files = self.find_requirement_files()
        if not req_files:
            self.error("No requirement files found")
            return 1

        # Process each file
        success_count = 0
        for req_file in req_files:
            success, lock_file = self.compile_requirements(req_file)

            if success:
                success_count += 1
                package_count = self.count_packages(lock_file)

                # Run safety check
                safety_results = self.run_safety_check(lock_file)

                # Record results
                file_info = {
                    "file": str(req_file.name),
                    "lock_file": str(lock_file.name),
                    "packages": package_count,
                    "vulnerabilities": (
                        len(safety_results) if isinstance(safety_results, list) else 0
                    ),
                }
                self.audit_results["files_processed"].append(file_info)
                self.audit_results["total_packages"] += package_count

                # Collect vulnerabilities
                if isinstance(safety_results, list):
                    self.audit_results["vulnerabilities"].extend(safety_results)

        # Generate reports
        self.generate_report()

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Files processed: {len(self.audit_results['files_processed'])}/{len(req_files)}")
        print(f"Total packages locked: {self.audit_results['total_packages']}")
        print(f"Vulnerabilities found: {len(self.audit_results['vulnerabilities'])}")
        print(f"Errors: {len(self.audit_results['errors'])}")
        print("=" * 60)

        # Return exit code
        if self.audit_results["errors"]:
            return 1
        if self.audit_results["vulnerabilities"]:
            print("\n⚠️  WARNING: Vulnerabilities detected!")
            return 1

        print("\n✅ All checks passed!")
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate .lock files from requirements and run security audits"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--allow-unsafe", action="store_true", help="Allow unsafe packages (e.g., setuptools, pip)"
    )

    args = parser.parse_args()

    manager = LockRequirementsManager(verbose=args.verbose, allow_unsafe=args.allow_unsafe)

    return manager.run()


if __name__ == "__main__":
    sys.exit(main())
