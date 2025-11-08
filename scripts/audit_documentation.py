#!/usr/bin/env python3
"""
Documentation Audit Tool for Heimdall Project

This script provides comprehensive auditing of documentation files to ensure:
1. All files in docs/agents/ are linked from appropriate parent documents
2. No orphaned files exist
3. Clear actionable reports are generated

Implements requirements from AGENTS.md:
"All documentation files MUST be discoverable and contextual.
Orphaned files (not linked from anywhere) are NOT allowed."

Usage:
    python scripts/audit_documentation.py                    # Simple output
    python scripts/audit_documentation.py --format=json      # JSON report only
    python scripts/audit_documentation.py --format=markdown  # Markdown report only
    python scripts/audit_documentation.py --format=both      # Both reports
    python scripts/audit_documentation.py --verbose          # Detailed output

Exit Codes:
    0 - No orphaned files found
    1 - Orphaned files detected
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path


class DocumentationAuditor:
    """Main auditor class for documentation validation."""

    def __init__(self, base_path: Path, verbose: bool = False):
        self.base_path = base_path
        self.docs_path = base_path / "docs"
        self.agents_path = self.docs_path / "agents"
        self.verbose = verbose

        # Entry points for documentation
        self.entry_points = [base_path / "AGENTS.md", self.docs_path / "index.md"]

        # Link graph: file -> list of files it links to
        self.link_graph: dict[Path, list[Path]] = {}

        # All markdown files in docs/
        self.all_files: set[Path] = set()

        # Files reachable from entry points
        self.reachable_files: set[Path] = set()

        # Orphaned files
        self.orphaned_files: set[Path] = set()

        # Broken links
        self.broken_links: list[tuple[Path, str]] = []

    def log(self, message: str, level: str = "info"):
        """Log message if verbose mode is enabled."""
        if self.verbose or level == "error":
            prefix = {"info": "   ", "success": "✅ ", "warning": "⚠️  ", "error": "❌ "}.get(
                level, "   "
            )
            print(f"{prefix}{message}")

    def find_markdown_links(self, content: str, source_file: Path) -> list[Path]:
        """
        Extract all internal markdown links from content.

        Returns list of resolved absolute paths to linked files.
        Handles relative paths, absolute paths, and anchors.
        """
        links = []

        # Pattern for markdown links: [text](url)
        # Also handle HTML-style links: <a>text</a> pointing to same-named file
        md_pattern = r"\[([^\]]+)\]\(([^)]+)\)"

        for match in re.finditer(md_pattern, content):
            text, url = match.groups()

            # Skip external links
            if url.startswith(("http://", "https://", "mailto:", "ftp://")):
                continue

            # Skip anchor-only links
            if url.startswith("#"):
                continue

            # Remove anchor from URL
            url_without_anchor = url.split("#")[0]
            if not url_without_anchor:
                continue

            # Resolve relative path
            try:
                if url_without_anchor.startswith("/"):
                    # Absolute path from repo root
                    linked_file = (self.base_path / url_without_anchor.lstrip("/")).resolve()
                else:
                    # Relative path from source file
                    linked_file = (source_file.parent / url_without_anchor).resolve()

                # Only track links to files within docs/ directory
                if linked_file.exists() and str(linked_file).startswith(str(self.docs_path)):
                    links.append(linked_file)
                elif not linked_file.exists():
                    # Track broken link
                    self.broken_links.append((source_file, url))
                    self.log(
                        f"Broken link in {source_file.relative_to(self.base_path)}: {url}",
                        "warning",
                    )
            except Exception as e:
                self.log(
                    f"Error resolving link '{url}' in {source_file.relative_to(self.base_path)}: {e}",
                    "warning",
                )

        return links

    def build_link_graph(self):
        """Build graph of all documentation links."""
        self.log("Building link graph...")

        # Get all markdown files in docs/
        all_md_files = list(self.docs_path.glob("**/*.md"))
        self.all_files = set(all_md_files)
        self.log(f"Found {len(all_md_files)} markdown files in docs/")

        # Process all markdown files in docs/
        for file_path in all_md_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                links = self.find_markdown_links(content, file_path)
                self.link_graph[file_path] = links

                if self.verbose and links:
                    self.log(f"  {file_path.relative_to(self.base_path)} -> {len(links)} links")
            except Exception as e:
                self.log(f"Error reading {file_path.relative_to(self.base_path)}: {e}", "error")

        # Also process entry point files (AGENTS.md, etc.)
        for entry_point in self.entry_points:
            if entry_point.exists():
                try:
                    with open(entry_point, encoding="utf-8") as f:
                        content = f.read()

                    links = self.find_markdown_links(content, entry_point)
                    self.link_graph[entry_point] = links
                    self.log(
                        f"Entry point {entry_point.relative_to(self.base_path)} -> {len(links)} links"
                    )
                except Exception as e:
                    self.log(
                        f"Error reading {entry_point.relative_to(self.base_path)}: {e}", "error"
                    )

    def find_reachable_files(self, start_file: Path) -> set[Path]:
        """
        Find all files reachable from start_file via BFS traversal of link graph.
        """
        reachable = {start_file}
        to_visit = [start_file]

        while to_visit:
            current = to_visit.pop(0)

            if current not in self.link_graph:
                continue

            for linked in self.link_graph[current]:
                if linked not in reachable:
                    reachable.add(linked)
                    to_visit.append(linked)

        return reachable

    def categorize_orphans(self) -> dict[str, list[Path]]:
        """Categorize orphaned files by type for easier review."""
        categories = {
            "phase1": [],
            "phase2": [],
            "phase3": [],
            "phase4": [],
            "phase5": [],
            "phase6": [],
            "phase7": [],
            "phase8": [],
            "session_reports": [],
            "handoffs": [],
            "status_updates": [],
            "guides": [],
            "implementation": [],
            "testing": [],
            "other": [],
        }

        for orphan in self.orphaned_files:
            name = orphan.name.lower()

            # Check phase
            for phase_num in range(1, 9):
                if f"phase{phase_num}" in name or f"phase_{phase_num}" in name:
                    categories[f"phase{phase_num}"].append(orphan)
                    break
            else:
                # Not a phase file, check other categories
                if any(word in name for word in ["session", "summary", "report", "complete"]):
                    categories["session_reports"].append(orphan)
                elif any(word in name for word in ["handoff", "transition"]):
                    categories["handoffs"].append(orphan)
                elif "status" in name:
                    categories["status_updates"].append(orphan)
                elif any(word in name for word in ["guide", "readme", "start_here", "quickstart"]):
                    categories["guides"].append(orphan)
                elif any(word in name for word in ["implementation", "fix", "debug"]):
                    categories["implementation"].append(orphan)
                elif any(word in name for word in ["test", "e2e", "integration"]):
                    categories["testing"].append(orphan)
                else:
                    categories["other"].append(orphan)

        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    def suggest_linking_location(self, orphan: Path) -> list[str]:
        """Generate suggestions for where to link the orphaned file."""
        suggestions = []
        name = orphan.name.lower()

        # Phase-specific suggestions
        for phase_num in range(1, 9):
            if f"phase{phase_num}" in name:
                suggestions.append(f"AGENTS.md - Phase {phase_num} section under '📋 Tracking'")
                suggestions.append(f"Create/update docs/agents/PHASE{phase_num}_INDEX.md")
                break

        # Type-specific suggestions
        if "handoff" in name:
            suggestions.append("Link from phase completion summary documents")
        if any(word in name for word in ["guide", "start_here", "quickstart"]):
            suggestions.append("Link from AGENTS.md or phase index files")
        if any(word in name for word in ["session", "summary", "report"]):
            suggestions.append("Link from session tracking or phase progress documents")
        if any(word in name for word in ["api", "architecture", "implementation"]):
            suggestions.append("Link from docs/index.md 'Additional Resources' section")
        if any(word in name for word in ["test", "e2e", "debug"]):
            suggestions.append("Link from testing documentation or developer guide")
        if any(word in name for word in ["fix", "debug", "troubleshoot"]):
            suggestions.append("Link from docs/index.md 'Troubleshooting' section")

        # Default suggestion
        if not suggestions:
            suggestions.append("Review content and link from appropriate documentation entry point")
            suggestions.append("Consider if file is obsolete and can be archived/deleted")

        return suggestions

    def audit(self):
        """Main audit method."""
        print("=" * 80)
        print("  HEIMDALL DOCUMENTATION AUDIT")
        print("=" * 80)
        print()

        # Build link graph
        self.build_link_graph()

        # Find reachable files from all entry points
        for entry_point in self.entry_points:
            if entry_point.exists():
                self.log(f"Scanning from entry point: {entry_point.relative_to(self.base_path)}")
                reachable = self.find_reachable_files(entry_point)
                self.reachable_files.update(reachable)
            else:
                self.log(
                    f"Entry point not found: {entry_point.relative_to(self.base_path)}", "warning"
                )

        # Calculate orphans
        self.orphaned_files = self.all_files - self.reachable_files

        # Print statistics
        print("📊 AUDIT STATISTICS:")
        print(f"   Total markdown files in docs/: {len(self.all_files)}")
        print(f"   Reachable from entry points: {len(self.reachable_files)}")
        print(f"   Orphaned files: {len(self.orphaned_files)}")
        print(f"   Broken links: {len(self.broken_links)}")
        print()

        return len(self.orphaned_files) == 0 and len(self.broken_links) == 0

    def generate_json_report(self, output_file: Path):
        """Generate JSON report of audit results."""
        categories = self.categorize_orphans()

        report = {
            "audit_timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_files": len(self.all_files),
                "reachable_files": len(self.reachable_files),
                "orphaned_files": len(self.orphaned_files),
                "broken_links": len(self.broken_links),
            },
            "orphaned_files": {
                category: [
                    {
                        "path": str(orphan.relative_to(self.base_path)),
                        "name": orphan.name,
                        "size_kb": round(orphan.stat().st_size / 1024, 2),
                        "modified": datetime.fromtimestamp(orphan.stat().st_mtime).isoformat(),
                        "suggestions": self.suggest_linking_location(orphan),
                    }
                    for orphan in sorted(files)
                ]
                for category, files in categories.items()
            },
            "broken_links": [
                {"source_file": str(source.relative_to(self.base_path)), "broken_url": url}
                for source, url in self.broken_links
            ],
            "entry_points": [
                str(ep.relative_to(self.base_path)) for ep in self.entry_points if ep.exists()
            ],
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"✅ JSON report saved to: {output_file.relative_to(self.base_path)}")

    def generate_markdown_report(self, output_file: Path):
        """Generate Markdown report of audit results."""
        categories = self.categorize_orphans()

        lines = [
            "# Documentation Audit Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Total markdown files**: {len(self.all_files)}",
            f"- **Reachable files**: {len(self.reachable_files)}",
            f"- **Orphaned files**: {len(self.orphaned_files)}",
            f"- **Broken links**: {len(self.broken_links)}",
            "",
        ]

        if self.orphaned_files:
            lines.extend(
                [
                    "## Orphaned Files",
                    "",
                    "Files not linked from any entry point (AGENTS.md, docs/index.md):",
                    "",
                ]
            )

            for category, files in sorted(categories.items()):
                lines.append(f"### {category.upper().replace('_', ' ')} ({len(files)} files)")
                lines.append("")

                for orphan in sorted(files):
                    rel_path = orphan.relative_to(self.base_path)
                    stat = orphan.stat()
                    size_kb = stat.st_size / 1024
                    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")

                    lines.append(f"#### `{rel_path}`")
                    lines.append(f"- **Size**: {size_kb:.1f} KB")
                    lines.append(f"- **Last Modified**: {modified}")
                    lines.append("- **Suggested Actions**:")

                    suggestions = self.suggest_linking_location(orphan)
                    for suggestion in suggestions[:3]:  # Top 3 suggestions
                        lines.append(f"  - {suggestion}")

                    lines.append("")

                lines.append("")
        else:
            lines.extend(
                ["## ✅ No Orphaned Files", "", "All documentation files are properly linked!", ""]
            )

        if self.broken_links:
            lines.extend(["## Broken Links", "", "Links pointing to non-existent files:", ""])

            for source, url in sorted(self.broken_links):
                rel_source = source.relative_to(self.base_path)
                lines.append(f"- `{rel_source}` → `{url}`")

            lines.append("")

        lines.extend(
            [
                "## Next Steps",
                "",
                "1. Review each orphaned file's content",
                "2. Determine if file is still relevant",
                "3. Link from appropriate documentation (AGENTS.md, phase indices, etc.)",
                "4. Archive or delete obsolete files",
                "5. Fix broken links",
                "6. Re-run audit to verify: `make audit-docs`",
                "",
            ]
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"✅ Markdown report saved to: {output_file.relative_to(self.base_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Audit documentation for orphaned files and broken links",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/audit_documentation.py                    # Console output
  python scripts/audit_documentation.py --format=json      # JSON report
  python scripts/audit_documentation.py --format=markdown  # Markdown report
  python scripts/audit_documentation.py --format=both      # Both formats
  python scripts/audit_documentation.py --verbose          # Detailed output

Exit Codes:
  0 = No issues found (all files linked, no broken links)
  1 = Issues detected (orphaned files or broken links)
        """,
    )

    parser.add_argument(
        "--format",
        choices=["json", "markdown", "both"],
        default=None,
        help="Output format for reports",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed output during processing"
    )

    args = parser.parse_args()

    # Get base path
    base_path = Path(__file__).parent.parent

    # Create auditor and run audit
    auditor = DocumentationAuditor(base_path, verbose=args.verbose)
    is_clean = auditor.audit()

    # Generate reports if requested
    if args.format in ["json", "both"]:
        json_file = base_path / "audit_report.json"
        auditor.generate_json_report(json_file)

    if args.format in ["markdown", "both"]:
        md_file = base_path / "audit_report.md"
        auditor.generate_markdown_report(md_file)

    # Print final status
    print()
    if is_clean:
        print("✅ AUDIT PASSED: No orphaned files or broken links found!")
        sys.exit(0)
    else:
        print("❌ AUDIT FAILED: Orphaned files or broken links detected!")
        print()
        print("Run with --format=both to generate detailed reports:")
        print("  python scripts/audit_documentation.py --format=both")
        sys.exit(1)


if __name__ == "__main__":
    main()
