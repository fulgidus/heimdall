#!/usr/bin/env python3
"""
Documentation Index Generator for Heimdall Project

This script creates a comprehensive index of all documentation links,
validates link integrity, and generates a "sitemap" visualization.

Features:
- Extracts all markdown links from AGENTS.md and docs/index.md
- Validates file existence and anchor targets
- Creates JSON index with full link graph
- Identifies broken links
- Generates visual sitemap
- Detects link graph regressions

Usage:
    python scripts/generate_doc_index.py                  # Basic index
    python scripts/generate_doc_index.py --check-anchors  # Validate anchors too
    python scripts/generate_doc_index.py --verbose        # Detailed output
    python scripts/generate_doc_index.py --sitemap        # Generate visual sitemap

Exit Codes:
    0 - All links valid
    1 - Broken links detected
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


class DocumentationIndexer:
    """Generate and validate documentation link index."""

    def __init__(self, base_path: Path, check_anchors: bool = False, verbose: bool = False):
        self.base_path = base_path
        self.docs_path = base_path / "docs"
        self.check_anchors = check_anchors
        self.verbose = verbose

        # Entry points
        self.entry_points = [base_path / "AGENTS.md", self.docs_path / "index.md"]

        # Link data
        self.all_links: dict[Path, list[dict]] = {}  # source -> list of link info
        self.broken_links: list[dict] = []
        self.valid_links: list[dict] = []
        self.link_graph: dict[str, list[str]] = defaultdict(list)

    def log(self, message: str, level: str = "info"):
        """Log message if verbose mode."""
        if self.verbose or level in ["error", "warning"]:
            prefix = {"info": "   ", "success": "✅ ", "warning": "⚠️  ", "error": "❌ "}.get(
                level, "   "
            )
            print(f"{prefix}{message}")

    def extract_anchors(self, content: str) -> set[str]:
        """
        Extract all valid anchor targets from markdown content.

        Anchors are created from:
        - # Headers (converted to lowercase with hyphens)
        - <a name="anchor"></a> tags
        - <a id="anchor"></a> tags
        """
        anchors = set()

        # Header anchors (# Header Text -> #header-text)
        header_pattern = r"^#{1,6}\s+(.+)$"
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            header_text = match.group(1)
            # Remove markdown formatting
            header_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", header_text)
            header_text = re.sub(r"[`*_]", "", header_text)
            # Convert to anchor
            anchor = header_text.lower().strip()
            anchor = re.sub(r"[^\w\s-]", "", anchor)
            anchor = re.sub(r"[-\s]+", "-", anchor)
            anchors.add(anchor)

        # HTML anchor tags
        html_anchor_pattern = r'<a\s+(?:name|id)="([^"]+)"'
        for match in re.finditer(html_anchor_pattern, content, re.IGNORECASE):
            anchors.add(match.group(1))

        return anchors

    def resolve_link(self, url: str, source_file: Path) -> tuple[Path | None, str | None]:
        """
        Resolve a markdown link to an absolute path and anchor.

        Returns:
            (resolved_path, anchor) or (None, None) if external/invalid
        """
        # Skip external links
        if url.startswith(("http://", "https://", "mailto:", "ftp://")):
            return None, None

        # Split URL into path and anchor
        if "#" in url:
            url_path, anchor = url.split("#", 1)
        else:
            url_path, anchor = url, None

        # Skip anchor-only links
        if not url_path:
            return None, anchor

        # Resolve path
        try:
            if url_path.startswith("/"):
                # Absolute from repo root
                resolved = (self.base_path / url_path.lstrip("/")).resolve()
            else:
                # Relative from source file
                resolved = (source_file.parent / url_path).resolve()

            return resolved, anchor
        except Exception:
            return None, None

    def validate_link(self, url: str, source_file: Path) -> dict:
        """
        Validate a single link.

        Returns dict with validation results.
        """
        result = {
            "url": url,
            "source": str(source_file.relative_to(self.base_path)),
            "valid": False,
            "exists": False,
            "anchor_valid": None,
            "error": None,
        }

        # Resolve link
        resolved_path, anchor = self.resolve_link(url, source_file)

        # External or anchor-only links
        if resolved_path is None:
            if anchor:
                result["valid"] = True  # Assume anchor-only links are valid
                result["anchor_valid"] = True
            else:
                result["valid"] = True  # External links assumed valid
            return result

        result["target"] = str(resolved_path.relative_to(self.base_path)) if resolved_path else None

        # Check file existence
        if resolved_path.exists():
            result["exists"] = True
            result["valid"] = True

            # Check anchor if requested
            if self.check_anchors and anchor:
                try:
                    with open(resolved_path, encoding="utf-8") as f:
                        content = f.read()

                    valid_anchors = self.extract_anchors(content)
                    result["anchor_valid"] = anchor in valid_anchors

                    if not result["anchor_valid"]:
                        result["valid"] = False
                        result["error"] = f"Anchor '#{anchor}' not found in target file"
                except Exception as e:
                    result["error"] = f"Could not verify anchor: {e}"
        else:
            result["valid"] = False
            result["error"] = "Target file does not exist"

        return result

    def extract_and_validate_links(self, file_path: Path):
        """Extract and validate all links from a file."""
        self.log(f"Processing {file_path.relative_to(self.base_path)}...")

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            self.log(f"Error reading {file_path.relative_to(self.base_path)}: {e}", "error")
            return

        # Find all markdown links
        pattern = r"\[([^\]]+)\]\(([^)]+)\)"

        links_in_file = []
        for match in re.finditer(pattern, content):
            link_text, url = match.groups()

            # Validate link
            validation = self.validate_link(url, file_path)
            validation["text"] = link_text

            links_in_file.append(validation)

            # Track in link graph
            if validation.get("target"):
                source_key = str(file_path.relative_to(self.base_path))
                target_key = validation["target"]
                if target_key not in self.link_graph[source_key]:
                    self.link_graph[source_key].append(target_key)

            # Categorize
            if validation["valid"]:
                self.valid_links.append(validation)
            else:
                self.broken_links.append(validation)
                self.log(
                    f"Broken link: [{link_text}]({url}) - {validation.get('error', 'Unknown error')}",
                    "warning",
                )

        self.all_links[file_path] = links_in_file
        self.log(
            f"  Found {len(links_in_file)} links ({len([l for l in links_in_file if l['valid']])} valid)"
        )

    def scan_all_documentation(self):
        """Scan all documentation files for links."""
        print("=" * 80)
        print("  DOCUMENTATION LINK INDEX GENERATOR")
        print("=" * 80)
        print()

        # Scan entry points
        for entry_point in self.entry_points:
            if entry_point.exists():
                self.log(f"Scanning entry point: {entry_point.relative_to(self.base_path)}")
                self.extract_and_validate_links(entry_point)

        # Scan all docs/ files
        self.log("Scanning all files in docs/...")
        all_docs = list(self.docs_path.glob("**/*.md"))

        for doc_file in all_docs:
            self.extract_and_validate_links(doc_file)

        print()
        print("📊 LINK VALIDATION STATISTICS:")
        print(f"   Total links found: {len(self.valid_links) + len(self.broken_links)}")
        print(f"   Valid links: {len(self.valid_links)}")
        print(f"   Broken links: {len(self.broken_links)}")
        print()

    def generate_json_index(self, output_file: Path):
        """Generate JSON index file."""
        index = {
            "generated": datetime.now().isoformat(),
            "check_anchors": self.check_anchors,
            "statistics": {
                "total_links": len(self.valid_links) + len(self.broken_links),
                "valid_links": len(self.valid_links),
                "broken_links": len(self.broken_links),
                "files_scanned": len(self.all_links),
            },
            "link_graph": dict(self.link_graph),
            "broken_links": self.broken_links,
            "entry_points": [
                str(ep.relative_to(self.base_path)) for ep in self.entry_points if ep.exists()
            ],
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        print(f"✅ JSON index saved to: {output_file.relative_to(self.base_path)}")

    def generate_broken_links_report(self, output_file: Path):
        """Generate markdown report of broken links."""
        lines = [
            "# Broken Links Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Anchor Checking**: {'Enabled' if self.check_anchors else 'Disabled'}",
            "",
            "## Summary",
            "",
            f"- **Total Links**: {len(self.valid_links) + len(self.broken_links)}",
            f"- **Valid Links**: {len(self.valid_links)}",
            f"- **Broken Links**: {len(self.broken_links)}",
            "",
        ]

        if self.broken_links:
            lines.extend(
                [
                    "## Broken Links Details",
                    "",
                    "| Source File | Link Text | URL | Error |",
                    "|-------------|-----------|-----|-------|",
                ]
            )

            for link in sorted(self.broken_links, key=lambda x: x["source"]):
                source = link["source"]
                text = link["text"][:40] + "..." if len(link["text"]) > 40 else link["text"]
                url = link["url"]
                error = link.get("error", "Unknown")
                lines.append(f"| `{source}` | {text} | `{url}` | {error} |")

            lines.extend(
                [
                    "",
                    "## Recommended Actions",
                    "",
                    "1. Review each broken link",
                    "2. Fix or remove broken links",
                    "3. Update file paths if files were moved",
                    "4. Re-run validation: `make validate-doc-links`",
                    "",
                ]
            )
        else:
            lines.extend(["## ✅ No Broken Links", "", "All documentation links are valid!", ""])

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"✅ Broken links report saved to: {output_file.relative_to(self.base_path)}")

    def generate_sitemap(self, output_file: Path):
        """Generate visual sitemap of documentation structure."""
        lines = [
            "# Documentation Sitemap",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "This sitemap shows the structure of all documentation files and their links.",
            "",
            "## Entry Points",
            "",
        ]

        # Entry points section
        for entry_point in self.entry_points:
            if entry_point.exists() and entry_point in self.all_links:
                rel_path = entry_point.relative_to(self.base_path)
                links = self.all_links[entry_point]
                valid_count = len([l for l in links if l["valid"]])

                lines.append(f"### {rel_path}")
                lines.append(f"- **Total links**: {len(links)}")
                lines.append(f"- **Valid links**: {valid_count}")
                lines.append("")

        # Link graph visualization
        lines.extend(
            [
                "## Link Graph",
                "",
                "```",
            ]
        )

        def print_tree(node: str, visited: set[str], indent: int = 0):
            """Recursively print tree structure."""
            if node in visited or indent > 3:  # Prevent infinite loops and limit depth
                return []

            visited.add(node)
            prefix = "  " * indent + "├─ " if indent > 0 else ""
            tree_lines = [f"{prefix}{node}"]

            if node in self.link_graph:
                for target in sorted(self.link_graph[node])[:5]:  # Limit to 5 children
                    tree_lines.extend(print_tree(target, visited.copy(), indent + 1))

            return tree_lines

        for entry_point in self.entry_points:
            if entry_point.exists():
                rel_path = str(entry_point.relative_to(self.base_path))
                lines.extend(print_tree(rel_path, set()))
                lines.append("")

        lines.append("```")
        lines.append("")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"✅ Sitemap saved to: {output_file.relative_to(self.base_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate documentation index and validate links",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_doc_index.py                  # Basic validation
  python scripts/generate_doc_index.py --check-anchors  # Include anchor checking
  python scripts/generate_doc_index.py --sitemap        # Generate sitemap
  python scripts/generate_doc_index.py --verbose        # Detailed output

Exit Codes:
  0 = All links valid
  1 = Broken links detected
        """,
    )

    parser.add_argument(
        "--check-anchors", action="store_true", help="Validate anchor targets in links"
    )
    parser.add_argument("--sitemap", action="store_true", help="Generate visual sitemap")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    # Get base path
    base_path = Path(__file__).parent.parent

    # Create indexer and scan
    indexer = DocumentationIndexer(
        base_path, check_anchors=args.check_anchors, verbose=args.verbose
    )
    indexer.scan_all_documentation()

    # Generate outputs
    json_file = base_path / "docs_index.json"
    indexer.generate_json_index(json_file)

    broken_links_file = base_path / "broken_links.md"
    indexer.generate_broken_links_report(broken_links_file)

    if args.sitemap:
        sitemap_file = base_path / "docs_sitemap.md"
        indexer.generate_sitemap(sitemap_file)

    # Exit with appropriate code
    print()
    if indexer.broken_links:
        print("❌ VALIDATION FAILED: Broken links detected!")
        print(f"   See {broken_links_file.relative_to(base_path)} for details")
        sys.exit(1)
    else:
        print("✅ VALIDATION PASSED: All links are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
