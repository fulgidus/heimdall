#!/usr/bin/env python3
"""
Orphaned Documentation Finder for Heimdall Project

This script identifies documentation files that are not linked from any other
documentation file, particularly from the main docs/index.md entry point.

Unlike previous versions, this script:
- Does NOT automatically move or rename files
- Does NOT create deep nested directory structures
- Does NOT dump orphans into a single report file
- DOES provide clear identification of orphaned files
- DOES suggest appropriate linking locations
- DOES facilitate AI-assisted manual review

Usage:
    python scripts/find_orphan_docs.py
    python scripts/find_orphan_docs.py --detailed
    python scripts/find_orphan_docs.py --by-category
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
import json

# Base path
BASE_PATH = Path(__file__).parent.parent
DOCS_PATH = BASE_PATH / 'docs'
AGENTS_PATH = DOCS_PATH / 'agents'

# Key documentation files that should exist
KEY_FILES = [
    'index.md',
    'ARCHITECTURE.md',
    'API.md',
    'TRAINING.md',
    'installation.md',
    'usage.md',
    'contributing.md',
    'developer_guide.md'
]


class OrphanFinder:
    def __init__(self):
        self.docs_path = DOCS_PATH
        self.agents_path = AGENTS_PATH
        self.link_graph = {}
        self.all_files = set()
        self.linked_files = set()
        self.orphans = set()
        
    def find_all_markdown_links(self, content: str, source_file: Path) -> List[Path]:
        """Find all internal markdown links in content and resolve them to absolute paths."""
        links = []
        
        # Pattern for markdown links: [text](url)
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(pattern, content)
        
        for text, url in matches:
            # Skip external links
            if url.startswith('http://') or url.startswith('https://'):
                continue
            
            # Skip anchors-only links
            if url.startswith('#'):
                continue
            
            # Remove anchor from URL
            url = url.split('#')[0]
            if not url:
                continue
            
            # Resolve relative path
            try:
                # Handle relative paths
                if not url.startswith('/'):
                    linked_file = (source_file.parent / url).resolve()
                else:
                    linked_file = (BASE_PATH / url.lstrip('/')).resolve()
                
                # Check if file exists and is within docs
                if linked_file.exists() and str(linked_file).startswith(str(self.docs_path)):
                    links.append(linked_file)
            except Exception as e:
                # Silently skip invalid paths
                pass
        
        return links
    
    def build_link_graph(self):
        """Build a graph of all internal documentation links."""
        print("🔍 Scanning documentation files...")
        
        # Get all markdown files in docs/
        all_md_files = list(self.docs_path.glob('**/*.md'))
        self.all_files = set(all_md_files)
        
        print(f"   Found {len(all_md_files)} markdown files in docs/")
        
        # Build link graph for all files in docs/
        for file_path in all_md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                links = self.find_all_markdown_links(content, file_path)
                self.link_graph[file_path] = links
                
            except Exception as e:
                print(f"   ⚠️  Error reading {file_path.relative_to(BASE_PATH)}: {e}")
        
        # Also add AGENTS.md to link graph (it's in root, not docs/)
        agents_md = BASE_PATH / 'AGENTS.md'
        if agents_md.exists():
            try:
                with open(agents_md, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                links = self.find_all_markdown_links(content, agents_md)
                self.link_graph[agents_md] = links
                
            except Exception as e:
                print(f"   ⚠️  Error reading AGENTS.md: {e}")
    
    def find_reachable_files(self, start_file: Path) -> Set[Path]:
        """Find all files reachable from a starting file via links (BFS)."""
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
    
    def categorize_orphans(self) -> Dict[str, List[Path]]:
        """Categorize orphaned files by type for easier review."""
        categories = {
            'phase1': [],
            'phase2': [],
            'phase3': [],
            'phase4': [],
            'phase5': [],
            'phase6': [],
            'phase7': [],
            'general': [],
            'session_reports': [],
            'handoffs': [],
            'status_updates': [],
            'guides': [],
            'other': []
        }
        
        for orphan in self.orphans:
            name = orphan.name.lower()
            relative = orphan.relative_to(self.docs_path)
            
            # Check phase
            if 'phase1' in name or 'phase_1' in name:
                categories['phase1'].append(orphan)
            elif 'phase2' in name or 'phase_2' in name:
                categories['phase2'].append(orphan)
            elif 'phase3' in name or 'phase_3' in name:
                categories['phase3'].append(orphan)
            elif 'phase4' in name or 'phase_4' in name:
                categories['phase4'].append(orphan)
            elif 'phase5' in name or 'phase_5' in name:
                categories['phase5'].append(orphan)
            elif 'phase6' in name or 'phase_6' in name:
                categories['phase6'].append(orphan)
            elif 'phase7' in name or 'phase_7' in name:
                categories['phase7'].append(orphan)
            elif 'session' in name or 'summary' in name or 'report' in name:
                categories['session_reports'].append(orphan)
            elif 'handoff' in name or 'transition' in name:
                categories['handoffs'].append(orphan)
            elif 'status' in name:
                categories['status_updates'].append(orphan)
            elif 'guide' in name or 'readme' in name or 'start_here' in name:
                categories['guides'].append(orphan)
            elif str(relative).startswith('agents/'):
                categories['general'].append(orphan)
            else:
                categories['other'].append(orphan)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def suggest_linking_location(self, orphan: Path) -> List[str]:
        """Suggest where this orphaned file should be linked from."""
        suggestions = []
        name = orphan.name.lower()
        
        # Read file content for context
        try:
            with open(orphan, 'r', encoding='utf-8') as f:
                content = f.read()[:500]  # First 500 chars
        except:
            content = ""
        
        # Phase-specific suggestions
        if 'phase1' in name:
            suggestions.append("AGENTS.md - Phase 1 section")
            suggestions.append("Create docs/agents/PHASE1_INDEX.md and link from AGENTS.md")
        elif 'phase2' in name:
            suggestions.append("AGENTS.md - Phase 2 section")
            suggestions.append("Create docs/agents/PHASE2_INDEX.md and link from AGENTS.md")
        elif 'phase3' in name:
            suggestions.append("AGENTS.md - Phase 3 section")
            suggestions.append("Create docs/agents/PHASE3_INDEX.md and link from AGENTS.md")
        elif 'phase4' in name:
            suggestions.append("AGENTS.md - Phase 4 section")
            suggestions.append("Create docs/agents/PHASE4_INDEX.md and link from AGENTS.md")
        elif 'phase5' in name:
            suggestions.append("AGENTS.md - Phase 5 section")
            suggestions.append("Create docs/agents/PHASE5_INDEX.md and link from AGENTS.md")
        elif 'phase6' in name:
            suggestions.append("AGENTS.md - Phase 6 section")
            suggestions.append("Create docs/agents/PHASE6_INDEX.md and link from AGENTS.md")
        elif 'phase7' in name:
            suggestions.append("AGENTS.md - Phase 7 section")
            suggestions.append("Create docs/agents/PHASE7_INDEX.md and link from AGENTS.md")
        
        # Type-specific suggestions
        if 'handoff' in name:
            suggestions.append("Link from phase completion documents")
        if 'guide' in name or 'start_here' in name:
            suggestions.append("Link from AGENTS.md or phase index files")
        if 'session' in name or 'summary' in name:
            suggestions.append("Link from session tracking documents")
        if 'api' in name or 'architecture' in name:
            suggestions.append("Link from docs/index.md Additional Resources")
        
        # Default suggestion
        if not suggestions:
            suggestions.append("Review content and link from appropriate phase or general documentation")
        
        return suggestions
    
    def find_orphans(self):
        """Main method to find orphaned documentation files."""
        print("\n" + "=" * 70)
        print("  HEIMDALL ORPHANED DOCUMENTATION FINDER")
        print("=" * 70 + "\n")
        
        # Build link graph
        self.build_link_graph()
        
        # Find files reachable from docs/index.md
        index_file = self.docs_path / 'index.md'
        if not index_file.exists():
            print("❌ ERROR: docs/index.md not found!")
            return
        
        print(f"📍 Starting from: {index_file.relative_to(BASE_PATH)}")
        self.linked_files = self.find_reachable_files(index_file)
        
        # Also check AGENTS.md (important entry point)
        agents_md = BASE_PATH / 'AGENTS.md'
        if agents_md.exists():
            print(f"📍 Also checking: AGENTS.md")
            reachable_from_agents = self.find_reachable_files(agents_md)
            self.linked_files.update(reachable_from_agents)
        
        # Find orphans
        self.orphans = self.all_files - self.linked_files
        
        print(f"\n📊 Statistics:")
        print(f"   Total markdown files: {len(self.all_files)}")
        print(f"   Reachable files: {len(self.linked_files)}")
        print(f"   Orphaned files: {len(self.orphans)}")
        
        return self.orphans
    
    def print_detailed_report(self):
        """Print a detailed report of orphaned files with suggestions."""
        if not self.orphans:
            print("\n✅ No orphaned files found!")
            return
        
        print("\n" + "=" * 70)
        print("  ORPHANED FILES - DETAILED REPORT")
        print("=" * 70 + "\n")
        
        categories = self.categorize_orphans()
        
        for category, files in sorted(categories.items()):
            print(f"\n📂 Category: {category.upper()}")
            print("   " + "-" * 60)
            
            for orphan in sorted(files):
                rel_path = orphan.relative_to(self.docs_path)
                stat = orphan.stat()
                size_kb = stat.st_size / 1024
                modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d')
                
                print(f"\n   📄 {rel_path}")
                print(f"      Size: {size_kb:.1f} KB | Modified: {modified}")
                
                suggestions = self.suggest_linking_location(orphan)
                if suggestions:
                    print(f"      💡 Suggested linking locations:")
                    for suggestion in suggestions[:2]:  # Show top 2
                        print(f"         • {suggestion}")
        
        print("\n" + "=" * 70)
        print(f"  Total orphaned files: {len(self.orphans)}")
        print("=" * 70 + "\n")
    
    def print_category_summary(self):
        """Print a summary grouped by category."""
        if not self.orphans:
            print("\n✅ No orphaned files found!")
            return
        
        print("\n" + "=" * 70)
        print("  ORPHANED FILES - CATEGORY SUMMARY")
        print("=" * 70 + "\n")
        
        categories = self.categorize_orphans()
        
        for category, files in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"\n📂 {category.upper()}: {len(files)} files")
            for orphan in sorted(files)[:5]:  # Show first 5
                rel_path = orphan.relative_to(self.docs_path)
                print(f"   • {rel_path}")
            
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more")
        
        print("\n" + "=" * 70)
        print(f"  Total orphaned files: {len(self.orphans)}")
        print("=" * 70 + "\n")
    
    def print_simple_list(self):
        """Print a simple list of orphaned files."""
        if not self.orphans:
            print("\n✅ No orphaned files found!")
            return
        
        print("\n📋 Orphaned Files:\n")
        for orphan in sorted(self.orphans):
            rel_path = orphan.relative_to(self.docs_path)
            print(f"   • {rel_path}")
        
        print(f"\n   Total: {len(self.orphans)} files\n")
    
    def generate_fix_plan(self, output_file: Path):
        """Generate a markdown file with a plan to fix orphaned files."""
        categories = self.categorize_orphans()
        
        lines = [
            "# Orphaned Documentation Files - Fix Plan",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Orphaned Files**: {len(self.orphans)}",
            "",
            "## Overview",
            "",
            "This document provides a structured plan to properly integrate orphaned",
            "documentation files into the project's documentation hierarchy.",
            "",
            "## Categories and Recommendations",
            ""
        ]
        
        for category, files in sorted(categories.items()):
            lines.append(f"### {category.upper()} ({len(files)} files)")
            lines.append("")
            
            for orphan in sorted(files):
                rel_path = orphan.relative_to(self.docs_path)
                stat = orphan.stat()
                size_kb = stat.st_size / 1024
                
                lines.append(f"#### `{rel_path}`")
                lines.append(f"- Size: {size_kb:.1f} KB")
                lines.append(f"- **Action needed**: Review and link from:")
                
                suggestions = self.suggest_linking_location(orphan)
                for suggestion in suggestions[:3]:
                    lines.append(f"  - {suggestion}")
                
                lines.append("")
            
            lines.append("")
        
        lines.extend([
            "## Next Steps",
            "",
            "1. Review each file's content",
            "2. Determine if file is still relevant",
            "3. Link from appropriate documentation (AGENTS.md, phase indices, etc.)",
            "4. Remove obsolete files",
            "5. Update phase tracking documents to reference key files",
            ""
        ])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"💾 Fix plan saved to: {output_file.relative_to(BASE_PATH)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Find orphaned documentation files in Heimdall project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/find_orphan_docs.py                 # Simple list
  python scripts/find_orphan_docs.py --detailed      # Detailed report with suggestions
  python scripts/find_orphan_docs.py --by-category   # Summary by category
  python scripts/find_orphan_docs.py --generate-plan # Create fix plan document
        """
    )
    
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed report with linking suggestions')
    parser.add_argument('--by-category', action='store_true',
                        help='Show summary grouped by category')
    parser.add_argument('--generate-plan', action='store_true',
                        help='Generate a markdown fix plan document')
    
    args = parser.parse_args()
    
    finder = OrphanFinder()
    finder.find_orphans()
    
    if args.detailed:
        finder.print_detailed_report()
    elif args.by_category:
        finder.print_category_summary()
    elif args.generate_plan:
        plan_file = BASE_PATH / 'docs' / 'agents' / f"ORPHAN_FIX_PLAN.md"
        finder.generate_fix_plan(plan_file)
    else:
        finder.print_simple_list()


if __name__ == '__main__':
    main()
