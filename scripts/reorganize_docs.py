#!/usr/bin/env python3
"""
Comprehensive Documentation Reorganization Script for Heimdall Project

This script handles:
1. Translation of Italian markdown files to English
2. Standardization of file names in docs/agents/ to YYYYMMDD_HHmmss_description.md format
3. Identification and reporting of orphaned documentation files
4. Link validation and updates after renaming

Usage:
    python scripts/reorganize_docs.py --translate     # Translate Italian files
    python scripts/reorganize_docs.py --rename        # Rename files in docs/agents/
    python scripts/reorganize_docs.py --find-orphans  # Find orphaned files
    python scripts/reorganize_docs.py --all           # Run all tasks
"""

import os
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
import json

# Base path
BASE_PATH = Path(__file__).parent.parent

# Italian to English translations (comprehensive dictionary)
ITALIAN_TRANSLATIONS = {
    # Major section headers
    r'\bFASE\b': 'PHASE',
    r'\bCOMPLETATO\b': 'COMPLETED',
    r'\bCOMPLETA\b': 'COMPLETE',
    r'\bSTATUS\b': 'STATUS',
    r'\bSESSIONE\b': 'SESSION',
    
    # Time and dates
    r'\bDurata\b': 'Duration',
    r'\bData\b': 'Date',
    r'\bgiorno\b': 'day',
    r'\bgiorni\b': 'days',
    r'\bore\b': 'hours',
    
    # Results and status
    r'\bRisultati\b': 'Results',
    r'\brisultati\b': 'results',
    r'\bRiepilogo\b': 'Summary',
    r'\briepilogo\b': 'summary',
    
    # Actions
    r'\bImplementazione\b': 'Implementation',
    r'\bimplementazione\b': 'implementation',
    r'\bVerifica\b': 'Verification',
    r'\bverifica\b': 'verification',
    r'\bGuida\b': 'Guide',
    r'\bguida\b': 'guide',
    
    # Documentation
    r'\bDocumentazione\b': 'Documentation',
    r'\bdocumentazione\b': 'documentation',
    r'\bFile\b': 'Files',
    
    # Quality
    r'\bMetriche\b': 'Metrics',
    r'\bmetriche\b': 'metrics',
    r'\bProblemi\b': 'Issues',
    r'\bproblemi\b': 'issues',
    r'\bRisolto\b': 'Fixed',
    r'\brisolto\b': 'fixed',
    
    # Project terms
    r'\bProgetto\b': 'Project',
    r'\bprogetto\b': 'project',
    r'\bProssimo\b': 'Next',
    r'\bprossimo\b': 'next',
    
    # Common phrases
    r'\bè stato\b': 'has been',
    r'\bsono stati\b': 'have been',
    r'\bQuesto\b': 'This',
    r'\bquesto\b': 'this',
    r'\bQuesta\b': 'This',
    r'\bquesta\b': 'this',
    r'\bAbbiamo\b': 'We have',
    r'\babbiamo\b': 'we have',
    
    # Technical
    r'\bservizi\b': 'services',
    r'\bservizio\b': 'service',
    r'\bsistema\b': 'system',
    r'\bcodice\b': 'code',
    
    # State
    r'\bModificati\b': 'Modified',
    r'\bmodificati\b': 'modified',
    r'\bGenerati\b': 'Generated',
    r'\bgenerati\b': 'generated',
    r'\bCreati\b': 'Created',
    r'\bcreati\b': 'created',
    r'\bAggiunto\b': 'Added',
    r'\baggiunto\b': 'added',
    r'\bAggiornato\b': 'Updated',
    r'\baggiornato\b': 'updated',
    
    # Misc
    r'\btutto\b': 'all',
    r'\btutti\b': 'all',
    r'\btutte\b': 'all',
    r'\bpronto\b': 'ready',
    r'\bpronta\b': 'ready',
}

# Files that should not be renamed (exceptions)
RENAME_EXCEPTIONS = [
    'README.md',
    'CHANGELOG.md', 
    'AGENTS.md',
    '.gitignore',
    '.env.example'
]

class DocumentationReorganizer:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.docs_agents_path = base_path / 'docs' / 'agents'
        self.docs_path = base_path / 'docs'
        self.root_path = base_path
        
    def is_italian_content(self, content: str) -> bool:
        """Check if content contains Italian language markers."""
        italian_markers = [
            'è stato', 'sono stati', 'fase', 'abbiamo', 'completata',
            'questo', 'questa', 'sono state', 'è stata'
        ]
        return any(marker in content.lower() for marker in italian_markers)
    
    def translate_content(self, content: str) -> str:
        """Translate Italian content to English."""
        translated = content
        in_code_block = False
        lines = content.split('\n')
        translated_lines = []
        
        for line in lines:
            # Track code blocks
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                translated_lines.append(line)
                continue
            
            # Don't translate code blocks or lines with URLs
            if in_code_block or 'http://' in line or 'https://' in line:
                translated_lines.append(line)
                continue
            
            # Apply translations
            translated_line = line
            for it_pattern, en_replacement in ITALIAN_TRANSLATIONS.items():
                translated_line = re.sub(it_pattern, en_replacement, translated_line)
            
            translated_lines.append(translated_line)
        
        return '\n'.join(translated_lines)
    
    def translate_file(self, file_path: Path) -> bool:
        """Translate a single file if it contains Italian content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not self.is_italian_content(content):
                print(f"  ℹ️  {file_path.name} - Already in English, skipping")
                return False
            
            translated = self.translate_content(content)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(translated)
            
            print(f"  ✅ {file_path.name} - Translated successfully")
            return True
        except Exception as e:
            print(f"  ❌ {file_path.name} - Error: {e}")
            return False
    
    def translate_all_files(self):
        """Translate all Italian files in docs/agents/ and root."""
        print("\n🌍 Phase 1: Translation")
        print("=" * 60)
        
        # Files in docs/agents/
        print("\n📁 docs/agents/ directory:")
        agents_files = list(self.docs_agents_path.glob('*.md'))
        agents_translated = 0
        for file_path in agents_files:
            if self.translate_file(file_path):
                agents_translated += 1
        
        # Files in root
        print("\n📁 Root directory:")
        root_files = [f for f in self.root_path.glob('*.md') if f.name != 'README.md']
        root_translated = 0
        for file_path in root_files:
            if self.translate_file(file_path):
                root_translated += 1
        
        print(f"\n✅ Translation complete:")
        print(f"   - docs/agents/: {agents_translated} files translated")
        print(f"   - Root: {root_translated} files translated")
    
    def generate_standardized_filename(self, original_name: str, existing_names: Set[str]) -> str:
        """Generate a standardized filename in YYYYMMDD_HHmmss_description.md format."""
        # Default timestamp (2025-10-22 08:00:00)
        default_timestamp = "20251022_080000"
        
        # Remove .md extension
        name_without_ext = original_name.replace('.md', '')
        
        # Check if already has timestamp
        timestamp_pattern = r'^\d{8}_\d{6}'
        if re.match(timestamp_pattern, name_without_ext):
            return original_name  # Already standardized
        
        # Extract description from current name
        description = name_without_ext.lower()
        description = re.sub(r'[^a-z0-9]+', '_', description)
        description = description.strip('_')
        
        # Generate new name
        new_name = f"{default_timestamp}_{description}.md"
        
        # Handle duplicates
        counter = 1
        while new_name in existing_names:
            new_name = f"{default_timestamp}_{description}_{counter}.md"
            counter += 1
        
        return new_name
    
    def rename_agents_files(self, dry_run: bool = False):
        """Rename files in docs/agents/ to standardized format."""
        print("\n📝 Phase 2: File Renaming")
        print("=" * 60)
        
        agents_files = list(self.docs_agents_path.glob('*.md'))
        rename_map = {}
        existing_names = set()
        
        print(f"\n📊 Found {len(agents_files)} markdown files in docs/agents/")
        
        # Generate rename map
        for file_path in agents_files:
            if file_path.name in RENAME_EXCEPTIONS:
                print(f"  ⏭️  {file_path.name} - Exception, skipping")
                continue
            
            new_name = self.generate_standardized_filename(file_path.name, existing_names)
            existing_names.add(new_name)
            
            if new_name != file_path.name:
                rename_map[file_path] = self.docs_agents_path / new_name
                action = "Would rename" if dry_run else "Renaming"
                print(f"  📋 {action}: {file_path.name} → {new_name}")
        
        if not dry_run:
            # Perform renames
            for old_path, new_path in rename_map.items():
                try:
                    old_path.rename(new_path)
                    print(f"  ✅ Renamed: {old_path.name} → {new_path.name}")
                except Exception as e:
                    print(f"  ❌ Error renaming {old_path.name}: {e}")
        
        print(f"\n✅ Rename complete: {len(rename_map)} files {'would be' if dry_run else 'were'} renamed")
        
        return rename_map
    
    def find_all_links(self, content: str, file_path: Path) -> List[str]:
        """Find all markdown links in content."""
        # Pattern for markdown links: [text](url)
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(pattern, content)
        return [match[1] for match in matches]
    
    def build_link_graph(self) -> Dict[Path, List[str]]:
        """Build a graph of all internal links in documentation."""
        print("\n🔗 Building documentation link graph...")
        link_graph = {}
        
        # Scan all markdown files
        all_md_files = list(self.docs_path.glob('**/*.md'))
        
        for file_path in all_md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                links = self.find_all_links(content, file_path)
                link_graph[file_path] = links
            except Exception as e:
                print(f"  ⚠️  Error reading {file_path}: {e}")
        
        return link_graph
    
    def find_orphaned_files(self):
        """Find orphaned documentation files."""
        print("\n🔍 Phase 3: Finding Orphaned Files")
        print("=" * 60)
        
        link_graph = self.build_link_graph()
        
        # Get all markdown files
        all_files = set(self.docs_path.glob('**/*.md'))
        
        # Find files that are linked to
        linked_files = set()
        index_file = self.docs_path / 'index.md'
        
        # Start from index.md
        if index_file in link_graph:
            to_process = [(index_file, link) for link in link_graph[index_file]]
            visited = {index_file}
            
            while to_process:
                parent, link = to_process.pop(0)
                
                # Resolve relative link
                if link.startswith('http://') or link.startswith('https://'):
                    continue
                
                # Remove anchors
                link = link.split('#')[0]
                if not link:
                    continue
                
                # Resolve path
                try:
                    linked_file = (parent.parent / link).resolve()
                    if linked_file.exists() and linked_file not in visited:
                        visited.add(linked_file)
                        linked_files.add(linked_file)
                        if linked_file in link_graph:
                            to_process.extend([(linked_file, l) for l in link_graph[linked_file]])
                except Exception:
                    pass
        
        # Find orphans
        orphaned = all_files - linked_files - {index_file}
        
        print(f"\n📊 Documentation Statistics:")
        print(f"   - Total markdown files: {len(all_files)}")
        print(f"   - Files reachable from index: {len(linked_files) + 1}")
        print(f"   - Orphaned files: {len(orphaned)}")
        
        if orphaned:
            print(f"\n📋 Orphaned Files:")
            for orphan in sorted(orphaned):
                rel_path = orphan.relative_to(self.docs_path)
                print(f"   - {rel_path}")
                
            # Save report
            report_path = self.base_path / 'docs' / 'agents' / f"20251023_145400_orphaned_files_report.md"
            self.generate_orphan_report(orphaned, report_path)
            print(f"\n💾 Report saved to: {report_path.relative_to(self.base_path)}")
        
        return orphaned
    
    def generate_orphan_report(self, orphaned_files: Set[Path], report_path: Path):
        """Generate a detailed report of orphaned files."""
        report_lines = [
            "# Orphaned Documentation Files Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Orphaned Files**: {len(orphaned_files)}",
            "",
            "## Summary",
            "",
            "These files are not reachable from the main documentation index (`docs/index.md`).",
            "They may be:",
            "- Obsolete files that can be removed",
            "- Important files that need to be integrated into the documentation",
            "- Temporary/working files that should be moved or deleted",
            "",
            "## Orphaned Files",
            ""
        ]
        
        for orphan in sorted(orphaned_files):
            rel_path = orphan.relative_to(self.docs_path)
            stat = orphan.stat()
            modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d')
            size = stat.st_size
            
            report_lines.append(f"### `{rel_path}`")
            report_lines.append(f"- **Last Modified**: {modified}")
            report_lines.append(f"- **Size**: {size} bytes")
            report_lines.append("")
        
        report_lines.append("## Recommended Actions")
        report_lines.append("")
        report_lines.append("1. Review each orphaned file")
        report_lines.append("2. If important: Link from appropriate documentation")
        report_lines.append("3. If obsolete: Remove the file")
        report_lines.append("4. If temporary: Move to appropriate location")
        report_lines.append("")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

def main():
    parser = argparse.ArgumentParser(description='Reorganize Heimdall documentation')
    parser.add_argument('--translate', action='store_true', help='Translate Italian files to English')
    parser.add_argument('--rename', action='store_true', help='Rename files in docs/agents/')
    parser.add_argument('--find-orphans', action='store_true', help='Find orphaned files')
    parser.add_argument('--all', action='store_true', help='Run all tasks')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (no actual changes)')
    
    args = parser.parse_args()
    
    if not any([args.translate, args.rename, args.find_orphans, args.all]):
        parser.print_help()
        return
    
    reorganizer = DocumentationReorganizer(BASE_PATH)
    
    print("\n" + "=" * 60)
    print("  HEIMDALL DOCUMENTATION REORGANIZATION TOOL")
    print("=" * 60)
    
    if args.all or args.translate:
        reorganizer.translate_all_files()
    
    if args.all or args.rename:
        reorganizer.rename_agents_files(dry_run=args.dry_run)
    
    if args.all or args.find_orphans:
        reorganizer.find_orphaned_files()
    
    print("\n" + "=" * 60)
    print("  ✅ REORGANIZATION COMPLETE")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()
