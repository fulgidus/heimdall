#!/usr/bin/env python3
"""Generate SVG badge for coverage percentage"""

import json
import os
import sys


def get_coverage_percentage(coverage_json_path: str) -> float:
    """Extract coverage percentage from coverage.json"""
    try:
        with open(coverage_json_path) as f:
            data = json.load(f)
            return float(data["totals"]["percent_covered"])
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        return 0.0


def get_badge_color(percentage: float) -> str:
    """Determine badge color based on coverage percentage"""
    if percentage >= 85:
        return "31C754"  # Bright Green
    elif percentage >= 75:
        return "3fb950"  # Green
    elif percentage >= 60:
        return "DFCE00"  # Yellow
    elif percentage >= 40:
        return "FF9200"  # Orange
    else:
        return "E05D44"  # Red


def get_status_text(percentage: float) -> str:
    """Get status text for badge"""
    if percentage >= 85:
        return "excellent"
    elif percentage >= 75:
        return "good"
    elif percentage >= 60:
        return "fair"
    elif percentage >= 40:
        return "poor"
    else:
        return "critical"


def create_badge_svg(percentage: float, output_path: str) -> None:
    """Create SVG badge for coverage"""
    color = get_badge_color(percentage)
    status = get_status_text(percentage)

    # Calcola larghezze per il testo
    coverage_text = f"{percentage:.1f}%"

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="140" height="20" role="img" aria-label="coverage: {coverage_text}">
  <title>coverage: {coverage_text}</title>
  <defs>
    <linearGradient id="s" x2="0" y2="100%">
      <stop offset="0" stop-color="#bbb"/>
      <stop offset="1" stop-color="#999"/>
    </linearGradient>
    <clipPath id="r">
      <rect width="140" height="20" rx="3" fill="#fff"/>
    </clipPath>
  </defs>
  <g clip-path="url(#r)">
    <rect width="98" height="20" fill="#555"/>
    <rect x="98" width="42" height="20" fill="#{color}"/>
    <rect width="140" height="20" fill="url(#s)" opacity="0.1"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="110">
    <text aria-hidden="true" x="495" y="150" fill="#010101" fill-opacity="0.3" transform="scale(.1)" textLength="880">coverage</text>
    <text x="495" y="140" transform="scale(.1)" fill="#fff" textLength="880">coverage</text>
    <text aria-hidden="true" x="1180" y="150" fill="#010101" fill-opacity="0.3" transform="scale(.1)" textLength="320">{coverage_text}</text>
    <text x="1180" y="140" transform="scale(.1)" fill="#fff" textLength="320">{coverage_text}</text>
  </g>
</svg>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(svg_content)

    print(f"✅ Badge created: {output_path}")
    print(f"   Coverage: {percentage:.2f}% ({status})")


def create_na_badge_svg(output_path: str) -> None:
    """Create SVG badge with N/A when tests fail"""

    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="140" height="20" role="img" aria-label="coverage: N/A">
  <title>coverage: N/A</title>
  <defs>
    <linearGradient id="s" x2="0" y2="100%">
      <stop offset="0" stop-color="#bbb"/>
      <stop offset="1" stop-color="#999"/>
    </linearGradient>
    <clipPath id="r">
      <rect width="140" height="20" rx="3" fill="#fff"/>
    </clipPath>
  </defs>
  <g clip-path="url(#r)">
    <rect width="98" height="20" fill="#555"/>
    <rect x="98" width="42" height="20" fill="#9f9f9f"/>
    <rect width="140" height="20" fill="url(#s)" opacity="0.1"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="110">
    <text aria-hidden="true" x="495" y="150" fill="#010101" fill-opacity="0.3" transform="scale(.1)" textLength="880">coverage</text>
    <text x="495" y="140" transform="scale(.1)" fill="#fff" textLength="880">coverage</text>
    <text aria-hidden="true" x="1180" y="150" fill="#010101" fill-opacity="0.3" transform="scale(.1)" textLength="280">N/A</text>
    <text x="1180" y="140" transform="scale(.1)" fill="#fff" textLength="280">N/A</text>
  </g>
</svg>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(svg_content)

    print(f"✅ N/A Badge created: {output_path}")
    print("   Status: Tests failed or no coverage data available")


def main():
    """Main entry point"""
    coverage_json = "coverage_reports/backend.json"
    # Save badge in the correct path for GitHub Pages
    badge_path = "docs/coverage/develop/badge.svg"

    # Check if --na flag is passed
    if "--na" in sys.argv:
        create_na_badge_svg(badge_path)
    else:
        percentage = get_coverage_percentage(coverage_json)
        create_badge_svg(percentage, badge_path)


if __name__ == "__main__":
    main()
