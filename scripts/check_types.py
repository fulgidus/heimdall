#!/usr/bin/env python3
"""Comprehensive type checking script."""
import subprocess
import sys
from pathlib import Path
from typing import Tuple


def run_mypy(service_path: Path) -> Tuple[int, str]:
    """Run mypy on a service."""
    result = subprocess.run(
        ['mypy', str(service_path / 'src'), '--config-file', 'pyproject.toml', '--show-error-codes'],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout + result.stderr


def run_pylint(service_path: Path) -> Tuple[int, str]:
    """Run pylint on a service."""
    # Get absolute path to .pylintrc
    repo_root = Path(__file__).parent.parent
    pylintrc = repo_root / '.pylintrc'
    
    result = subprocess.run(
        ['pylint', str(service_path / 'src'), f'--rcfile={pylintrc}'],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout


def run_flake8(service_path: Path) -> Tuple[int, str]:
    """Run flake8 on a service."""
    result = subprocess.run(
        ['flake8', str(service_path / 'src')],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout


def main() -> int:
    """Run all type checks."""
    services_dir = Path('services')
    services = [d for d in services_dir.iterdir() if d.is_dir() and (d / 'src').exists()]
    
    total_issues = 0
    results: dict[str, dict[str, bool]] = {}
    
    for service in services:
        print(f"\n{'='*60}")
        print(f"Checking {service.name}")
        print('='*60)
        
        mypy_code, mypy_output = run_mypy(service)
        pylint_code, pylint_output = run_pylint(service)
        flake8_code, flake8_output = run_flake8(service)
        
        if mypy_code != 0:
            print("❌ mypy errors:")
            print(mypy_output)
            total_issues += mypy_code
        else:
            print("✅ mypy: passed")
        
        if pylint_code != 0:
            print("❌ pylint errors:")
            print(pylint_output)
            total_issues += pylint_code
        else:
            print("✅ pylint: passed")
        
        if flake8_code != 0:
            print("❌ flake8 errors:")
            print(flake8_output)
            total_issues += flake8_code
        else:
            print("✅ flake8: passed")
        
        results[service.name] = {
            'mypy': mypy_code == 0,
            'pylint': pylint_code == 0,
            'flake8': flake8_code == 0,
        }
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    for service, checks in results.items():
        status = "✅" if all(checks.values()) else "❌"
        print(f"{status} {service}")
    
    return 1 if total_issues > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
