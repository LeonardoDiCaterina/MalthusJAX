#!/usr/bin/env python3
"""
Docstring Coverage Checker for MalthusJAX

Scans Python files and reports on docstring coverage for classes and functions.
Helps track progress on the docstring refactoring effort.

Usage:
    python check_docstrings.py src/malthusjax/core/base.py
    python check_docstrings.py src/malthusjax/operators/
"""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class DocItem:
    """Represents a class, function, or method requiring a docstring."""
    name: str
    kind: str  # 'class', 'function', 'method'
    lineno: int
    has_docstring: bool
    is_public: bool  # True if not starting with _


class DocstringVisitor(ast.NodeVisitor):
    """AST visitor to find all classes and functions."""

    def __init__(self, filename: str):
        self.filename = filename
        self.items: List[DocItem] = []
        self.current_class: str = ""
        self._indent_stack: List[int] = [0]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        if not node.name.startswith('_'):  # Skip private classes
            has_docstring = ast.get_docstring(node) is not None
            self.items.append(DocItem(
                name=node.name,
                kind='class',
                lineno=node.lineno,
                has_docstring=has_docstring,
                is_public=True
            ))
            self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function and method definitions."""
        if not node.name.startswith('_'):  # Skip private methods
            has_docstring = ast.get_docstring(node) is not None
            kind = 'method' if self.current_class else 'function'
            self.items.append(DocItem(
                name=node.name,
                kind=kind,
                lineno=node.lineno,
                has_docstring=has_docstring,
                is_public=True
            ))
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef


def check_file(filepath: Path) -> Tuple[List[DocItem], int]:
    """Check docstring coverage in a single file."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        tree = ast.parse(source)
        visitor = DocstringVisitor(str(filepath))
        visitor.visit(tree)
        return visitor.items, len(source.splitlines())
    except SyntaxError as e:
        print(f"  ❌ Syntax error in {filepath}: {e}")
        return [], 0


def check_directory(dirpath: Path) -> Dict[Path, Tuple[List[DocItem], int]]:
    """Check all Python files in a directory."""
    results = {}
    for pyfile in dirpath.rglob('*.py'):
        if '__pycache__' not in str(pyfile):
            items, linecount = check_file(pyfile)
            if items:  # Only record files with classes/functions
                results[pyfile] = (items, linecount)
    return results


def print_file_report(filepath: Path, items: List[DocItem]) -> None:
    """Print coverage report for a single file."""
    documented = sum(1 for item in items if item.has_docstring)
    total = len(items)
    percentage = (documented / total * 100) if total > 0 else 0

    status_icon = "✅" if percentage == 100 else "⚠️ " if percentage >= 75 else "❌"
    try:
        display_path = filepath.relative_to(Path.cwd())
    except ValueError:
        display_path = filepath
    print(f"\n{status_icon} {display_path}")
    print(f"   Coverage: {documented}/{total} ({percentage:.0f}%)")

    # Show undocumented items
    undocumented = [item for item in items if not item.has_docstring]
    if undocumented:
        print("   Missing docstrings:")
        for item in undocumented:
            print(f"     - Line {item.lineno}: {item.kind} '{item.name}'")


def print_summary(results: Dict[Path, Tuple[List[DocItem], int]]) -> None:
    """Print overall summary."""
    total_items = 0
    documented_items = 0
    total_files = len(results)
    complete_files = 0

    for items, _ in results.values():
        for item in items:
            total_items += 1
            if item.has_docstring:
                documented_items += 1
        if all(item.has_docstring for item in items):
            complete_files += 1

    percentage = (documented_items / total_items * 100) if total_items > 0 else 0

    print("\n" + "=" * 70)
    print("DOCSTRING COVERAGE SUMMARY")
    print("=" * 70)
    print(f"Files scanned:        {total_files}")
    print(f"Files complete (100%): {complete_files}")
    print(f"Total items:          {total_items}")
    print(f"Documented items:     {documented_items}")
    print(f"Overall coverage:     {percentage:.1f}%")
    print("=" * 70)


def main(targets: List[str]) -> None:
    """Main entry point."""
    if not targets:
        print("Usage: python check_docstrings.py <file_or_directory> [...]")
        sys.exit(1)

    all_results = {}

    for target_str in targets:
        target = Path(target_str)
        if not target.exists():
            print(f"❌ Not found: {target}")
            continue

        if target.is_file():
            items, _ = check_file(target)
            if items:
                all_results[target] = (items, 0)
                print_file_report(target, items)
        elif target.is_dir():
            results = check_directory(target)
            all_results.update(results)
            for filepath in sorted(results.keys()):
                items, _ = results[filepath]
                print_file_report(filepath, items)

    if all_results:
        print_summary(all_results)
    else:
        print("No Python classes or functions found.")


if __name__ == '__main__':
    main(sys.argv[1:])
