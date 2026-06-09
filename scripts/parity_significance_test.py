#!/usr/bin/env python3
"""Compatibility wrapper for archived script.

This entrypoint is kept for backwards compatibility.
The implementation moved to scripts/_archive/parity_significance_test.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "scripts" / "_archive" / "parity_significance_test.py"
    print(
        "[DEPRECATED] scripts/parity_significance_test.py moved to "
        "scripts/_archive/parity_significance_test.py",
        file=sys.stderr,
    )
    cmd = [sys.executable, str(target), *sys.argv[1:]]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
