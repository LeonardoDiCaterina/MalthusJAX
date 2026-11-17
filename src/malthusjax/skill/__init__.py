"""Retrieve the agent guide bundled with this installation."""

import argparse
import sys
from importlib.resources import files


def read() -> str:
    """Return the bundled skill as UTF-8 Markdown."""
    return files(__package__).joinpath("SKILL.md").read_text(encoding="utf-8")


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(args)
    sys.stdout.write(read())
    return 0
