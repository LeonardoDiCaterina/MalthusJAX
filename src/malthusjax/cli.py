"""Console entry point with a plugin-free documentation path."""

import sys


def main(args: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if args is None else args
    if arguments and arguments[0] == "skill":
        from .skill import main as skill_main

        return skill_main(arguments[1:])

    from .benchmarking.cli import main as benchmark_main

    return benchmark_main(arguments)
