"""Allow running benchmarking CLI with python -m malthusjax.benchmarking"""

from .cli import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
