import argparse
import sys
from pathlib import Path
from typing import List, Optional

from malthusjax.core.logger import configure_logging, get_logger

logger = get_logger("dash.cli")


def main(args: Optional[List[str]] = None) -> None:
    log_parser = argparse.ArgumentParser(add_help=False)
    log_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable verbose debug logging (DEBUG level)",
    )
    log_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Suppress informational logging (WARNING level)",
    )
    log_parser.add_argument(
        "--log-file",
        type=Path,
        default=argparse.SUPPRESS,
        help="Destination file path for structured logs",
    )
    log_parser.add_argument(
        "--log-json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Format log output as JSON lines",
    )

    parser = argparse.ArgumentParser(
        prog="malthusdash",
        description="MalthusDash: Analytical engine for MalthusJAX.",
        parents=[log_parser],
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: run
    parser_run = subparsers.add_parser(
        "run", parents=[log_parser], help="Run a MalthusDash analysis plan."
    )
    parser_run.add_argument("config", type=str, help="Path to the TOML configuration file.")
    parser_run.add_argument(
        "-o", "--output", type=str, default="./dash_output", help="Output directory path."
    )

    parsed = parser.parse_args(args)

    verbose = getattr(parsed, "verbose", False)
    quiet = getattr(parsed, "quiet", False)
    log_file = getattr(parsed, "log_file", None)
    log_json = getattr(parsed, "log_json", False)

    if verbose:
        level = "DEBUG"
    elif quiet:
        level = "WARNING"
    else:
        level = "INFO"

    format_type = "json" if log_json else "color"
    configure_logging(level=level, log_file=log_file, format_type=format_type)

    if parsed.command == "run":
        try:
            from malthusjax.dash.config import load_config
            from malthusjax.dash.plan import AnalysisPlan

            config_dict = load_config(parsed.config)
            plan = AnalysisPlan(config_dict, output_dir=parsed.output)
            logger.info("Executing plan from %s...", parsed.config)
            plan.execute()
            logger.info("Success. Outputs saved to %s/", parsed.output)
        except ImportError as e:
            logger.error(
                "MalthusDash requires additional dependencies (%s). "
                "Please install them via: pip install 'malthusjax[stats]'",
                e,
            )
            sys.exit(1)
        except Exception as e:
            logger.error("Error executing plan: %s", e)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

