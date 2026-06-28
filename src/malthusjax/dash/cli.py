import argparse
import sys

from malthusjax.dash.config import load_config
from malthusjax.dash.plan import AnalysisPlan


def main():
    parser = argparse.ArgumentParser(description="MalthusDash: Analytical engine for MalthusJAX.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: run
    parser_run = subparsers.add_parser("run", help="Run a MalthusDash analysis plan.")
    parser_run.add_argument("config", type=str, help="Path to the TOML configuration file.")
    parser_run.add_argument(
        "-o", "--output", type=str, default="./dash_output", help="Output directory path."
    )

    args = parser.parse_args()

    if args.command == "run":
        try:
            config_dict = load_config(args.config)
            plan = AnalysisPlan(config_dict, output_dir=args.output)
            print(f"Executing plan from {args.config}...")
            plan.execute()
            print(f"Success. Outputs saved to {args.output}/")
        except Exception as e:
            print(f"Error executing plan: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
