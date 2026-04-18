import argparse
import faulthandler
import os
import signal
import sys
import time
from pathlib import Path
from threading import Event, Thread

from malthusjax.composer import Composer
from malthusjax.composer.config import load_experiment_config


def _log(message: str) -> None:
    print(message, flush=True)


def _start_heartbeat(pipeline_name: str, interval_s: float = 30.0) -> tuple[Event, Thread]:
    stop_event = Event()
    start_time = time.time()

    def _heartbeat() -> None:
        while not stop_event.wait(interval_s):
            elapsed = time.time() - start_time
            _log(
                f"  ... still running pipeline '{pipeline_name}' "
                f"(elapsed {elapsed:.1f}s)"
            )

    thread = Thread(target=_heartbeat, daemon=True)
    thread.start()
    return stop_event, thread


def _configure_runtime_diagnostics() -> None:
    """Enable low-overhead runtime diagnostics for long nohup runs.

    - `kill -USR1 <pid>` dumps Python stack traces into the active log stream.
    - Set `MALTHUSJAX_STACK_DUMP_INTERVAL=<seconds>` to emit periodic stack dumps.
    """
    faulthandler.enable()

    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True)
        _log("Diagnostics: send SIGUSR1 to dump Python stack traces.")
    except (AttributeError, OSError, RuntimeError):
        _log("Diagnostics: SIGUSR1 stack-dump hook unavailable on this platform.")

    interval_raw = os.environ.get("MALTHUSJAX_STACK_DUMP_INTERVAL", "").strip()
    if not interval_raw:
        return

    try:
        interval_s = float(interval_raw)
    except ValueError:
        _log(
            "Diagnostics: ignoring invalid MALTHUSJAX_STACK_DUMP_INTERVAL "
            f"value={interval_raw!r}."
        )
        return

    if interval_s <= 0:
        return

    faulthandler.dump_traceback_later(interval_s, repeat=True)
    _log(f"Diagnostics: periodic stack dump enabled every {interval_s:.1f}s.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MalthusJAX experiments from TOML configuration files"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/mock_binary_experiment.toml",
        help="Path to TOML configuration file (default: examples/mock_binary_experiment.toml)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default=None,
        help="Run only a specific pipeline (optional; runs all if not specified)",
    )
    args = parser.parse_args()

    _configure_runtime_diagnostics()

    _log(f"Loading TOML configuration: {args.config}")
    try:
        # load_experiment_config returns ExperimentLoadResult with: meta, pipelines, data_registry
        result = load_experiment_config(
            args.config, pipelines=[args.pipeline] if args.pipeline else None
        )
        meta = result.meta
        pipelines = result.pipelines
        data_registry = result.data_registry
    except FileNotFoundError as e:
        _log(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        _log(f"Error loading config: {e}")
        sys.exit(1)

    _log(f"✓ Loaded config: {meta.get('name')}")
    _log(f"  Output dir: {meta.get('output_dir')}")
    _log(f"  Running {len(pipelines)} pipeline(s)...\n")

    composer = Composer()

    total_time = 0.0
    results_summary = []
    had_errors = False

    for pipeline_name, kwargs in pipelines.items():
        _log(f"→ Pipeline: {pipeline_name}")
        start_t = time.time()

        pipeline_output_dir = Path(meta.get("output_dir")) / pipeline_name

        # kwargs already has seeds, generations, operators, experiment_name from TOML
        heartbeat_stop, heartbeat_thread = _start_heartbeat(pipeline_name)
        try:
            result = composer.quick_run(
                output_dir=pipeline_output_dir,
                data_config=data_registry,
                **kwargs
            )
        except Exception as e:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=0.2)
            _log(f"  ✗ ERROR: {str(e)[:100]}")
            results_summary.append((pipeline_name, "ERROR", 0.0, str(e)[:50]))
            had_errors = True
            continue
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=0.2)

        end_t = time.time()
        elapsed = end_t - start_t
        total_time += elapsed

        run_metrics = result.runs[0].metrics
        best_fit = run_metrics.get("best_fitness", "N/A")

        if best_fit != "N/A" and hasattr(best_fit, "item"):
            best_fit_val = float(best_fit.item())
        else:
            best_fit_val = best_fit

        status = "✓" if not result.runs[0].error else "✗"
        results_summary.append((pipeline_name, status, elapsed, best_fit_val))

        _log(f"  {status} Best: {best_fit_val:.6f} | Time: {elapsed:.2f}s")
        if result.runs[0].error:
            _log(f"  Error: {result.runs[0].error}")
            had_errors = True

    # Summary
    _log("\n" + "=" * 70)
    _log("SUMMARY")
    _log("=" * 70)
    for name, status, elapsed, fitness in results_summary:
        fit_str = f"{fitness:.6f}" if isinstance(fitness, float) else str(fitness)
        _log(f"{status} {name:30s} | Fitness: {fit_str:15s} | Time: {elapsed:7.2f}s")
    _log(f"\nTotal time: {total_time:.2f}s")

    if had_errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
