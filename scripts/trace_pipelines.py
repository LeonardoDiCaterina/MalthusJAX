import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import jax

from malthusjax.benchmarking.config import BenchmarkConfig
from malthusjax.composer import Composer


def trace_single_pipeline(toml_path: str, trace_dir: str, target_pipeline: str, port: int):
    config = BenchmarkConfig.from_toml(toml_path)
    base_trace_dir = Path(trace_dir)

    composer = Composer.create_default()

    shared = {
        "fitness": "sphere:dim=10",
        "pop_size": 128,
        "generations": 200,
        "genome_length": 10,
        "elite_k": 21,
    }

    pipeline_def = config.pipelines[target_pipeline]

    formatted_pipeline_def = {}
    for k, v in pipeline_def.items():
        if isinstance(v, str):
            formatted_pipeline_def[k] = v.format(**shared)
        else:
            formatted_pipeline_def[k] = v

    out_dir = base_trace_dir / target_pipeline
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Tracing Pipeline: {target_pipeline} (seed 1) ---")
    jax.profiler.start_trace(str(out_dir))

    composer.compare(
        pipelines={target_pipeline: formatted_pipeline_def},
        seeds=[1],
        output_dir=f"results/traces/dummy_{target_pipeline}",
        **shared,
    )

    jax.profiler.stop_trace()
    print(f"  Trace saved to: {out_dir}")
    print(f"  View with:  tensorboard --logdir {base_trace_dir} --port {port}")
    print(f"  URL:        http://localhost:{port}")


def trace_all_pipelines(toml_path: str, trace_dir: str, port: int):
    print(f"Loading config from {toml_path}...")
    config = BenchmarkConfig.from_toml(toml_path)

    base_trace_dir = Path(trace_dir)
    if base_trace_dir.exists():
        shutil.rmtree(base_trace_dir)

    if config.suite.num_seeds > 1:
        print(f"\n[WARNING] TOML config specifies {config.suite.num_seeds} seeds.")
        print("[WARNING] The profiler will only trace the FIRST seed (seed 1)")
        print("[WARNING] to avoid redundant XLA compilations and excessive disk usage.\n")

    for pipeline_name in config.pipelines.keys():
        cmd = [
            sys.executable,
            __file__,
            "--toml",
            toml_path,
            "--out-dir",
            trace_dir,
            "--pipeline",
            pipeline_name,
            "--port",
            str(port),
        ]
        subprocess.run(cmd, check=True)

    print("\nAll traces complete!")
    print(f"  Launch TensorBoard:  tensorboard --logdir {base_trace_dir} --port {port}")
    print(f"  URL:                 http://localhost:{port}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trace all pipelines in a TOML config and view in TensorBoard."
    )
    parser.add_argument("--toml", required=True, help="Path to the TOML configuration file")
    parser.add_argument(
        "--out-dir",
        default="results/perf/traces",
        help="Output directory for traces (default: results/perf/traces)",
    )
    parser.add_argument(
        "--pipeline",
        default=None,
        help="Specific pipeline to trace (used internally for subprocess isolation)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="TensorBoard port to display in instructions (default: 6006)",
    )
    args = parser.parse_args()

    if args.pipeline:
        trace_single_pipeline(args.toml, args.out_dir, args.pipeline, args.port)
    else:
        trace_all_pipelines(args.toml, args.out_dir, args.port)
