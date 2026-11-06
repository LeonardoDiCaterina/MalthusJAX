#!/usr/bin/env python3
"""DeepMind MuJoCo MJX Benchmark Showcase: MalthusJAX vs EvoSAX vs QDax.

This showcase benchmarks three distinct evolutionary paradigms on continuous control
physics powered by DeepMind MuJoCo MJX:
  1. MalthusJAX Native GA (Tournament + Blend Crossover + Gaussian Mutation)
  2. EvoSAX Open_ES (Evolution Strategies for neural policy search)
  3. QDax MAP-Elites (Quality Diversity behavioral archive search)

Options:
  --task {cartpole,ant,both} : Choose between lightweight Cartpole and complex Ant quadruped.
  --render-gifs              : Record animated GIFs of initial vs final policy rollouts.
  --generations              : Number of evolutionary generations (default: 50).
  --pop-size                 : Population size per generation (default: 32).
  --seeds                    : List of random seeds (default: [42, 43, 44]).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Set default single-GPU placement on multi-GPU nodes before JAX initialization
# unless already configured by Slurm or user
if "CUDA_VISIBLE_DEVICES" not in os.environ and "SLURM_STEP_GPUS" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    import imageio
except ImportError:
    imageio = None

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from malthusjax.composer import Composer
from malthusjax.core.fitness.composable.environments import MuJoCoMJXEnv
from malthusjax.core.fitness.composable.interpreters import MLPInterpreter
from malthusjax.core.genome.real_genome import RealGenome


def get_task_config(task_name: str) -> Dict[str, Any]:
    """Retrieve environment and MLP policy configuration for the specified task."""
    if task_name == "cartpole":
        return {
            "env_name": "cartpole",
            "hidden_dims": (32, 32),
            "max_steps": 500,
            "render_height": 240,
            "render_width": 320,
            "fps": 30,
        }
    elif task_name == "ant":
        return {
            "env_name": "ant",
            "hidden_dims": (64, 64),
            "max_steps": 1000,
            "render_height": 240,
            "render_width": 320,
            "fps": 30,
        }
    else:
        raise ValueError(f"Unknown task: {task_name}. Choose 'cartpole' or 'ant'.")


def build_pipeline_definitions(pop_size: int) -> Dict[str, Dict[str, Any]]:
    """Build the 3 benchmark pipeline definitions."""
    return {
        "malthusjax_ga": {
            "backend": "malthusjax",
            "selection": f"tournament:num_selections={pop_size // 2},tournament_size=3",
            "crossover": "blend:alpha=0.5",
            "mutation": "gaussian:mutation_rate=0.2,mutation_strength=0.1",
        },
        "evosax_open_es": {
            "backend": "evosax",
            "evosax_strategy": "Open_ES",
        },
        "qdax_map_elites": {
            "backend": "qdax",
            "qdax_strategy": "MAPElites",
            "qdax_num_descriptors": 2,
            "qdax_num_centroids": 100,
        },
    }


def render_policy_rollout(
    env: MuJoCoMJXEnv,
    interpreter: MLPInterpreter,
    genome_weights: np.ndarray,
    max_steps: int = 250,
    height: int = 240,
    width: int = 320,
    seed: int = 0,
) -> List[np.ndarray]:
    """Roll out a single policy trajectory and render video frames using MuJoCo Renderer."""
    raw_model = getattr(env, "raw_model", None)
    if raw_model is None and hasattr(env, "mujoco_model"):
        raw_model = env.mujoco_model
    if hasattr(raw_model, "mj_model"):
        raw_model = raw_model.mj_model

    renderer = mujoco.Renderer(raw_model, height=height, width=width)
    data = mujoco.MjData(raw_model)

    key = jax.random.PRNGKey(seed)
    obs, state = env.reset(key)
    genome = RealGenome(values=jnp.asarray(genome_weights))

    @jax.jit
    def jitted_step(curr_state, curr_obs):
        flat_obs = env.preprocess_obs(curr_obs)
        action_logits = interpreter.apply(genome, flat_obs)
        action = env.postprocess_action(action_logits)
        next_obs, next_state, r, d, _ = env.step(curr_state, action)
        return next_obs, next_state, d

    # 1. Run JIT-compiled trajectory rollout collecting states
    states_to_render = [state]
    curr_obs = obs
    curr_state = state

    for _ in range(max_steps):
        curr_obs, curr_state, done = jitted_step(curr_state, curr_obs)
        states_to_render.append(curr_state)
        if bool(np.asarray(done)):
            break

    # 2. Render frames offline via MuJoCo Renderer
    frames: List[np.ndarray] = []
    for s in states_to_render:
        if hasattr(s, "data") and hasattr(s.data, "qpos"):
            data.qpos[:] = np.asarray(s.data.qpos)
            data.qvel[:] = np.asarray(s.data.qvel)
        elif hasattr(s, "pipeline_state") and hasattr(s.pipeline_state, "q"):
            data.qpos[:] = np.asarray(s.pipeline_state.q)
            data.qvel[:] = np.asarray(s.pipeline_state.qd)

        mujoco.mj_forward(raw_model, data)
        renderer.update_scene(data)
        frames.append(renderer.render())

    return frames


def run_showcase(
    task_name: str,
    generations: int = 50,
    pop_size: int = 32,
    seeds: Sequence[int] = (42, 43, 44),
    render_gifs: bool = True,
    output_dir: Optional[str] = None,
) -> None:
    """Run end-to-end 3-way benchmark showcase for a given task."""
    cfg = get_task_config(task_name)
    out_dir = Path(output_dir or f"results/showcase_{task_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_dir = out_dir / "gifs"
    if render_gifs:
        if imageio is None:
            print("⚠️  Warning: 'imageio' is not installed. Animated policy GIFs will be skipped.")
            print("    To enable GIFs: pip install imageio")
            render_gifs = False
        else:
            gif_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  DEEPMIND MUJOCO MJX SHOWCASE: {task_name.upper()}")
    print("  Pipelines: MalthusJAX GA vs EvoSAX Open_ES vs QDax MAP-Elites")
    print(f"  Pop Size: {pop_size} | Generations: {generations} | Seeds: {list(seeds)}")
    print(f"  Output Directory: {out_dir}")
    print(f"{'='*70}\n")

    # 1. Instantiate environment & interpreter to inspect policy structure
    env = MuJoCoMJXEnv.create(env_name=cfg["env_name"])
    interpreter = MLPInterpreter(
        input_dim=env.obs_dim,
        output_dim=env.action_dim,
        hidden=cfg["hidden_dims"],
        activation="tanh",
    )
    print(f"• Environment: {cfg['env_name']} (Obs: {env.obs_dim}, Action: {env.action_dim})")
    print(f"• Policy MLP: {env.obs_dim} -> {list(cfg['hidden_dims'])} -> {env.action_dim}")
    print(f"• Genome Parameters: {interpreter.num_params:,} weights & biases")

    # 2. Run multi-pipeline comparison
    composer = Composer.create_default()
    pipelines = build_pipeline_definitions(pop_size)

    fitness_spec = {
        "type": "RLEvaluator",
        "max_steps": cfg["max_steps"],
        "env": {"type": "MuJoCoMJXEnv", "env_name": cfg["env_name"]},
        "interpreter": {
            "type": "MLPInterpreter",
            "hidden": list(cfg["hidden_dims"]),
            "activation": "tanh",
        },
        "output": {"type": "ScalarOutput", "maximize": True},
    }

    print("\n>>> Launching Composer.compare() across 3 backends...")
    comparison = composer.compare(
        pipelines=pipelines,
        fitness=fitness_spec,
        pop_size=pop_size,
        generations=generations,
        seeds=tuple(seeds),
        shared_initial_population=False,
    )

    # 3. Print Summary Table
    print(f"\n{'='*70}")
    print("  BENCHMARK SUMMARY RESULTS")
    print(f"{'='*70}")
    summary = comparison.summary_table()
    for name, metrics in summary.items():
        best_f = metrics.get("best_fitness", {}).get("mean", float("nan"))
        exec_t = metrics.get("time_execution", {}).get("mean", float("nan"))
        warmup_t = metrics.get("time_warmup", {}).get("mean", float("nan"))
        total_t = metrics.get("time_total", {}).get("mean", float("nan"))
        print(f"\n[{name.upper()}]")
        print(f"  • Best Return (Reward) : {-best_f:.2f}")
        print(f"  • Execution Time       : {exec_t:.2f}s (Warmup: {warmup_t:.2f}s, Total: {total_t:.2f}s)")
        if "coverage" in metrics:
            print(f"  • QD Archive Coverage  : {metrics['coverage'].get('mean', 0.0):.1f}%")

    # 4. Generate Convergence Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    comparison.plot_convergence(seed_index=0, ax=ax)
    ax.set_title(f"MuJoCo MJX Continuous Control: {task_name.capitalize()}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Return (Higher is Better)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    plot_path = out_dir / "convergence.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n• Saved convergence plot: {plot_path}")

    # 5. Render Animated GIFs (Initial vs Final Best Policies)
    if render_gifs:
        print(f"\n{'='*70}")
        print("  RENDERING INITIAL VS FINAL POLICY GIFS")
        print(f"{'='*70}")

        for pipe_name in pipelines.keys():
            exp_result = comparison.pipelines[pipe_name]
            first_run = exp_result.runs[0]
            summary_dict = first_run.metrics

            # Extract final converged best policy weights
            final_weights = summary_dict.get("best_solution")
            if final_weights is None:
                # Fallback: draw random weights for untrained policy
                final_weights = np.zeros(interpreter.num_params)
            else:
                final_weights = np.asarray(final_weights)

            # Untrained initial policy (small random Gaussian perturbation)
            rng_init = np.random.default_rng(int(seeds[0]))
            initial_weights = rng_init.normal(0.0, 0.1, size=interpreter.num_params)

            # Render Initial GIF
            init_gif_path = gif_dir / f"{pipe_name}_initial.gif"
            print(f"  -> Rendering {pipe_name} [INITIAL (Gen 0)]...")
            init_frames = render_policy_rollout(
                env=env,
                interpreter=interpreter,
                genome_weights=initial_weights,
                max_steps=min(cfg["max_steps"], 250),
                height=cfg["render_height"],
                width=cfg["render_width"],
                seed=int(seeds[0]),
            )
            imageio.mimsave(str(init_gif_path), init_frames, fps=cfg["fps"])
            print(f"     Saved: {init_gif_path} ({len(init_frames)} frames)")

            # Render Final GIF
            final_gif_path = gif_dir / f"{pipe_name}_final.gif"
            print(f"  -> Rendering {pipe_name} [FINAL CONVERGED]...")
            final_frames = render_policy_rollout(
                env=env,
                interpreter=interpreter,
                genome_weights=final_weights,
                max_steps=min(cfg["max_steps"], 250),
                height=cfg["render_height"],
                width=cfg["render_width"],
                seed=int(seeds[0]),
            )
            imageio.mimsave(str(final_gif_path), final_frames, fps=cfg["fps"])
            print(f"     Saved: {final_gif_path} ({len(final_frames)} frames)")

    print(f"\n[DONE] Showcase for '{task_name}' completed successfully! Artifacts written to {out_dir}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepMind MuJoCo MJX Benchmark Showcase: MalthusJAX vs EvoSAX vs QDax"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="cartpole",
        choices=["cartpole", "ant", "both"],
        help="Physics environment task: 'cartpole' (fast), 'ant' (quadruped), or 'both'.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations to evolve (default: 50).",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=32,
        help="Population size per generation (default: 32).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Random seeds to evaluate (default: 42 43 44).",
    )
    parser.add_argument(
        "--render-gifs",
        action="store_true",
        default=True,
        help="Render animated GIFs of initial vs final policies (default: True).",
    )
    parser.add_argument(
        "--no-gifs",
        dest="render_gifs",
        action="store_false",
        help="Disable GIF rendering.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (default: results/showcase_<task>).",
    )

    args = parser.parse_args()

    # Headless rendering support for cluster environments
    if "MUJOCO_GL" not in os.environ:
        # Default to egl on Linux clusters if available, else osmesa / software
        if os.name != "nt" and os.uname().sysname == "Linux":
            os.environ["MUJOCO_GL"] = "egl"

    tasks = ["cartpole", "ant"] if args.task == "both" else [args.task]
    for t in tasks:
        out = (
            Path(args.output_dir) / t
            if (args.output_dir and args.task == "both")
            else args.output_dir
        )
        run_showcase(
            task_name=t,
            generations=args.generations,
            pop_size=args.pop_size,
            seeds=args.seeds,
            render_gifs=args.render_gifs,
            output_dir=str(out) if out else None,
        )


if __name__ == "__main__":
    main()
