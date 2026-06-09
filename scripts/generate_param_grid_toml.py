#!/usr/bin/env python3
"""Generate a TOML parity suite by expanding a parameter grid.

Writes a TOML file with paired MalthusJAX/Evosax pipelines for each
combination of parameters so Composer can run wide parameter-space tests.

Usage: python scripts/generate_param_grid_toml.py [output_path]
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path


def fmt_float(f: float) -> str:
    s = f"{f}"
    if "." in s:
        return s
    return s + ".0"


def main(argv: list[str] | None = None) -> int:
    out = Path(argv[1]) if argv and len(argv) > 1 else Path("examples/param_grid_parity.toml")

    # Parameter grid (customise as needed)
    pop_sizes = [12, 24, 48]
    generations = [10, 20]
    crossover_rates = [0.2, 0.5]
    mutation_strengths = [0.05, 0.1]
    selections = [
        "tournament:num_selections=4,tournament_size=2",
        "elite_pool:elite_k=3",
    ]

    # Shared experiment metadata
    shared = {
        "fitness": "bbob:fn_name=rastrigin,num_dims=10,seed=0,maximize=false",
        "seeds": [0, 1, 2],
    }

    pipelines = []
    for ps, gen, cr, ms, sel in itertools.product(
        pop_sizes, generations, crossover_rates, mutation_strengths, selections
    ):
        name_base = f"p{ps}_g{gen}_cr{int(cr*100)}_ms{int(ms*1000)}_{sel.split(":")[0]}"
        # malthusjax pipeline
        mj = {
            "name": f"{name_base}_malthusjax",
            "config": {
                "pop_size": ps,
                "generations": gen,
                "crossover": f"blend:alpha={fmt_float(cr)}",
                "mutation": f"gaussian:mutation_strength={fmt_float(ms)}",
                "selection": sel,
            },
        }
        ev = {
            "name": f"{name_base}_evosax",
            "config": {
                "backend": "evosax",
                "evosax_strategy": "SimpleGA",
                "pop_size": ps,
                "generations": gen,
                "strategy_params": f"crossover_rate={fmt_float(cr)},std_schedule={fmt_float(ms)}",
                "selection": sel,
            },
        }
        pipelines.append(mj)
        pipelines.append(ev)

    # Build TOML text
    lines = []
    lines.append("[experiment.shared]")
    for k, v in shared.items():
        if isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        else:
            lines.append(f"{k} = {v}")
    lines.append("")

    lines.append("[experiment.options]")
    lines.append("shared_initial_population = true")
    lines.append("")

    # Pipelines
    for p in pipelines:
        lines.append(f"[pipelines.{p['name']}]")
        for k, v in p["config"].items():
            # strategy_params and similar are passed as plain strings here
            if isinstance(v, str) and not v.startswith("\["):
                lines.append(f'{k} = "{v}"')
            else:
                lines.append(f"{k} = {v}")
        lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Wrote {out} with {len(pipelines)} pipelines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
