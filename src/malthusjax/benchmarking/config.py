from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import toml


@dataclasses.dataclass
class SuiteConfig:
    name: str
    mode: Literal["cartesian", "lhs"]
    output_dir: str
    num_seeds: int
    aggregation_level: Literal["full_trace", "summary_only", "final_only"] = "summary_only"


@dataclasses.dataclass
class CartesianGridConfig:
    functions: List[str]
    dims: List[int]
    pops: List[int]
    gens: List[int]


@dataclasses.dataclass
class LHSGridConfig:
    functions: List[str]
    dims_min: int
    dims_max: int
    pops_min: int
    pops_max: int
    gens_min: int
    gens_max: int
    num_samples: int


@dataclasses.dataclass
class AnalysisConfig:
    reference_pipeline: str
    target_metrics: List[str] = dataclasses.field(default_factory=lambda: ["best_fitness", "execution_time"])


@dataclasses.dataclass
class BenchmarkConfig:
    suite: SuiteConfig
    grid: Union[CartesianGridConfig, LHSGridConfig]
    analysis: AnalysisConfig
    pipelines: Dict[str, Dict[str, Any]]

    @classmethod
    def from_toml(cls, toml_path: Path | str) -> BenchmarkConfig:
        with open(toml_path, "r") as f:
            data = toml.load(f)

        suite_data = data["suite"]
        suite = SuiteConfig(
            name=suite_data["name"],
            mode=suite_data["mode"],
            output_dir=suite_data["output_dir"],
            num_seeds=suite_data["num_seeds"],
            aggregation_level=suite_data.get("aggregation_level", "summary_only"),
        )

        grid_data = data["grid"]
        if suite.mode == "cartesian":
            grid: Any = CartesianGridConfig(
                functions=grid_data["functions"],
                dims=grid_data["dims"],
                pops=grid_data["pops"],
                gens=grid_data["gens"],
            )
        elif suite.mode == "lhs":
            grid = LHSGridConfig(
                functions=grid_data["functions"],
                dims_min=grid_data["dims_min"],
                dims_max=grid_data["dims_max"],
                pops_min=grid_data["pops_min"],
                pops_max=grid_data["pops_max"],
                gens_min=grid_data["gens_min"],
                gens_max=grid_data["gens_max"],
                num_samples=grid_data["num_samples"],
            )
        else:
            raise ValueError(f"Unknown suite mode: {suite.mode}")

        analysis_data = data.get("analysis", {})
        analysis = AnalysisConfig(
            reference_pipeline=analysis_data.get("reference_pipeline", ""),
            target_metrics=analysis_data.get("target_metrics", ["best_fitness", "execution_time"]),
        )

        pipelines = data.get("pipelines", {})
        if not pipelines:
            raise ValueError("No pipelines defined in TOML configuration.")

        return cls(
            suite=suite,
            grid=grid,
            analysis=analysis,
            pipelines=pipelines,
        )
