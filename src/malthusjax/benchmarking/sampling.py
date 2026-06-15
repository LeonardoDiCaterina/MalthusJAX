from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List

from scipy.stats import qmc

from .config import BenchmarkConfig, CartesianGridConfig, LHSGridConfig


def generate_grid(config: BenchmarkConfig) -> List[Dict[str, Any]]:
    """Generate the coordinate space based on the suite mode."""
    if config.suite.mode == "cartesian":
        return _generate_cartesian_grid(config)
    elif config.suite.mode == "lhs":
        return _generate_lhs_grid(config)
    else:
        raise ValueError(f"Unknown mode: {config.suite.mode}")


def _generate_cartesian_grid(config: BenchmarkConfig) -> List[Dict[str, Any]]:
    """Generate an exhaustive Cartesian product of all parameters."""
    assert isinstance(config.grid, CartesianGridConfig)
    
    coordinates = []
    
    # We create the full outer product
    for fn_name, D, P, G in itertools.product(
        config.grid.functions,
        config.grid.dims,
        config.grid.pops,
        config.grid.gens,
    ):
        coordinates.append({
            "fn_name": fn_name,
            "D": D,
            "P": P,
            "G": G,
        })
        
    return coordinates


def _generate_lhs_grid(config: BenchmarkConfig) -> List[Dict[str, Any]]:
    """Generate optimally stratified Latin Hypercube samples.
    
    Uses Log-Uniform mapping for Population Size to ensure coverage across
    exponentially increasing memory thresholds. Uses uniform mapping for 
    Dimensionality and Generations.
    """
    assert isinstance(config.grid, LHSGridConfig)
    
    sampler = qmc.LatinHypercube(d=3, seed=42)
    sample = sampler.random(n=config.grid.num_samples)
    
    # Pre-calculate log bounds for the population
    log_p_min = math.log10(config.grid.pops_min)
    log_p_max = math.log10(config.grid.pops_max)
    
    coordinates = []
    
    for fn_name in config.grid.functions:
        for i in range(config.grid.num_samples):
            # Column 0: Dimensionality (Linear mapping)
            d_raw = sample[i, 0]
            D = round(d_raw * (config.grid.dims_max - config.grid.dims_min) + config.grid.dims_min)
            
            # Column 1: Population Size (Log-Uniform mapping)
            p_raw = sample[i, 1]
            p_log = p_raw * (log_p_max - log_p_min) + log_p_min
            P = round(10 ** p_log)
            
            # Column 2: Generations (Linear mapping)
            g_raw = sample[i, 2]
            G = round(g_raw * (config.grid.gens_max - config.grid.gens_min) + config.grid.gens_min)
            
            # Ensure boundaries are strictly respected
            D = max(config.grid.dims_min, min(D, config.grid.dims_max))
            P = max(config.grid.pops_min, min(P, config.grid.pops_max))
            G = max(config.grid.gens_min, min(G, config.grid.gens_max))
            
            coordinates.append({
                "fn_name": fn_name,
                "D": D,
                "P": P,
                "G": G,
                "lhs_id": f"lhs{i:03d}"
            })
            
    return coordinates
