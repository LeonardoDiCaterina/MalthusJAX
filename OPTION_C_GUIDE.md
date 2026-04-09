# Option C: Data-Driven Evaluators Guide

## Overview

MalthusJAX now supports reproducible optimization by decoupling data specifications from evaluator logic. This allows you to load large datasets matrices, CSVs, NPZ files, or TSPLib files robustly while keeping TOML configs clean.

## Usage

### 1. The Classic Way (Still Works!)
```python
result = composer.quick_run(fitness="sphere:dim=10")
```

### 2. The Data ID Way (New)
You can store your datasets in a dictionary (Registry) and reference them dynamically:

```python
data_config = {
    "my_tsp_instance": {
        "source": "synthetic",
        "num_cities": 52,
        "random_seed": 42
    },
    "knapsack_data": {
        "source": "file",
        "path": "data/kp_100.csv"
    }
}

result = composer.quick_run(
    fitness="tsp:data_id=my_tsp_instance",
    data_config=data_config
)
```

## TOML Integration

You can define data sources directly in your experiment `.toml` files under `[data.*]` blocks. These sections are automatically loaded and parsed by `load_experiment_config()`.

```toml
[experiment]
name = "tsp_demo"

[data.berlin52_synthetic]
source = "synthetic"
num_cities = 52
random_seed = 42

[experiment.shared]
fitness = "tsp:data_id=berlin52_synthetic"
genome = "real:dim=52,bounds=(0,1)"

[pipelines.ga_baseline]
selection = "tournament:tournament_size=3"
crossover = "blend:alpha=0.5"
mutation = "gaussian:mutation_rate=0.1"
```

## Creating Your Own Data-Driven Evaluator

Extending this pattern to your own evaluator is simple. Ensure your factory method pops `_resolved_data`:

```python
def _create_my_evaluator(**kwargs):
    _resolved_data = kwargs.pop("_resolved_data", None)
    
    if _resolved_data is not None:
        # Load from the provided data dictionary/array
        return MyEvaluator.create_from_data(_resolved_data)
        
    # Default fallback
    return MyEvaluator.create_synthetic()
```
