# Guide 6: CLI Reference & Declarative TOML Orchestration

`malthus-gp` provides both a standalone command-line tool (`mgp`) and declarative TOML experiment orchestration via the MalthusJAX engine.

---

## 1. CLI Overview (`mgp` & `mjax gp`)

Because `malthus-gp` registers a console entrypoint with MalthusJAX, all commands can be invoked using either `mgp` or `mjax gp`:

```bash
mgp --help
# or
mjax gp --help
```

### Core Subcommands

| Command | Purpose |
| :--- | :--- |
| `fit` | Trains a symbolic model on dataset $X, y$ and exports the best program. |
| `predict` | Runs batch inference using a saved model TOML on new input data. |
| `formula` | Decodes and prints the mathematical expression from a saved model TOML. |
| `export-toml` | Generates a template TOML configuration for reproducible experiments. |

---

## 2. Command Reference & Examples

### 2.1 Training a Model (`mgp fit`)

Train a Cartesian GP Regressor on a CSV dataset:

```bash
mgp fit \
  --data data/benchmark.csv \
  --target target_col \
  --model cgp_regressor \
  --generations 500 \
  --pop-size 5 \
  --params "n_cols=30,mutation_rate=0.04" \
  --output models/cgp_best.toml \
  --plot plots/fit_result.png
```

To fit a Differentiable CGP model with inner Adam gradient steps:
```bash
mgp fit \
  --data data/benchmark.csv \
  --model dcgp_regressor \
  --generations 200 \
  --params "n_cols=25,lsmf_epochs=15,learning_rate=0.02" \
  --output models/dcgp_best.toml
```

### 2.2 Inspecting Mathematical Formulas (`mgp formula`)

Print the algebraic expression evolved in a saved model:

```bash
mgp formula --model-toml models/cgp_best.toml
```

### 2.3 Running Inference (`mgp predict`)

Apply the saved model to evaluate new test features:

```bash
mgp predict \
  --model-toml models/cgp_best.toml \
  --data data/test_features.npy \
  --output predictions.npy
```

---

## 3. Distributed Island Models with MalthusJAX

`malthus-gp` can be orchestrated across parallel distributed islands using MalthusJAX's `IslandModel`. Multiple independent $(1+\lambda)$-ES populations drift through neutral networks concurrently, periodically exchanging top programs to escape local basins:

```python
from malthus_gp.learn import CGPRegressor

# Configure 4 parallel islands with program migration
model = CGPRegressor(
    n_rows=1,
    n_cols=25,
    pop_size=5,
    max_generations=500,
    engine="island",
    engine_params={
        "num_islands": 4,          # 4 concurrent island populations
        "migration_interval": 50,  # Exchange programs every 50 generations
        "num_migrants": 1,         # Send 1 elite per migration
    },
    random_state=42,
)
model.fit(X_train, y_train)
```
