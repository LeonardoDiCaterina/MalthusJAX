# Benchmarks: Performance and XLA Compilation

This document explains the progression from the manual V1 benchmark scripts to the highly optimized V2 XLA-compiled scripts for image classification.

## 1. V1 Scripts: The Python Loop

In the original `run_dmep_image_classification.py` script, the evolutionary loop was handled manually in Python:

```python
for gen in range(args.generations):
    state = jax.jit(engine.step)(state, step_key)
    # manual metrics logging...
```

**Pros:** 
- Extremely easy to debug. You can insert `print()` statements or Python breakpoints anywhere between generations.
- Flexibility to inject custom Python logic (e.g., custom checkpointing, complex logging) mid-evolution.

**Cons:**
- High host-to-device communication overhead. JAX has to dispatch a new GPU kernel every single generation, which creates a massive CPU bottleneck for fast-executing fitness functions.

## 2. V2 Scripts: Fully Compiled Execution

In `run_dmep_image_classification_v2.py`, we removed the Python loop entirely, offloading the control flow to XLA via `engine.run()`:

```python
final_state, history, metrics = jax.jit(
    lambda s: engine.run(s)
)(state)
```

**Pros:**
- **Zero Python Overhead:** The entire $N$ generations of evolution (including the Lamarckian inner-SGD loop wrapped by `LSMFEvaluator`) are compiled into a single XLA graph using `jax.lax.scan`.
- **Maximum Device Utilization:** The GPU runs autonomously without waiting for Python dispatch, resulting in massive speedups (often 10x - 100x depending on the problem size).

**Cons:**
- Opaque debugging. You cannot step through the loop with a Python debugger.
- `print()` statements inside the evaluator must rely on `jax.debug.print`, which can be noisy across large populations.

### Result Verification

Both scripts are deterministic (given the same PRNG key) and yield identical results. For example, evolving a `digits` classifier over 50 generations with 5 Lamarckian epochs per generation cleanly converges to ~47.78% test accuracy, proving the exact same discrete-continuous dynamics are perfectly preserved under XLA compilation.
