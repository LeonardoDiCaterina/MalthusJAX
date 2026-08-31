# MalthusJAX Experimental Walkthrough

This document serves as a comprehensive portfolio of the four major evolutionary architectures we've integrated and benchmarked within the MalthusJAX framework.

---

## Experiment 1: Multi Expression Programming (MEP)

**Objective:** Implement and benchmark a linear sequence representation where every gene can act as a potential output, allowing the evolutionary search to concurrently optimize multiple sub-programs.

### Implementation Details
- **`BasePrefixAwareGenome`**: We built a custom linear genome representation that tracks opcodes and argument pointers.
- **`PrefixEvaluator`**: Leveraging JAX's `lax.scan`, we vectorized the execution of the entire program sequence. Crucially, the evaluator evaluates the fitness of *every* node in the sequence concurrently, allowing the genome's "best" internal node to dynamically become the output.
- **Operators**: We implemented `LinearMicroMutation` to safely perturb operations and connection pointers without violating causality constraints (a node can only point to preceding nodes).

### Results
The `run_mep_experiment_1.py` successfully demonstrated that the MEP architecture could fit target curves efficiently by preserving sub-components within the same chromosome without destructive bloat.

---

## Experiment 2: Cartesian Genetic Programming (CGP)

**Objective:** Implement a 2D directed acyclic graph (DAG) representation subject to strict feed-forward topological boundaries (levels-back), and evaluate its baseline performance on regression tasks.

### Implementation Details
- **`CartesianGenome` & Config**: We designed a robust configuration that explicitly maps `(row, col)` grids, enforcing that connections can only point to primary inputs or previous nodes within a defined `levels_back` window.
- **`CartesianMicroMutation`**: We perfectly replicated Miller's 1+4 ES mutation operator, ensuring constant-time point mutations of `ops`, `args`, and `out_nodes`.
- **`CGPSelection`**: We enforced strict neutral drift, ensuring that when an offspring ties its parent in fitness, it aggressively replaces the parent.

### Results
The `run_cgp_experiment.py` benchmarked the architecture across various graph widths. We observed that while standard CGP is incredibly robust, it fundamentally struggled to efficiently navigate the continuous loss landscape of a polynomial regression task strictly via discrete operator mutations. This motivated our third experiment!

---

## Experiment 3: Differentiable CGP for Neural Architecture Search (dCGPANN)

**Objective:** Combine the topological evolution of CGP with the continuous gradient-descent optimization of Neural Networks via a Memetic algorithm (LSMF).

### Implementation Details
- **`NeuralCartesianGenome`**: We extended standard CGP to include two new arrays: continuous `weights` (Glorot initialized) and `biases` (Zero initialized). We explicitly overrode the engine's casting logic so these parameters remained pristine `jnp.float32` arrays while the topology remained `jnp.int32`.
- **`NeuralCartesianEvaluator`**: We built a fully vectorized forward-pass for arbitrary graph topologies: $N_i = F_i(\sum w \cdot N + b)$, dynamically selecting activations via `jax.lax.switch`.
- **`LSMFEvaluator`**: This was the masterstroke. We wrapped the base evaluator inside an `optax.adam` loop executed via `jax.lax.scan`. The evaluator trains the population for $C$ epochs per evaluation step, and executes a **Lamarckian update**, reconstructing the population with the newly optimized weights.

### Results
The `run_dcgpann_experiment.py` script successfully coupled $1+\lambda$ discrete search with continuous SGD. We witnessed the "Forget" step dynamically resetting continuous parameters after $K$ evolutionary cycles, correctly restarting the local search over the newly optimized graphs!

---

## Experiment 4: Differentiable Multi Expression Programming (dMEP)

**Objective:** Invent a novel architecture that hybridizes the multi-output, dynamically-routed architecture of MEP with the differentiable gradient-descent capabilities of dCGPANN.

### Implementation Details
- **`NeuralPrefixGenome`**: We extended the standard `LinearGenome` to include continuous `weights` and `biases` for every node in the sequence.
- **`NeuralPrefixEvaluator`**: We implemented a differentiable forward-pass that computes $a_i = F_i(\sum w_i \cdot N + b_i)$ for all $L$ nodes in the program.
- **Dynamic Gradient Routing**: The `LSMFEvaluator` was generalized to accept `Any` genome topology. Crucially, because MEP selects the best output via `jnp.min(mse_per_tree)`, the SGD gradient naturally and exclusively routes through the sub-graph that produced the minimal loss!
- **Bug Fix**: We identified a critical shape-broadcasting bug in JAX (`all_preds - y[:, None]`) that was silently reducing the loss function to the dataset variance! Fixing this unleashed the true power of both MEP and dMEP.

### Results
The `run_dmep_experiment.py` script yielded phenomenal results. While standard MEP achieved an MSE of `5.47e-03` on the complex Nguyen-7 function, our novel **dMEP** achieved an incredible **`7.45e-04`**, dominating even dCGPANN. The dynamic gradient routing successfully optimized the continuous weights of the active sub-program.

> [!TIP]
> **Next Steps:** We now have the foundation to scale these differentiable benchmarks to real-world datasets (like MNIST/CIFAR) or to execute massive distributed parameter sweeps using the MalthusJAX command-line tools.
