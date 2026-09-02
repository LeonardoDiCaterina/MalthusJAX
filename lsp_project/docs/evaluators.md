# Evaluators: Execution and Lamarckian Learning

The Evaluator classes bridge the structural definitions of the Genomes with actionable execution routines (phenotypes) and training regimens. In MalthusJAX, evaluators are designed to directly mirror the theoretical architectures outlined in the Genomes documentation.

## 1. Discrete Evaluators (The Baseline)

The baseline evaluators handle discrete architectures. Because genome topologies (the sequence of instructions and arguments) can change dynamically during evolution, they present a challenge for XLA, which requires static graph definitions. We bypass this by wrapping the discrete genome arrays in a `jax.lax.scan`, compiling the sequential memory buffer lookups and opcode branching into a single, ultra-fast GPU kernel.

### `CartesianGPEvaluator`
This evaluator processes the `CartesianGenome`. According to Miller (2011), evaluating a CGP involves executing the feed-forward DAG sequentially. 
- **The Forward Pass:** Inputs are placed in a memory buffer. The evaluator iterates through the node indices, pulling inputs dynamically from the buffer using the connection pointers (`args`), executing the math function (`opcode`), and writing the result back.
- **The Readout & Neutral Drift:** Once the forward pass completes, this evaluator looks up the specific indices in the `out_nodes` integer array and returns those values as the final phenotype. Because fitness is computed strictly on these output nodes, disconnected nodes accumulate **Neutral Drift** (silent mutations).

### `LinearGPEvaluator`
This evaluator processes the `LinearGenome` (Multi Expression Programming). It utilizes the exact same `jax.lax.scan` sequential forward pass as CGP, but drastically alters the readout phase.
- **Dynamic Output Routing (SIP):** Instead of using a fixed `out_nodes` pointer, this evaluator leverages MEP's Strong Implicit Parallelism. Using `jax.vmap`, it simultaneously calculates the loss for *every single node* in the memory buffer against the target dataset. It then applies `jnp.argmin` to dynamically route the final output to whichever node achieved the lowest loss ("Best Gene Wins").

## 2. Differentiable Evaluators (Leap 1: Pure dCGP)

To implement "Pure dCGP", we must evaluate rigid mathematical operations (like `+`, `*`, `sin`) while extracting analytical derivatives of the graph output with respect to the inputs ($\frac{\partial y}{\partial x}$).

### `DifferentiableLinearGPEvaluator` (and Cartesian equivalent)
Instead of relying on Izzo's custom C++ Taylor polynomial algebra (AuDi), this evaluator simply takes the discrete `LinearGPEvaluator` or `CartesianGPEvaluator` and wraps the entire `jax.lax.scan` forward pass in `jax.grad`. 
This seamlessly extracts the exact, high-order analytical derivatives of the genetic program, allowing the fitness function to be defined by the residuals of **Differential Equations (PDEs/ODEs)** rather than standard symbolic regression points.

## 3. Neural Evaluators (Leap 2: dCGPANN & dMEP)

These evaluators process the hybrid neural variants by injecting continuous weights, biases, and non-linear activation functions into the `scan_fn` loop. 

### `NeuralCartesianEvaluator` & `NeuralPrefixEvaluator`
These evaluate the `NeuralCartesianGenome` and `NeuralPrefixGenome` respectively. During the sequential forward pass, the discrete topological pointers (`args`) dictate how the continuous parameters are multiplied:
```python
# 1. Discrete Topology: Gather values from previous nodes using integer pointers
gathered_inputs = memory[genome.args[node_idx]]

# 2. Continuous Parameters: Compute the weighted sum
z = jnp.sum(gathered_inputs * genome.weights[node_idx]) + genome.bias[node_idx]

# 3. Activation: Apply the mutated non-linear activation function
node_output = jax.lax.switch(genome.ops[node_idx], activation_funcs, z)
```
This architecture perfectly mimics a dense artificial neuron, entirely replacing rigid math operators.

## 4. The Lamarckian Wrapper

Because the Neural Evaluators above form a completely differentiable computational graph, we can perform gradient descent on the continuous weights. However, Gradient Descent is a local optimization technique, while Genetic Algorithms (NSGA-2, MAP-Elites) perform global structural search. MalthusJAX decouples these brilliantly.

### `LSMFEvaluator` (Lamarckian Surrogate Model-Free Evaluator)
This is a wrapper class. It takes any of the Neural Evaluators (e.g., `NeuralPrefixEvaluator`) and encapsulates it within an inner stochastic gradient descent loop (using `optax.adam`).

```mermaid
flowchart TD
    A[Start Generation] --> B[Population Initialization / Selection]
    B --> C[Discrete Mutation / Crossover (Global Topology Search)]
    C --> D{Evaluate Population}
    
    subgraph LSMFEvaluator
    D --> E[Inner Loop: SGD Epoch 1..N]
    E --> F[Forward Pass & Loss]
    F --> G[Backprop Gradients via Optax (Local Weight Search)]
    G --> E
    end
    
    E --> H[Return Updated Genome]
    H --> I[Lamarckian Write-Back]
    I --> J[End Generation / Select Best via NSGA-2]
    J --> A
```

**Decoupling:** The `LSMFEvaluator` returns both the fitness *and* the genome with its newly optimized continuous weights. This allows the overarching Genetic Engine (like NSGA-2) to perform non-dominated sorting and selection completely unaware of the inner SGD loop. The offspring then inherit these learned weights, fulfilling the definition of **Lamarckian Evolution** and drastically accelerating convergence.

---

## 5. Summary of Genome-to-Evaluator Mapping

To provide a quick reference, here is exactly how every core genome class maps to its corresponding evaluator in the MalthusJAX ecosystem:

### Discrete Architectures
1. **`CartesianGenome`** $\rightarrow$ **`CartesianGPEvaluator`**
   - *Execution:* Standard CGP symbolic regression using fixed `out_nodes` readouts.
2. **`LinearGenome`** $\rightarrow$ **`LinearGPEvaluator`**
   - *Execution:* Standard MEP symbolic regression using Dynamic Output Routing (SIP).

### Differentiable / Analytical Architectures (Leap 1)
3. **`LinearGenome`** / **`CartesianGenome`** $\rightarrow$ **`DifferentiableLinearGPEvaluator`** / **`DifferentiableMOEvaluator`**
   - *Execution:* Pure dCGP ODE/PDE solving. Evaluates discrete operations but wraps the execution in `jax.grad` to extract exact analytical derivatives.

### Neural / Continuous Architectures (Leap 2)
4. **`NeuralCartesianGenome`** $\rightarrow$ **`NeuralCartesianEvaluator`**
   - *Execution:* dCGPANN. Maps continuous weights onto the 2D Cartesian grid edges with activation functions.
5. **`NeuralPrefixGenome`** $\rightarrow$ **`NeuralPrefixEvaluator`**
   - *Execution:* dMEP. Maps continuous weights onto the 1D MEP sequence with activation functions.

### The Lamarckian Wrapper
6. **(Any Neural Genome)** $\rightarrow$ **`LSMFEvaluator`**
   - *Execution:* Wraps the Neural Evaluators above in an `optax.adam` SGD loop to locally optimize continuous weights *before* returning them to the global genetic engine (e.g., NSGA-2).
