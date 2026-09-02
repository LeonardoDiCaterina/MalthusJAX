# Evolutionary Operators: Architecture & Theory

This document details the mutation, crossover, and reproduction operators implemented within the MalthusJAX framework. Our operators are specifically designed to adhere to modern paradigms in Genetic Programming (GP)—specifically addressing node integrity, topological awareness, and Lamarckian continuous updates.

---

## 1. Linear Operators (MEP)

The operators for `LinearGenome` strictly adhere to the Multi Expression Programming paradigm. Our implementation avoids destructive arbitrary crossover points in favor of modern structural preservation.

### 1.1 Mutation (`MEPMicroMutation` & Variants)
- **`MEPMicroMutation`**: Randomly alters a node's operator or routing arguments.
- **`SmoothMutation`**: Instead of completely random resampling, this operator biases changes towards numerically adjacent opcodes or argument values. This localized stepping ensures the semantic distance between the parent and offspring remains relatively small, flattening the rugged fitness landscape.
- **`AnnealedTopologicalMutation`**: Employs a Simulated Annealing schedule. Early in evolution, mutation rates (and specifically the input-pointer bias) are high to encourage global exploration of the topological space. Over successive generations, the mutation rate decays exponentially to promote local exploitation.

### 1.2 Crossover (`crossover.py`)
- **`MEPOnePointCrossover` & `MEPTwoPointCrossover`**: Swaps contiguous blocks of genetic material. 
- **`MEPUniformCrossover`**: Evaluates a crossover mask gene-by-gene.
- **`HomologousPrefixCrossover`**: *A state-of-the-art operator based on structural similarity.* 

#### Mathematics of Homologous Crossover
Instead of blindly selecting a crossover index $k$ which might sever critical node dependencies, `HomologousPrefixCrossover` aligns the genomes by computing an approximation of their Longest Common Subsequence (LCS) or graph edit distance. 

Let parent genomes be represented as sequences of nodes $A = [a_1, a_2, \dots, a_n]$ and $B = [b_1, b_2, \dots, b_n]$. A similarity matrix $S$ is computed where:

$$ S_{i, j} = \begin{cases} 1 & \text{if } a_i.\text{op} == b_j.\text{op} \text{ and arguments match structurally} \\ 0 & \text{otherwise} \end{cases} $$

The crossover index $k$ is selected stochastically from the set of indices where structural alignment is maximized. By swapping material only at structurally homologous boundaries, the operator guarantees **Node Integrity**, dramatically reducing the destructiveness typically associated with crossover in acyclic graph representations.

---

## 2. Cartesian Operators (Pure dCGP)

Our implementation for `CartesianGenome` currently provides baseline operations.
- **`CartesianMicroMutation`**: Structurally modifies the Cartesian grid's nodes or final `out_nodes` routing, constrained strictly by the graph's acyclic boundaries (`levels_back`).

- **`SubgraphCrossover`**: Resolves the historical problem of crossover destructiveness in CGP. Instead of blindly swapping nodes (which shatters active paths and creates positional bias), this operator dynamically traces the *phenotypically active* computational subgraph via a reverse `jax.lax.scan`. Recombination is then preferentially restricted to nodes that contribute to the active path of at least one parent, perfectly adhering to the subgraph mechanisms advocated by Kalkreuth & Kocherovsky (2025).

---

## 3. Hybrid / Neural Operators (dCGPANN / dMEP)

For architectures that fuse discrete topology (ops/args) with continuous parameters (weights/biases), uniform variation is impossible. We utilize a bifurcated approach (`neural_mutation.py`).

### 3.1 `ArchitectureMutation` (Discrete)
Governs the structural evolution of the program topology.
- **Operator Rate (`op_rate`)**: Probability of swapping the activation function or mathematical operator.
- **Argument Rate (`arg_rate`)**: Probability of altering the internal wiring of the nodes.

### 3.2 `WeightMutation` (Continuous)
Explores the continuous parameter space independently of SGD.
- Applies standard Gaussian noise $\mathcal{N}(0, \sigma)$ to the neural weight matrices. This prevents gradient stagnation when the inner Lamarckian optimizer (`LSMFEvaluator`) falls into local minima.

### 3.3 `HybridMutation`
A meta-operator that acts as a sequential composition of both structural and continuous variation. It executes the structural `ArchitectureMutation`, repairs broken argument pointers using an `autocorrect()` pass, and subsequently applies the continuous `WeightMutation`.

### 3.4 Continuous Crossover Mechanisms
To achieve true Lamarckian variation, structural topology crossovers must be mathematically composed with continuous weight interpolation.
- **`ContinuousBlendCrossover`**: Applies standard uniform blend crossover to the continuous parameter arrays (`weights`, `biases`), generating a scalar $\alpha \sim U(-0.5, 1.5)$ to interpolate between the parent matrices.
- **`HybridNeuralCrossover`**: A foundational meta-operator for dCGPANN / dMEP. It accepts both a discrete topology policy (e.g., `SubgraphCrossover`) and a continuous weight policy (e.g., `ContinuousBlendCrossover`). Utilizing `jax.tree_map`, it seamlessly simultaneously splices the discrete computational arrays and blends the continuous structural arrays, solving the open problem of homologous continuous crossover.

---

## 4. Orchestration: The Lamarckian Emitter (`LSPMOEmitter`)

The Emitter orchestrates the generational lifecycle. For our differentiable architectures, it operates as a **Lamarckian Write-Back pipeline**.

During evaluation, `LSMFEvaluator` performs local SGD, shifting the continuous weight matrices to minimize loss. When `LSPMOEmitter` performs non-dominated sorting (NSGA-II) and selects elites, it physically extracts the **post-SGD** weights from the evaluator's state and writes them back into the population graph. Thus, when `HybridMutation` is applied to create the offspring, the offspring inherit the *learned* traits of their parents, dramatically accelerating the convergence of the genetic algorithm.

---

## 5. Theoretical Grounding: CGP Best Practices

The design of the above operators is heavily influenced by recent paradigm shifts in the Genetic Programming literature, most notably synthesized by Kalkreuth & Kocherovsky (2025).

### Overcoming Historical Fallacies
Historically, crossover was deemed inherently destructive for CGP. The literature (dating back to Clegg et al., 2007) relied almost exclusively on mutation-only $(1+\lambda)$-ES algorithms. However, Kalkreuth & Kocherovsky demonstrate that crossover is only destructive when it shatters the integrity of computational nodes. 

### Enforcing Node Integrity & Homology
Our implementation of `HomologousPrefixCrossover` is a direct response to this finding. By explicitly taking into account the structure of the DAG and ensuring that nodes (and their immediate dependencies) remain intact as atomic units during recombination, we preserve beneficial phenotypic patterns. 

### Avoiding Positional Bias
Standard uniform mutations suffer from Positional Bias (or length bias)—nodes near the inputs have a disproportionate probability of remaining active in the final graph trace. Our utilization of `AnnealedTopologicalMutation` and the proposed structural distance metrics are active measures designed to mitigate this structural bias and ensure a more uniform exploration of the phenotypic space.

**References:**
> 1. Oltean, M. (2001). *Multi Expression Programming*. (Origin of MEP Operators)
> 2. Miller, J. F. (2011). *Cartesian Genetic Programming*. (Origin of CGP Operators)
> 3. Francone, F. D., et al. (1999). *Sticky Crossover*. (Origin of homologous alignment in GP)
> 4. Kalkreuth, R., & Kocherovsky, M. (2025). *Cartesian Genetic Programming: Best Practices and Future Directions*. (Best practices, Node Integrity, and Subgraph mechanisms)
