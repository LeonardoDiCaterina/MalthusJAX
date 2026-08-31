# Neural Network Architecture Search with Differentiable CGP

**Source Paper:** Märtens, M., & Izzo, D. (2019). Neural Network Architecture Search with Differentiable Cartesian Genetic Programming for Regression.

## Overview
This paper introduces **dCGPANN**, an encoding that maps Cartesian Genetic Programming to an Artificial Neural Network. It presents a memetic algorithm, "LSMF" (Learn, Select, Mutate, Forget), that combines gradient descent for learning continuous weights with evolutionary algorithms for discovering optimal network topologies (Neural Architecture Search).

## Core Characteristics

### 1. The dCGPANN Encoding
The standard CGP node equation is modified to include trainable weights and biases:
$$N_i = F_i\left(\sum_{j=0}^{a_i} w_{i,j} NC_{i,j} + b_j\right)$$
- **Topological Chromosome ($x_I$):** The standard integer genes controlling which nodes connect to which, and what activation function $F_i$ each node uses (e.g., `tanh`, `sigmoid`, `ReLU`, `ELU`).
- **Continuous Chromosome ($x_R$):** The floating-point weights $w_{i,j}$ and biases $b_j$ for each connection and node.

### 2. The LSMF Algorithm
To evolve these networks, the authors introduce a 4-step memetic algorithm:
1.  **Learn:** For every network in the population, run stochastic gradient descent (SGD) for a set number of epochs (the "cooldown" period). The topology $x_I$ is frozen; only weights $x_R$ are updated.
2.  **Select:** Evaluate the training error of all networks. Keep only the best individual (elitism = 1) and discard the rest.
3.  **Mutate:** Create $N-1$ offspring by mutating the best individual's topology $x_I$ (rewiring connections or changing activation functions). The continuous weights $x_R$ are inherited directly from the parent (Lamarckian inheritance).
4.  **Forget:** After $K$ cycles, the weights $x_R$ are completely randomized while keeping the evolved topology $x_I$ intact. This prevents the search from getting trapped in weight-space local minima.

### 3. Findings
- **Evolutionary Pruning:** The algorithm naturally pruned redundant connections. About 40% of the network was compressed by dropping inactive or duplicate connections.
- **Skip Connections:** The evolution naturally discovered "skip connections" (connections bypassing hidden layers), which are known to mitigate the vanishing gradient problem (similar to ResNets).
- **Activation Functions:** The evolution favored sigmoidal activations in the final layers but learned to aggressively replace `tanh` with other functions depending on the problem.
- **Performance:** The evolved topologies trained faster and reached lower errors than standard feed-forward networks of the same initial size.

## MalthusJAX Implementation Implications

This paper provides the exact blueprint for our `NeuralGenome` implementation!

1.  **Neural Cartesian Genome Structure:** We need to extend our `CartesianGenome` to include `weights` and `biases` arrays. 
2.  **Neural Evaluator:** The evaluator will perform the weighted sum before passing the result to the node's activation function. JAX's `jax.lax.switch` will handle the dynamic activation functions (`ReLU`, `tanh`, etc.).
3.  **LSMF via JAX Optax:** Because our evaluator is written in JAX, the "Learn" step can be implemented by simply running `optax` gradient descent updates on the `weights` and `biases` arrays natively within the JAX compilation boundary during the evaluation phase! 
4.  **Engine Alignment:** The $1 + \lambda$ Evolutionary Strategy we just perfected for CGP using `CGPSelection` is exactly what the "Select" and "Mutate" steps require.

We are perfectly positioned to implement this.
