# Differentiable Genetic Programming (dCGP) Analysis

**Source Paper:** Izzo, D., Biscani, F., & Mereta, A. (2016). Differentiable Genetic Programming. arXiv preprint arXiv:1611.04766.

## Overview
Differentiable Cartesian Genetic Programming (dCGP) extends Cartesian Genetic Programming (CGP) by incorporating high-order automatic differentiation. It evaluates programs using the algebra of **truncated Taylor polynomials** (generalized dual numbers). This means that a single forward pass of the encoded program yields not only the output but also its exact derivatives (gradients, Hessians, and higher-order derivatives) with respect to inputs, ephemeral constants, or edge weights.

## Core Characteristics

### 1. Truncated Taylor Polynomial Algebra (AuDi)
- Traditional GP evaluates a tree/graph using floating-point arithmetic.
- dCGP evaluates the graph using an overloaded data type representing truncated Taylor polynomials.
- This allows computing high-order exact derivatives of the program analytically in a single forward pass.
- A C++ library called `AuDi` was developed by the authors to handle the complex underlying algebra efficiently.

### 2. Weighted dCGP vs Ephemeral Constants
The paper presents two paradigms for learning constants in symbolic regression:
- **Ephemeral Constants Approach:** A few additional input terminals containing real constants are added. The Taylor expansion is computed with respect to these constants. The error is minimized using a Newton step $c_{i+1} = c_i - H^{-1} \nabla \epsilon$. If the Newton step fails, gradient descent is used. Lamarckian evolution applies the locally optimized constants back to the genome.
- **Weighted dCGP Approach:** Edges (connections $C_{ij}$) in the CGP grid are assigned floating-point weights $w_{i,j}$, similar to Neural Networks. The Taylor expansion is computed with respect to a random batch of weights, and a Newton step updates the weights. This makes the method highly general, as evolution can assemble arbitrary constants implicitly via edge weights.

### 3. Solving Differential Equations
- dCGP can be directly applied to search for analytical solutions to Ordinary and Partial Differential Equations (ODEs and PDEs).
- **Error Function:** Because dCGP can output analytical derivatives of the candidate solution $S(\mathbf{x})$, the fitness can be directly evaluated as the residual of the differential equation $f(\partial^\alpha S, \mathbf{x}) = 0$ plus boundary condition violations.
- Compared to prior grammatical evolution techniques (which used stacked nodes for differentiation), dCGP natively and exactly computes derivatives (even mixed derivatives for PDEs).

### 4. Discovery of Prime Integrals
- The paper demonstrates using dCGP to find conserved quantities (prime integrals) of dynamical systems, such as energy or angular momentum.
- The fitness is the derivative of the candidate function with respect to time $\frac{dP}{dt}$, expanded via the chain rule using the system's known equations of motion $\frac{dx_i}{dt}$.
- **Mutation Suppression:** To prevent the GP from trivially converging to $P(x) = C$ (where $\frac{dP}{dt} = 0$ trivially), mutants are rejected if their partial derivatives $\frac{\partial P}{\partial x_i}$ are strictly zero.

## Software Implementation (dcgp / dcgpy)
In a follow-up 2020 paper ("dcgp: Differentiable Cartesian Genetic Programming made easy"), the authors introduced their open-source C++ and Python framework for dCGP.
- They formalized specific classes like `expression_weighted` (adds continuous weights to connections) and `expression_ann` (adds biases, effectively evolving Artificial Neural Networks).
- They highlight how standard CGP mutations can be cleanly applied to these neural graphs by targeting only specific genes (e.g. only mutating kernels, or only connections).
- They emphasize the need for three levels of parallelization: evaluating batches of data points, vectorizing the Taylor algebra, and evaluating the population.

## MalthusJAX Implementation Implications

Integrating dCGP into MalthusJAX is extremely synergistic, because **JAX natively provides arbitrary-order automatic differentiation (`jax.grad`, `jax.hessian`, `jax.jacfwd`, `jax.jvp`)!**
We do not need to implement a complex algebra of truncated Taylor polynomials like `AuDi`. JAX's tracing handles the differentiable pass automatically. Furthermore, JAX's `vmap` natively gives us the exact three levels of parallelization that `dcgp` achieved in C++.

### 1. Native JAX Alignment
- **Weighted CGP:** Our `CartesianGenome` can easily be augmented to include an edge weight matrix `weights` of shape `(num_nodes, max_arity)`.
- **Differentiable Evaluator:** We already have a `DifferentiableMOEvaluator` architecture designed for multi-objective gradients. For dCGP, we can compute the value, gradients, and Hessians using `jax.value_and_grad` directly on the `predict_one` function inside the evaluator.
- **Lamarckian Weight Updates:** We can inject an `Optax` optimizer step (or a manual Newton step if the parameter count is small) directly inside the `evaluate` method. The optimized weights can then be returned and updated in the genome.

### 2. Differentiable Fitness Functions
- To replicate the PDE/ODE solver capabilities, we can create a `DifferentialEquationEvaluator` that takes a target ODE/PDE and computes the fitness as the mean squared residual.
- Thanks to `jax.grad`, computing $\frac{\partial S}{\partial x}$ is trivial inside the JAX `scan` loop that evaluates the DAG.

## Conclusion
The Izzo paper proves that analytical derivatives drastically expand the capabilities of GP. While they required a bespoke C++ Taylor polynomial library in 2016, modern JAX allows us to achieve the exact same differentiable CGP (dCGP) architecture natively and with GPU acceleration.
