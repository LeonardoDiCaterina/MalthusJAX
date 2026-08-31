# Cartesian Genetic Programming Analysis

**Source Paper:** Miller, J. F., & Thomson, P. (2000). Cartesian genetic programming. In European Conference on Genetic Programming (pp. 121-132). Springer, Berlin, Heidelberg.

## Overview
Cartesian Genetic Programming (CGP) represents programs as a two-dimensional grid of computational nodes. The representation provides an elegant way to enforce feed-forward structure by restricting how nodes can connect. Unlike tree-based GP, CGP allows multiple outputs and inherently supports subgraph reuse.

## Core Characteristics

### 1. Representation & Topology
- **Grid Layout:** Nodes are arranged in an $n_r \times n_c$ grid (rows $\times$ columns).
- **Inputs & Outputs:** The genome has $n_i$ inputs and $n_o$ outputs.
- **Node Connectivity (`levels_back`):** A critical parameter $l$ (or `levels_back`) dictates connection limits. A node in column $j$ can only take inputs from nodes in columns $k$ where $j - l \le k < j$. If $l$ is unconstrained (e.g., $l=n_c$), a node can connect to any previous node or primary input.
- **1-D vs 2-D Variant:** The authors note that a 1-D sequence ($n_r=1$) with unconstrained connectivity ($l=n_c$) is sufficient for most problems and simplifies implementation. In this variant, the column index is equivalent to the node's position in a linear list.

### 2. Node Evaluation
- Each node evaluates a function chosen from a primitive set $\mathcal{F}$.
- Node execution is completely feed-forward and deterministic.
- Unlike Multi Expression Programming (MEP) which uses symbiotic selection (choosing the best node post-evaluation), CGP uses designated output pointers to select the final phenotype.

### 3. Evolutionary Dynamics
- **Neutrality:** A defining feature of CGP. The genotype size is fixed, but the phenotype (the active graph) can be smaller. Many nodes in the genome may not be connected to the output nodes. Mutations in these inactive nodes have no effect on fitness. This "neutral bloat" allows the search process to traverse neutral networks, a mechanism the authors argue helps avoid local optima.
- **Evolutionary Strategy:** Standard CGP employs a $1 + \lambda$ Evolutionary Strategy, usually with $\lambda = 4$.
- **Crossover:** Due to the 1+4 ES approach and the positional dependence of the graph, crossover is typically omitted. The authors rely entirely on point mutations.
- **Mutation Rate:** Point mutations alter node opcodes, arguments, or output pointers. To preserve neutral drift without destroying the active graph, mutation rates are kept low (typically 1-5% of genes, targeting ~3 mutated genes per chromosome on average).

## Implementation in MalthusJAX

We mapped the CGP specification to the MalthusJAX vectorized architecture:

### 1. `CartesianGenome`
- Implemented as a batched PyTree structure.
- **Topological Integrity:** During mutation, we use `jnp.clip(new_args, 0, hi)` to guarantee that no node attempts to connect to a node positioned after it, or outside its `levels_back` window. This makes the genome XLA-friendly, avoiding runtime validation loops.

### 2. `CartesianMicroMutation` & `NoOpCrossover`
- **Mutation:** Applies uniform point mutations across opcodes, arguments, and output nodes, exactly as Miller described.
- **Crossover:** A dummy `NoOpCrossover` operator fulfills the `GeneticEngine` API requirements while acting as an identity pass-through.

### 3. `CartesianGPEvaluator`
- Evaluates the genome using `jax.lax.scan` to process nodes sequentially.
- Extracts the final prediction by gathering the values specified by `genome.out_nodes`.

### 4. Native Engine Alignment
- We mapped the 1+4 Evolutionary Strategy onto MalthusJAX's `GeneticEngine` by using `ElitePoolSelection` with `elite_k=1`. This enforces that the single best individual is always selected as the parent, and the offspring are purely its mutated clones.

## Experimental Validation

We ran the classic quartic polynomial symbolic regression problem ($y = x^4 + x^3 + x^2 + x$) to validate the implementation.

**Observations:**
- **Slower Convergence:** Unlike MEP, which uses crossover and large populations to converge in ~50 generations, CGP required 5,000+ generations. This aligns with Miller's findings that neutral drift via point mutations takes significant time.
- **Length Tolerance:** We verified that longer chromosomes (up to $n_c=100$) can still find optimal solutions provided the mutation rate scales dynamically (e.g. $\sim 3$ genes per chromosome) to avoid excessive disruption of the active graph.

## Next Steps: Quality-Diversity (QD)
While the current implementation uses the classic 1+4 ES approach, CGP's inherent neutrality makes it an excellent candidate for Quality-Diversity algorithms like MAP-Elites. In QD, we would use the active node count or topological depth as a behavioral descriptor, maintaining a diverse repertoire of active graphs across the search space.
