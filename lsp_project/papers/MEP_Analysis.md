# Analysis: Multi Expression Programming (MEP)

**Paper:** *Multi Expression Programming* by Mihai Oltean and D. Dumitrescu (2002)

## 1. Core Philosophy & Motivation
The authors introduce Multi Expression Programming (MEP) to address the inefficiencies of standard tree-based Genetic Programming (GP) and the limitations of single-expression linear variants like Gene Expression Programming (GEP).
MEP's primary motivation is to represent genetic programs in a way that closely mirrors how compilers evaluate mathematical expressions (e.g., three-address code). This makes the representation highly efficient for execution.

## 2. Representation & Strong Implicit Parallelism (SIP)
- **Linear Chromosome**: An MEP chromosome is a linear sequence (array) of instructions.
- **DAG Structure**: The first instruction must be a terminal (e.g., a constant or input variable). Subsequent instructions can be terminals or functions. Function arguments act as pointers to previous instructions, enforcing a valid Directed Acyclic Graph (DAG) and preventing cycles.
- **Strong Implicit Parallelism**: Unlike standard GP or GEP where a chromosome encodes a *single* program, an MEP chromosome encodes *multiple* sub-expressions. Every single instruction in the chromosome represents a valid mathematical expression (the composition of itself and its ancestors).

## 3. Decoding and Fitness Assignment
- **Single-Pass Evaluation**: Because of the DAG structure, the chromosome is parsed top-down in a single pass. Partial results are computed and cached (dynamic programming), making it O(N * NG) complexity where NG is the number of genes.
- **"Best Gene Wins"**: The fitness of each instruction's output is computed independently against the target objective. The fitness of the *entire chromosome* is then assigned as the fitness of its *best-performing instruction*.
- **SEP vs MEP**: The paper proves that Single Expression Programming (SEP)—where only the last instruction's output is evaluated—performs significantly worse than MEP. The search space exploration provided by SIP is crucial.

## 4. Genetic Operators
- **Recombination (Crossover)**:
  - *One-point & Two-point*: Standard contiguous segment swapping.
  - *Uniform*: Randomly takes genes from either parent.
- **Mutation**:
  - *Standard*: Mutates a terminal to another terminal/function, or a function to another function/terminal. Pointers are also randomly reassigned.
  - *Smooth Mutation*: A fine-grained search operator that changes each symbol in a gene with a specific probability to slowly perturb the tree.
- **Exception Handling**: Instead of using protected functions (e.g., protected division), MEP dynamically mutates any instruction that causes an exception (like division by zero) into a terminal symbol. This keeps the population strictly fertile.

## 5. Key Empirical Results
- **Symbolic Regression**: MEP vastly outperformed GEP on quartic polynomials. MEP achieved a 100% success rate with longer chromosomes, whereas GEP's success rate degraded if the chromosome was too long. MEP optimally utilized 30-50 individuals.
- **Tic-Tac-Toe**: MEP evolved an unbeatable strategy (against an all-moves procedure) in just 11 generations using a population of 50.

---

## 6. Relevance & Actionables for `lsp_project`

This paper is the foundational text for the `LinearGenome` and `LinearGPEvaluator` you have already built in MalthusJAX. 

### What You Have Already Achieved:
1. **The Representation**: Your `LinearGenome` strictly follows the MEP representation (opcodes + pointer indices strictly pointing backwards).
2. **The Evaluation**: Your `LinearGPEvaluator.evaluate()` uses `jnp.min(mse_per_tree)`. This is the exact embodiment of MEP's "Best Gene Wins" / Strong Implicit Parallelism.
3. **The Speed**: Because you use JAX's `vmap` and `lax.scan` over the linear array, you have taken the "compiler-like efficiency" Oltean envisioned to its absolute limit on the GPU.

### What Needs to be Implemented (Phase 1 of Roadmap):
Based on this paper, to complete the base MEP implementation, you need to implement the variation operators in `lsp/operators/`:
1. **MEP Crossover**: Implement Uniform and N-point crossover. (Since the structure is a simple array of instructions, standard JAX array slicing/masking will work perfectly).
2. **MEP Mutation**: Implement micro-mutation (randomly changing an `op` or `arg` using `jax.random.choice` while respecting the bounds `0 <= arg < current_row`).
3. **Exception Handling**: Because JAX operates on static arrays without runtime exceptions (e.g., div by zero yields `NaN`), you can either use protected functions (which is easier in JAX) or implement a JAX `jnp.where(jnp.isnan(result), terminal, result)` mask to simulate MEP's exception handling.
