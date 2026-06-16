# Chapter 4: Methodology and Experimental Design

## 4.1 Introduction
The empirical validation of evolutionary computation frameworks poses unique methodological challenges. Because evolutionary algorithms are inherently stochastic, and because modern hardware-accelerated frameworks like MalthusJAX utilize Just-In-Time (JIT) compilation, traditional simplistic mean aggregations are insufficient for rigorous scientific claims.

The overarching goal of the experimental design presented in this chapter is to systematically isolate and quantify the impact of MalthusJAX’s modular architecture. To ensure all empirical claims regarding convergence parity, algorithmic speedups, and computational overhead are statistically sound, the framework implements a strict benchmarking and hypothesis-testing infrastructure. This chapter details the methodological strategy across three distinct phases: Baseline Control (Parity), Structural Dissection (Ablation), and Domain Generality (Representation).

## 4.2 Benchmarking Infrastructure: Isolating Compilation vs Execution
JAX leverages the XLA (Accelerated Linear Algebra) compiler to fuse high-level Python logic into optimized bare-metal GPU machine code. The first time a JAX function is called with a specific input shape, JAX must trace the function, construct an intermediate computation graph, and compile it. Consequently, the execution time of the *first* generation of any evolutionary run is magnitudes slower than subsequent generations.

To measure purely algorithmic scaling without JIT contamination, the benchmarking infrastructure enforces a strict temporal separation. A standalone tracing "warmup" run is executed prior to the actual benchmark to ensure the XLA cache is fully populated. The actual multi-seed evolutionary loop is timed purely on its execution speed, perfectly isolating algorithmic overhead from the JAX compilation tax.

## 4.3 The Baseline Control: Pragmatic Equivalence
To scientifically evaluate MalthusJAX against established baselines, it is necessary to eliminate "mathematical noise." If MalthusJAX converges faster, it must be proven that the speedup is derived from the software architecture rather than unintended deviations in the underlying mathematical operators.

### 4.3.1 Pipeline Construction
Two primary pipelines were constructed to establish the baseline control:
1. **The Native EvoSAX Baseline (`pipeline_evosax_ga`)**: A completely independent execution utilizing the `evosax` SimpleGA algorithm. This pipeline runs entirely on EvoSAX's native logic and utilizes EvoSAX's native fitness function without *any* MalthusJAX interference or architectural overhead.
2. **The MalthusJAX Parity Control (`pipeline_malthusjax_ga`)**: The MalthusJAX engine, configured specifically to wrap `evosax`'s exact selection, crossover, and mutation functions using the `malthusjax` adapter pattern.

By forcing MalthusJAX to execute the exact same underlying mathematical operators as the native EvoSAX baseline, we created a perfect control. Because no native MalthusJAX operator is interfering with the algorithmic sequence, and because both pipelines evaluate identical fitness landscapes, any discrepancies in execution time can be wholly and safely attributed to the experimental variable: MalthusJAX's modular architectural configuration and XLA graph compilation overhead. This establishes the strict foundation of "Pragmatic Equivalence."

### 4.3.2 Software Guardrails and Fairness Guarantees
To guarantee absolute fairness and scientific validity in the parity controls and ablation sweeps, significant software engineering guardrails were enforced. These guardrails prevent silent algorithmic drift and ensure that MalthusJAX cannot gain an artificial performance advantage over the baselines:

- **The Closed-Loop Adapter Design (Fitness Fairness):** The `EvosaxEngineAdapter` (architected in **Section 3.8.1**) acts as a strict closed-loop system during benchmarking. Rather than MalthusJAX merely wrapping the candidate generation functions and handling the evaluations natively, the adapter intentionally relies on EvoSAX's built-in fitness modules to evaluate candidates. This guarantees that both pipelines compute the exact same fitness function down to the precise floating-point operations, preventing any hidden computational shortcuts.
- **Deterministic Seed Alignment (Initialization Fairness):** The benchmarking suite enforces strict PRNG topography locks. MalthusJAX seeds its engine using the exact same random key vector passed to the independent EvoSAX run. This guarantees mathematically identical initial population starting points. By ensuring both algorithms start from the exact same coordinates in the search space, any subsequent divergence is isolated solely to stochastic sampling sequences during mutation and crossover operations, eliminating initialization bias.

#### Operator-Level Mimicry (The Ablation Wrappers)
To properly isolate the computational overhead of MalthusJAX's modular architecture during the ablation suite, we implemented specialized "mimic" wrappers (e.g., `EvosaxUniformCrossoverWrapper` and `EvosaxDenseGaussianMutationWrapper`, whose exact injection mechanics are detailed in **Section 3.8.2**). 

Instead of relying on EvoSAX's outer algorithms, these wrappers directly subclass MalthusJAX's native operators. However, the standard MalthusJAX arithmetic and PRNG logic inside them are intentionally discarded. In their place, the exact mathematical functions from the `evosax` source code are injected straight into MalthusJAX's intermediate execution layers. 

This specific design choice guarantees that both the Native MalthusJAX operators and the Wrapped EvoSAX mimics are orchestrated by the exact same nested `jax.vmap` batching infrastructure. By keeping the outer execution logic identical and only swapping the inner mathematical core, the ablation suite mathematically isolates and quantifies the exact architectural cost of separating stochastic state generation from pure arithmetic logic, rather than simply measuring a superficial wrapper overhead.

## 4.4 Structural Dissection: The Ablation Suite
Having established Pragmatic Equivalence, the next methodological phase is to quantify the computational cost of crossing architectural boundaries (adapters) and to validate MalthusJAX's natively written operators against wrapped baseline operators.

To properly isolate this overhead, the wrappers (e.g., `EvosaxUniformCrossoverWrapper`) were built by subclassing MalthusJAX's native base operators. Inside these wrappers, standard arithmetic and PRNG logic were intentionally discarded, and the exact math function from `evosax` was injected. This ensures that both native and wrapped operators bypass the typical **Tier 1 Arithmetic Kernel (Section 3.4)**, but are still perfectly orchestrated by the exact same **Tier 3** nested `jax.vmap` batching infrastructure.

### 4.4.1 Ablation Pipelines
1. **Selection Ablation:** Native Tournament Selection vs. EvoSAX Wrapped Tournament.
2. **Crossover Ablation:** Native Uniform Crossover vs. EvoSAX Wrapped Uniform.
3. **Mutation Ablation (Dense vs Sparse):** Native Sparse Gaussian Mutation vs. EvoSAX Wrapped Dense Gaussian Mutation.

The Selection and Crossover ablations measure the performance overhead of marshaling state across an adapter boundary. Conversely, the Mutation ablation tests a fundamental difference in framework philosophy. EvoSAX utilizes *dense* mutation (perturbing 100% of a vector), while MalthusJAX philosophically separates the operator from the rate, allowing for *sparse* mutation (e.g., perturbing only 10% of a vector).

## 4.5 The Parameter Grid and Hypothesis Testing

To ensure empirical results are statistically significant and stress-test the XLA compilation paths across multiple hardware utilization profiles, the benchmarking suite is executed across a massive parameter grid.

### 4.5.1 The Benchmark Landscape Topographies
To rigorously test the system against varying mathematical geometries, the framework sweeps across two distinct tiers of Black-Box Optimization Benchmarking (BBOB) functions:

**1. The Standard Suite (Parity and Baseline Scaling)**
- **Sphere:** A purely separable, convex baseline to isolate framework overhead devoid of mathematical complexity.
- **Rosenbrock:** A classic ill-conditioned landscape featuring a narrow, flat valley that strains gradient-free pathing.
- **Rastrigin:** A highly symmetric multimodal landscape designed to test basic exploration/exploitation balances.

**2. The Hard Suite (Extreme Precision and Deception)**
To conclusively validate MalthusJAX under extreme computational duress, the ablation and precision parity hypotheses (H1, H2) were further evaluated against:
- **Lunacek (bi-Rastrigin):** A highly deceptive double-funnel structure. The global minimum is hidden within a narrow funnel, while a massive local minimum funnel naturally traps greedy selection models.
- **Schwefel:** A highly asymmetrical multimodal space where the global minimum is located at the extreme bounds of the geometric space, brutally testing mutation clamping logic.
- **Gallagher's Gaussian 21-hi Peaks:** A rugged landscape completely devoid of underlying global structure. It consists of 21 randomly positioned Gaussian peaks with massive condition numbers, designed to push floating-point precision ($float16$ vs $float32$) to the brink of numerical underflow.

### 4.5.2 The Dimensionality and Population Sweep
- **Dimensionality Sweep ($D \in \{10, 50, 100, 500\}$):** Low dimensionality tests baseline capability, while high dimensionality ($D=500$) strains operator scalability and forces the XLA compiler to handle massive matrix multiplications.
- **Population Scaling ($P \in \{64, 256, 1024, 4096, 16384\}$):** An exponentially growing sequence designed to transition the GPU workload from "compute-bound" (dominated by graph overhead at $P=64$) to "memory-bound" (saturating parallel cores at $P=16384$).

### 4.5.2 Statistical Rigor
Every coordinate pair in the $D \times P$ grid was evaluated across **100 independently seeded PRNG runs**. This massive sample size enables rigorous, hypothesis-driven statistical testing:
- **Two One-Sided Tests (TOST) for Equivalence:** Used in the Parity suite to statistically prove that MalthusJAX produces mathematically identical convergence profiles to independent baseline alternatives within a strict, pre-defined equivalence margin.
- **Wilcoxon Signed-Rank Test:** The primary non-parametric test used to evaluate location shifts in non-normally distributed evolutionary convergence profiles.
- **Cohen's $d_z$:** Quantifies the standardized mean difference (effect size) between paired runs, revealing the magnitude of algorithmic divergence beyond simple p-values.

### 4.5.3 Latin Hypercube Sampling (LHS)

> [!NOTE]
> **Status**: The full 3D Latin Hypercube Sampling parameter grid (Generations, Population Size, Dimensions) has been fully coded in `generate_lhs_configs.py`. The suite spans 270 independent LHS pipeline configurations and is currently executing 27,000 trace sequences unattended on the DAH2 cluster.

While Section 4.5.1 defined the discrete Cartesian grid required to perform the strict, seed-paired TOST equivalence tests, proving the systemic architectural scaling hypotheses required sweeping across highly multi-dimensional continuous configuration spaces (e.g., varying Dimensionality, Population Size, Generation count, and operator configurations simultaneously). An exhaustive full-factorial Cartesian grid across these dimensions would result in a combinatorial explosion, rendering the computational requirements intractable even on modern GPU clusters.

Formally, to draw $N$ samples across $K$ dimensions, the cumulative distribution function (CDF) of each parameter is divided into $N$ equiprobable intervals of size $1/N$. The $i$-th sample for the $j$-th parameter, $x_{ij}$, is generated via the inverse cumulative distribution function $F_j^{-1}$:

$$ x_{ij} = F_j^{-1}\left( \frac{\pi_j(i) - u_{ij}}{N} \right) $$

Where:
- $N$ is the total number of configurations to generate.
- $\pi_j$ is an independent, uniformly random permutation of the integer sequence $\{1, 2, \dots, N\}$ for dimension $j$.
- $u_{ij} \sim U(0, 1)$ is a continuous uniform random variable providing jitter within the stratum.

Because $\pi_j$ is a strict permutation, exactly one sample is drawn from each of the $N$ strata for every dimension. For variables requiring logarithmic scaling (such as the Population Size $P$), the framework maps the LHS output via a log-uniform transformation:

$$ P_{ij} = 10^{\, \text{LHS}_{ij} \cdot (\log_{10}(P_{max}) - \log_{10}(P_{min})) + \log_{10}(P_{min})} $$

**Practical Implications (Search Space Reduction):** 
In the context of the MalthusJAX benchmarking suite, deploying LHS provides a critical quantification of efficiency. The experimental design sweeps across a 3-dimensional continuous integer space: Dimensionality $D \in [2, 100]$, Population Size $P \in [10, 1000]$, and Generations $G \in [10, 1000]$.

If we were to construct a relatively coarse Cartesian grid by taking just 10 discrete intervals for each dimension, the grid would yield $10 \times 10 \times 10 = 1,000$ distinct coordinate points. Evaluating these across the 3 Hypotheses and 3 Benchmark functions (Sphere, Rosenbrock, Rastrigin) would require $9,000$ configuration pipelines. At $100$ random seeds per pipeline, this equates to **$900,000$ independent evolutionary runs**.

By deploying LHS, the framework extracts $N=30$ optimally stratified points across the 3D volume. This yields just $30 \times 3 \times 3 = 270$ configurations, totaling **$27,000$ independent runs**. This represents a mathematically rigorous **97.0% reduction** in the computational search space. Rather than generating biased scaling laws derived from an exhaustive grid that would cripple standard hardware, the LHS-generated coordinates guarantee uniform sampling across the entire operational manifold. These non-collapsing points are then fed into the OLS regression engine, interpolating the overarching architectural scaling laws with maximal statistical efficiency and a fraction of the carbon footprint.

## 4.6 OLS Regression Modeling and Diagnostics
To move beyond isolated paired comparisons and understand the systemic scaling behavior of MalthusJAX, the benchmarking infrastructure incorporates a comprehensive Ordinary Least Squares (OLS) regression engine.

### 4.6.1 The Log-Log Interaction Model
The primary analytical model employed to determine computational scaling is formulated as a log-log interaction regression:

$$ \ln(Y_i) = \beta_0 + \beta_1 I_{\text{mjx},i} + \beta_2 \ln(D_i) + \beta_3 (I_{\text{mjx},i} \times \ln(D_i)) + \epsilon_i $$

Where:
- $Y_i$ is the dependent metric of interest (e.g., Execution Time in seconds).
- $\beta_0$ represents the y-intercept (the base execution time or intrinsic framework initialization cost when all scaled inputs are mathematically zeroed).
- $\beta_1$ isolates the constant offset or performance penalty introduced purely by the architectural switch ($I_{\text{mjx},i}$).
- $\beta_2$ isolates the fundamental scaling effect of the dimension ($D_i$), independent of the specific framework architecture.
- $I_{\text{mjx},i} \in \{0, 1\}$ is a dummy indicator variable representing the architectural switch ($1$ for Native MalthusJAX, $0$ for Wrapped EvoSAX).
- $D_i$ represents the continuous scaling parameter (Dimensionality or Population).
- $\epsilon_i$ is the stochastic error term, assumed to be identically and independently distributed: $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$.

The critical coefficient in this model is the interaction term, $\beta_3$. By taking the partial derivative with respect to $\ln(D_i)$, the model mathematically formalizes the divergence in scaling. A statistically significant, negative $\beta_3$ coefficient ($H_1: \beta_3 < 0$) formally proves that MalthusJAX scales asymptotically better than the baseline as problem complexity increases. The execution time ratio between the two frameworks can be directly derived as $\exp(\beta_1 + \beta_3 \ln(D_i))$.

### 4.6.2 Statistical Diagnostic Suite
In standard, single-variable regressions, validating Gauss-Markov assumptions is often performed via heuristic visual inspection (e.g., plotting residuals against fitted values or utilizing Q-Q plots). However, the MalthusJAX benchmarking regression model incorporates multiple, simultaneously interacting predictors ($D, P, G, I_{\text{mjx}}$). In highly multi-dimensional regression spaces, projecting residual variance onto a two-dimensional plot is highly deceiving. Overlapping dimensional variance and the density of the Latin Hypercube continuous mapping can visually mask severe heteroskedasticity or non-normality, leading to a false sense of security.

Because visual heuristics fail in higher dimensions, and because the validity of the $\beta_3$ p-values strictly depends on the Gauss-Markov assumptions governing the OLS residuals ($\epsilon_i$), the framework eschews visual inspection entirely. Instead, it automatically subjects the residuals to a rigorous, hypothesis-driven diagnostic suite:

1. **Breusch-Pagan Test (Heteroskedasticity):** Tests the null hypothesis that the variance of the residuals is constant ($H_0: \sigma_i^2 = \sigma^2$). It regresses the squared OLS residuals against the independent variables. If the Lagrange Multiplier statistic ($LM = n R_{\text{aux}}^2$) yields a $p$-value $< 0.05$, heteroskedasticity is present, indicating that execution variance blows up at higher dimensionalities.
2. **Shapiro-Wilk Test (Univariate Normality):** Tests $H_0$ that the residuals are drawn from a normal distribution. The test statistic $W$ compares the variance of the residuals to the optimal variance of a normal distribution.
3. **Mardia's Test (Multivariate Normality):** For models with multiple dependent metrics (e.g., execution time and VRAM usage simultaneously), Mardia's test evaluates the multivariate skewness ($b_{1,p}$) and kurtosis ($b_{2,p}$) of the residual vectors.

If any of these formal diagnostics flag a severe violation ($p < 0.05$), the automated markdown compiler warns the researcher, indicating that the baseline OLS $p$-values may be overly optimistic and that robust standard errors (e.g., Huber-White estimators) must be applied to the interaction model.

This automated, hypothesis-driven statistical pipeline guarantees that all ablation claims and execution speedups published in this thesis are backed by strict mathematical modeling rather than anecdotal observation.
