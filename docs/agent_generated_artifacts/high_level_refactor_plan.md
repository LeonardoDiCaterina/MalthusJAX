# Overhauling MalthusJAX for JAX Show & Tell

This plan outlines the steps to restructure and polish the MalthusJAX repository. The primary objective is to highlight the project's core strength—a pure JAX state transition engine that compiles an entire evolutionary generation using `lax.scan`—and remove the "cognitive overload" caused by historical research artifacts, overly dense abstractions, and competing narratives. 

## User Review Required

- **Documentation Archiving**: We are moving a large amount of historical documentation and notebooks to an `archive/` folder. Please confirm you are okay with hiding these from the main public presentation.
- **API Change**: Renaming `copy()` to `clone_buffers()` to explicitly reflect buffer ownership semantics.
- **Equinox Migration**: This plan leaves the Equinox migration to Phase 4 (Optional) to ensure we don't block the presentation/cleanup efforts on a major technological refactor.

## Open Questions

- **Hero Benchmark**: Which specific configuration/benchmark should we use for the "Hero Benchmark" script to demonstrate the H100 throughput?
- **Python/JAX Support Matrix**: What are the exact tested versions of JAX, Flax, and CUDA we should specify in `pyproject.toml` instead of loose inequalities?
- **Archive vs Delete**: Should we commit the `archive/` folder, or just delete the historical artifacts entirely from the `main` branch (they will still exist in git history)?

## Proposed Changes

---

### Phase 1: Correctness and Presentation (The "Hero" Story)

Focus on fixing public-facing claims, broken links/commands, and setting up the core narrative for JAX engineers.

#### [MODIFY] README.md
- Update the hook to emphasize the core technical achievement: a JAX-native evolutionary framework built on PyTree state, vectorized operators, preallocated PRNG, and `lax.scan`.
- Fix statistical claims (replace "proving zero loss" with "TOST equivalence tests supported practical equivalence...").
- Fix complexity claims (replace "O(1) GPU compute scaling" with "Runtime increased only modestly over the tested dimensionality range").
- Surface the rigorous testing suite prominently (JIT equivalence, Phase-level correctness, PyTree invariants, etc.).
- Add a "Why JAX?" section explaining the single XLA program compilation and buffer donation.

#### [MODIFY] pyproject.toml
- Update the Python version support matrix (e.g., Python 3.10–3.12).
- Pin or explicitly state tested versions for JAX, Flax, and CUDA if appropriate.
- Remove stale Hatch commands pointing to nonexistent benchmark scripts.

#### [NEW] docs/architecture.md
- A concise guide outlining the JAX execution architecture (Composer -> Engine -> Operators -> Core).

#### [NEW] benchmarks/run_hero_benchmark.py
- Create a single, highly reproducible benchmark script demonstrating the core value proposition: compiling a generational loop into a single XLA program and achieving high throughput. 

---

### Phase 2: Reduce Noise

Remove cognitive overload by hiding development history and experimental branches.

#### [NEW] docs/archive/
- Move all assimilation docs (`docs/*_assimilation/`), audit summaries, and historical challenge reports here.
- Restructure public docs to just `architecture.md`, `performance.md`, `integrations.md`, `extending.md`, and `benchmarking.md`.

#### [NEW] examples/archive/
- Move `examples/_DEMO_LV_*` and other noisy scripts here.
- Keep only a clean `examples/quickstart.py` and structured `examples/showcase/` (e.g., `genetic_algorithm.py`, `island_model.py`, `qd.py`).

---

### Phase 3: Code Architecture

Reduce abstraction density in the most critical paths without rewriting the underlying technology.

#### [MODIFY] src/malthusjax/composer/composer.py
- Refactor this massive ~77KB file. 
- Extract configuration parsing, registries, factories, and adapters into their own smaller modules within `src/malthusjax/composer/`.
- Ensure `Composer` provides a small, clear surface area (`build()`, `run()`, `compare()`).

#### [MODIFY] src/malthusjax/engine/genetic_fastengine.py
- Refactor the main `step()` function so the 5 conceptual phases (entropy, selection, reproduction, merge, evaluation) are visually obvious and read like pseudocode.

#### [MODIFY] src/malthusjax/core/ 
- Rename `copy()` methods to `clone_buffers()` to clarify JAX buffer ownership and donation semantics.
- Document static vs. dynamic state transitions clearly.
- Update `src/malthusjax/core/README.md` to align with API changes (`clone_buffers()`), clarify genome subscriptability defaults, and resolve `UNCONFIRMED` flags.

#### [MODIFY] src/malthusjax/operators/
- **Retain the three-tier operator architecture** (Genome-level, Intermediate/Noise-level, Population-level) as a deliberate optimization feature.
- Document this architecture prominently as a progressive API that allows users to trade abstraction for control without leaving the framework.
- Add clear guidance in the docs: "Start at Genome level. Drop to Population level only when your algorithm is inherently population-wise or you need explicit control over batching/vectorization."
- Add to the Show & Tell narrative: *MalthusJAX doesn't force every evolutionary operator into one batching strategy. It lets the user choose the semantic level at which the operator is expressed, while the framework provides optimized lifting where possible.*

---

### Phase 4: Optional Equinox Prototype

#### [MODIFY] src/malthusjax/core/ (Prototyping)
- After the above steps, investigate replacing Flax with Equinox in the `core` PyTree layer if it substantially simplifies the codebase without performance regressions.

## Verification Plan

### Automated Tests
- Run `pytest` or `make test` (the existing comprehensive test suite covering genome invariants, JIT behavior, engine phases, and PRNG correctness) to ensure no regressions during the `Composer` and `GeneticFastEngine` refactoring.

### Manual Verification
- Run the new `benchmarks/run_hero_benchmark.py` and verify it executes without error.
- Verify `pip install -e .` and all `pyproject.toml` entry points work in a clean environment.
- Read through the refactored `GeneticFastEngine.step()` to ensure it reads cleanly.
- Review the `README.md` to ensure the core narrative aligns with the Show & Tell goals.
