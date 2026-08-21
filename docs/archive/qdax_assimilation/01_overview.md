# QDax Assimilation: Architectural Overview

This folder contains the formal design and implementation strategy for assimilating the **QDax** library into **MalthusJAX**.

## Objective
To enable MalthusJAX to natively support Quality-Diversity (QD) algorithms (like MAP-Elites) by leveraging QDax's hardware-accelerated components. The goal is to provide MalthusJAX's declarative (TOML) interface, unified CLI (`mjax`), and statistical rigorousness, while delegating the complex QD grid mechanics to QDax.

## Phased Assimilation Strategy
The assimilation is divided into two distinct technical phases, allowing us to ship value quickly while migrating towards a fully native integration:

1. **Phase 1: Level 1 Integration (The Black-Box Adapter)**
   - **Concept**: Wrap the entire QDax `jax.lax.scan` loop into a MalthusJAX `GeneticEngine` adapter.
   - **Pros**: Immediate compatibility. Zero performance penalty. Proved to work out of the box with JAX PyTrees.
   - **Cons**: MalthusJAX cannot hook into the `ask`/`tell` lifecycle mid-execution. Cannot be used with external/offline physics evaluators.

2. **Phase 2: Level 2 Integration (Native Orchestration)**
   - **Concept**: Build a native `QDEngine` inside MalthusJAX. MalthusJAX controls the `ask()` and `tell()` loop, but delegates the actual genotype generation to QDax Emitters, and archiving to QDax Repertoires.
   - **Pros**: True assimilation. Unlocks MalthusJAX's modularity (hybrid operators). Supports external evaluation loops.
   - **Cons**: Requires tighter coupling and careful translation of PyTree states between the two frameworks.

## Documents in this Specification
- `02_base_interfaces.md`: Required extensions to MalthusJAX's core objects.
- `03_level_1_scan.md`: Implementation guide for the Phase 1 Adapter.
- `04_level_2_orchestration.md`: Implementation guide for the Phase 2 QDEngine.
- `05_implementation_checklist.md`: Step-by-step TODO list to execute this assimilation.
