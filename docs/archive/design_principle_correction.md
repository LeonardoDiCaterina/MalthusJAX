# MalthusJAX v2.0: The Operator-Genome Design Principle (Corrected)

> Context: this corrects a misreading in `architecture_audit.md`. Read this **before** acting on that audit. It does not replace the audit's technical findings about the array-genome family (Section 3, CP-1 through CP-8 for Real/Binary/Categorical/Series) — those stand. It replaces the **framing** the audit built on top of them.

---

## 1. The error in the original audit

The audit assumed there must be **one** generic operator mechanism that every genome representation — present and future — has to fit into. Under that assumption, `LinearGenome`'s `(ops, args)` structure looked like a crisis: a representation the architecture "cannot scale to," evidence the whole design was broken, a case that had to be solved *now* or worked around with a compatibility shim.

That assumption is wrong. It is not part of the actual goal for this project.

## 2. The actual principle

MalthusJAX is a **general-purpose evolutionary computation library**: it must support many genome representations, used by many people, for many kinds of problems. That goal does not imply a single universal operator mechanism. It implies the opposite: different representations have genuinely different structure, and forcing them through one mechanism is what creates artificial pressure (like the "three-way trade-off" in the audit's executive summary).

The correct model:

- There can be **multiple generic operator mechanisms**, each scoped to a **family of representations that are actually structurally/semantically similar to each other**.
- A family deserves a shared generic mechanism when its members genuinely have the same shape of problem — not because they're all "genomes" in name.
- A representation that doesn't fit any existing family's mechanism is not a bug or an exception to patch around. It's a sign that either (a) it belongs to a family that doesn't have a generic mechanism *yet*, or (b) it's bespoke and gets hand-written operators, which is a perfectly acceptable permanent state.
- There is **no requirement that every family's generic mechanism look like every other family's**. The flat-array family might use schema-tagged pytree leaves; a future program/graph family (`LinearGenome` and similar) might need something structurally different — e.g. mechanisms that understand multi-array structure, ordering/topology, or variable-length encodings. These are different problems and are allowed to have different solutions.
- The boundary between families is **not a clean, fixed taxonomy decided up front**. It's case-by-case, judged by whether representations actually share structure — not by surface-level similarity (e.g. "they're all arrays of numbers" is not sufficient if their semantics differ).

## 3. What this means concretely

- **The "three-way trade-off" diagram in the audit's §1 is a false dilemma.** It only holds if you insist on one mechanism for all representations. Once you scope generic mechanisms per-family, Semantic Richness, JAX Purity, and Operator Generality stop fighting each other *within a family* — because the family was chosen precisely so those three things are compatible for its members.

- **`LinearGenome` is not a v2.0 blocker.** It is correctly described as having zero native operators today, but that's not evidence of architectural failure — it's just an unfinished family. It should eventually get its own generic operator mechanism, designed for its actual structure (two arrays with different roles, `ops` and `args`), not retrofitted into the array-family's `.values`-based mechanism. The audit's Open Question #2 ("is LinearGenome support a v2.0 requirement?") is answered: **no**, and it shouldn't be framed as a deferred requirement at all — it's simply a separate, later piece of work with its own design.

- **The audit's CP-1 through CP-8 analysis is still valid and useful — but only as a description of the flat-array family** (Real, Binary, Categorical, Series). These four representations genuinely share structure: single (or fixed-shape) array leaves, similar noise-shaping needs, similar bounds/dtype semantics. For *this* family, Direction A (semantic schemas) and Direction B (protocol typing) in the audit's §8 are the right kind of solution space, and the hybrid synthesis in §9 (decouple from `.values`, move domain logic to genome-level `autocorrect()`, derive noise shape from actual array shape, simplify the generics) is reasonable — **scoped to this family only**.

- **CP-6 (the triple generic `[G, C, P]`) and CP-7/CP-8 (`.values` reach-through in `cross_single_pair` and the engine)** are real boilerplate/coupling problems within the array family and should be fixed as described.

- **The coupling-point matrix in §4** should be read as "how the array-family mechanism needs to evolve to cleanly cover Real/Binary/Categorical/Series" — not as "how every current and hypothetical genome must fit one mechanism." Drop the ❌ marks for `LinearGenome` as alarms; they're just correctly showing it's a different family.

## 4. What changes about the v2.0 scope

- v2.0 work should focus on: making the **array-family** generic mechanism solid (the audit's §9 plan), without trying to make it also absorb `LinearGenome` or any other structurally different representation.
- Designing a generic mechanism for program/graph-like genomes (`LinearGenome` and future similar types) is a **separate, later design effort**, not a requirement gating v2.0, and not something to solve by stretching the array-family abstraction.
- When future representations are proposed, the design question is always: *"does this share real structure with an existing family, or does it need its own?"* — not *"how do we make the existing generic mechanism flexible enough to also cover this."*

## 5. One-line summary to give the agent

> Don't design one generic operator system for all genomes. Design one generic system per family of structurally-similar genomes, and let bespoke or not-yet-generalized representations exist without being treated as architectural failures.
