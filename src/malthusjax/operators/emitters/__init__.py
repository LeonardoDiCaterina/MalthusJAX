from typing import Any

from .base import BaseEmitter, EmitterState


def register_emitters_catalog(register_table_fn: Any) -> None:
    from malthusjax.operators.emitters.catalog_factories import (
        build_genetic_mixing_emitter,
        build_qdax_native_emitter,
        build_qdax_replica_emitter,
        build_tensorneat_crossover_emitter,
        build_tensorneat_emitter,
        build_tensorneat_mutation_emitter,
    )

    register_table_fn(
        [
            ("qdax_replica", build_qdax_replica_emitter, {}),
            ("qdax_native", build_qdax_native_emitter, {}),
            ("genetic_mixing", build_genetic_mixing_emitter, {}),
            ("tensorneat_emitter", build_tensorneat_emitter, {}),
            ("tensorneat_mutation_emitter", build_tensorneat_mutation_emitter, {}),
            ("tensorneat_crossover_emitter", build_tensorneat_crossover_emitter, {}),
        ],
        override=True,
    )


def _register_emitters(register_table_fn: Any = None) -> None:
    if register_table_fn is None:
        from malthusjax.composer._registry import register_table

        register_table_fn = register_table
    register_emitters_catalog(register_table_fn)


__all__ = ["BaseEmitter", "EmitterState"]
