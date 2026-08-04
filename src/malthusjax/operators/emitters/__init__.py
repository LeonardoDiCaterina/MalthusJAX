from .base import BaseEmitter, EmitterState


def _register_emitters() -> None:
    from malthusjax.composer._registry import register_table
    from malthusjax.operators.emitters.catalog_factories import (
        build_genetic_mixing_emitter,
        build_qdax_native_emitter,
        build_qdax_replica_emitter,
    )

    register_table(
        [
            ("qdax_replica", build_qdax_replica_emitter, {}),
            ("qdax_native", build_qdax_native_emitter, {}),
            ("genetic_mixing", build_genetic_mixing_emitter, {}),
        ]
    )


_register_emitters()

__all__ = ["BaseEmitter", "EmitterState"]
