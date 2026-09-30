"""Backend providers for MalthusJAX.

Importing this package triggers self-registration of all built-in backends
into the BackendRegistry.  External plugins register via entry points.

Provider modules are imported lazily — see Phase 2 of the refactoring plan.
"""
