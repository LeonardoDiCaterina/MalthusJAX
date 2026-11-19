# Develop a local extension

Use the [plugin architecture guide](framework_guides/plugin_architecture_guide.md)
or [corridor example](research_workflows.md) to register a fitness or operator
and refer to its registered name in Python or TOML.

Composer can execute Python files under the current working directory's
`plugins/` directory at import time. Run it from a trusted project directory.
Discovery uses module stems as import names. Choose unique stems and avoid
importing a discovered plugin again under a different name.

From a checkout, `python scripts/scaffold.py --help` lists the available
scaffolds. The operator catalog resolves operators, the native engine
registry constructs engines, and the backend registry connects Composer
to providers.
