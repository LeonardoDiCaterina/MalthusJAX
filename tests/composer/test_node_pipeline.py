import pytest

from malthusjax.composer.node import Node
from malthusjax.composer.pipeline import Pipeline
from malthusjax.composer.registry import Registry


def test_node_build_invokes_factory():
    registry = Registry()

    def factory(key, params, inputs=None):
        return {"params": params, "inputs": inputs}

    registry.register("dummy", factory)
    node = Node(id="n1", type="dummy", params={"x": 1})

    result = node.build(None, registry)

    assert result["params"] == {"x": 1}
    assert result["inputs"] is None


def test_pipeline_builds_nodes_in_order():
    registry = Registry()

    def source_factory(key, params, inputs=None):
        return {"value": params["value"]}

    def accumulate_factory(key, params, inputs=None):
        source = inputs["source"]
        return {"value": source["value"] + params["increment"]}

    registry.register("source", source_factory)
    registry.register("accumulate", accumulate_factory)

    pipeline = Pipeline(
        name="test",
        nodes=[
            Node(id="source", type="source", params={"value": 2}),
            Node(id="acc", type="accumulate", params={"increment": 3}),
        ],
    )

    outputs = pipeline.build(None, registry)

    assert outputs["source"]["value"] == 2
    assert outputs["acc"]["value"] == 5


def test_pipeline_validate_raises_for_unknown_node_type():
    registry = Registry()
    pipeline = Pipeline(name="broken", nodes=[Node(id="bad", type="missing", params={})])

    with pytest.raises(KeyError, match="missing"):
        pipeline.validate(registry)
