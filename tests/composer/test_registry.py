from malthusjax.composer.registry import Registry


def test_registry_register_get():
    reg = Registry()

    def dummy(key, params, inputs=None):
        return {"params": params, "inputs": inputs}

    reg.register("dummy", dummy)
    assert "dummy" in reg.list()
    res = reg.get("dummy")(None, {"a": 1}, None)
    assert res["params"]["a"] == 1
