def test_import_composer_package():
    import malthusjax.composer as composer
    assert hasattr(composer, "Registry")
    assert hasattr(composer, "Node")
    assert hasattr(composer, "Pipeline")
    assert hasattr(composer, "load_config")
    assert hasattr(composer, "Composer")
