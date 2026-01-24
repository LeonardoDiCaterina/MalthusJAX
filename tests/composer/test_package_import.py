def test_import_composer_package():
    import malthusjax.composer as composer
    assert hasattr(composer, "Composer")
    assert composer.Composer is not None
