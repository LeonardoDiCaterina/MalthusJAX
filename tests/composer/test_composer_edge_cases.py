from malthusjax.composer.composer import Composer


def test_composer_compare_api():
    """Test the compare() method of Composer."""
    composer = Composer()

    # Define two minimal pipelines
    pipeline1 = dict(
        selection="tournament",
        crossover="blend",
        mutation="gaussian",
    )

    pipeline2 = dict(
        selection="roulette",
        crossover="blend",
        mutation="polynomial",
    )

    # Run the compare method
    results = composer.compare(
        pipelines={"tourn": pipeline1, "roul": pipeline2},
        seeds=3,
        num_parallel=2,
        fitness="sphere:dim=2",
        pop_size=10,
        generations=2,
        genome_length=2,
    )

    # Verify the results object is populated
    assert "tourn" in results.pipelines
    assert "roul" in results.pipelines


def test_composer_backends():
    """Test alternative backends in Composer"""
    composer = Composer()

    # Evosax backend
    composer.quick_run(
        fitness="sphere:dim=2",
        backend="evosax",
        evosax_strategy="SimpleGA",
        pop_size=10,
        generations=1,
    )

    # TensorNEAT backend
    composer.quick_run(
        fitness="xor",
        backend="tensorneat",
        tensorneat_algorithm="NEAT",
        pop_size=10,
        generations=1,
    )


def test_composer_from_toml(tmp_path):
    """Test loading configuration from TOML file"""
    composer = Composer()
    toml_path = tmp_path / "config.toml"
    toml_content = """
[experiment]
fitness = "sphere:dim=2"
pop_size = 10
generations = 1
genome_length = 2

[pipelines.p1]
selection = "tournament"
crossover = "blend"
mutation = "gaussian"
    """
    toml_path.write_text(toml_content)

    results = composer.from_toml(str(toml_path))
    assert "p1" in results.pipelines
