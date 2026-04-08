import pytest

from malthusjax.composer.genome_catalog import GenomeCatalog
from malthusjax.core.genome.binary_genome import BinaryGenomeConfig
from malthusjax.core.genome.real_genome import RealGenomeConfig


def test_parse_spec_simple():
    catalog = GenomeCatalog()

    name, params = catalog.parse_spec("real:dim=5,bounds=(-1,1)")

    assert name == "real"
    assert params["dim"] == 5
    assert params["bounds"] == "(-1,1)"


def test_parse_spec_handles_booleans_and_numbers():
    catalog = GenomeCatalog()

    name, params = catalog.parse_spec("binary:length=8,flag=True,rate=0.25")

    assert name == "binary"
    assert params == {"length": 8, "flag": True, "rate": 0.25}


def test_parse_spec_skips_invalid_pairs():
    catalog = GenomeCatalog()

    name, params = catalog.parse_spec("binary:length=4,invalid_pair")

    assert name == "binary"
    assert params == {"length": 4}


def test_parse_spec_empty_raises_value_error():
    catalog = GenomeCatalog()

    with pytest.raises(ValueError, match="Empty genome specification"):
        catalog.parse_spec("")


def test_get_real_and_binary_configs():
    catalog = GenomeCatalog()

    real_config = catalog.get("real:dim=4,bounds=(-2.0,2.0)")
    assert isinstance(real_config, RealGenomeConfig)
    assert real_config.shape == (4,)
    assert real_config.bounds == (-2.0, 2.0)

    binary_config = catalog.get("binary:length=6")
    assert isinstance(binary_config, BinaryGenomeConfig)
    assert binary_config.shape == (6,)


def test_get_unknown_genome_raises_key_error():
    catalog = GenomeCatalog()

    with pytest.raises(KeyError, match="unknown"):
        catalog.get("unknown")
