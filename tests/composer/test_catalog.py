import pytest

from malthusjax.composer.catalog import OperatorCatalog
from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.tournament import TournamentSelection


def test_parse_spec_simple():
    """Test parsing simple operator names."""
    catalog = OperatorCatalog()

    op_type, params = catalog.parse_spec("tournament")
    assert op_type == "tournament"
    assert params == {}


def test_parse_spec_with_params():
    """Test parsing operator specs with parameters."""
    catalog = OperatorCatalog()

    op_type, params = catalog.parse_spec("tournament:selections=5,size=3")
    assert op_type == "tournament"
    assert params == {"selections": 5, "size": 3}


def test_parse_spec_type_conversion():
    """Test parameter type conversion."""
    catalog = OperatorCatalog()

    op_type, params = catalog.parse_spec("gaussian:rate=0.1,num_offspring=2,name=test")
    assert op_type == "gaussian"
    assert params == {"rate": 0.1, "num_offspring": 2, "name": "test"}
    assert isinstance(params["rate"], float)
    assert isinstance(params["num_offspring"], int)
    assert isinstance(params["name"], str)


def test_parse_spec_error_cases():
    """Test error handling in parsing."""
    catalog = OperatorCatalog()

    with pytest.raises(ValueError, match="Empty operator specification"):
        catalog.parse_spec("")

    with pytest.raises(ValueError, match="Invalid parameter format"):
        catalog.parse_spec("tournament:badparam")


def test_get_operator_simple():
    """Test getting operators with default parameters."""
    catalog = OperatorCatalog()

    op = catalog.get("tournament")
    assert isinstance(op, TournamentSelection)


def test_get_operator_with_params():
    """Test getting operators with specified parameters."""
    catalog = OperatorCatalog()

    op = catalog.get("gaussian:mutation_rate=0.2")
    assert isinstance(op, GaussianMutation)
    assert op.mutation_rate == 0.2


def test_get_fitness_evaluator():
    """Test getting fitness evaluators (special config handling)."""
    catalog = OperatorCatalog()

    evaluator = catalog.get("sphere:dim=10")
    assert isinstance(evaluator,
                      BBOBEvaluator)
    assert evaluator.config.num_dims == 10


def test_unknown_operator():
    """Test error for unknown operator type."""
    catalog = OperatorCatalog()

    with pytest.raises(KeyError, match="Unknown operator type: 'unknown'"):
        catalog.get("unknown")


def test_invalid_parameters():
    """Test error for invalid parameters."""
    catalog = OperatorCatalog()

    with pytest.raises(ValueError, match="Invalid parameters"):
        catalog.get("tournament:invalid_param=123")


def test_register_custom_operator():
    """Test registering custom operator types."""
    catalog = OperatorCatalog()

    def custom_factory(param=42):
        return {"custom": True, "param": param}

    catalog.register("custom", custom_factory)

    result = catalog.get("custom:param=100")
    assert result == {"custom": True, "param": 100}


def test_list_available():
    """Test listing available operators."""
    catalog = OperatorCatalog()
    available = catalog.list_available()

    assert "tournament" in available
    assert "gaussian" in available
    assert "sphere" in available
    assert isinstance(available, list)


def test_get_help():
    """Test getting help for operators."""
    catalog = OperatorCatalog()

    help_text = catalog.get_help("tournament")
    assert "tournament:" in help_text

    help_unknown = catalog.get_help("unknown")
    assert "Unknown operator" in help_unknown
