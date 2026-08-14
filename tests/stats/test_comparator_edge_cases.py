import pytest
import numpy as np
from malthusjax.stats.comparator import compare_paired_arrays

def test_comparator_identical_arrays():
    """Test comparator handling of identical arrays."""
    
    # Identical arrays have zero variance and zero difference
    arr1 = np.ones(50)
    arr2 = np.ones(50)
    
    from malthusjax.stats.core import StatisticalComparisonSpec
    spec = StatisticalComparisonSpec()
    
    # The Mann-Whitney U test or t-test might raise warnings or handle 
    # zero variance with fallbacks. We just ensure it doesn't crash.
    metrics = compare_paired_arrays(
        label="test",
        left_name="left",
        right_name="right",
        left=arr1,
        right=arr2, 
        spec=spec
    )
    
    assert metrics is not None

def test_comparator_mismatched_lengths():
    """Test comparator handling arrays with different numbers of seeds."""
    
    from malthusjax.stats.core import StatisticalComparisonSpec
    spec = StatisticalComparisonSpec()
    
    arr1 = np.random.randn(5, 10) # 5 seeds
    arr2 = np.random.randn(4, 10) # 4 seeds
    
    with pytest.raises(ValueError):
        compare_paired_arrays(
            label="test",
            left_name="left",
            right_name="right",
            left=arr1,
            right=arr2, 
            spec=spec
        )

def test_comparator_other_tests():
    """Test other statistical methods."""
    arr1 = np.random.randn(50)
    arr2 = np.random.randn(50) + 1.0
    
    from malthusjax.stats.core import StatisticalComparisonSpec
    spec = StatisticalComparisonSpec(include_tests=("wilcoxon", "paired_t", "tost", "sign"))
    metrics = compare_paired_arrays(
        label="test",
        left_name="left",
        right_name="right",
        left=arr1,
        right=arr2,
        spec=spec
    )
    assert metrics is not None
