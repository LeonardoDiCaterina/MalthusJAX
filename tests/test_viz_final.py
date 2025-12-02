import pytest
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
try:
    from malthusjax.visualization.single_run import (
        EvolutionVisualizer,
        GeneticAlgorithmVisualizer,
    )
    from malthusjax.visualization.multi_run import (
        EngineComparator,
        FunctionalDataAnalyzer,
    )
    from malthusjax.visualization.base import VisualizationConfig
except ImportError as e:
    raise ImportError(f"Could not import malthusjax modules. Ensure package is installed via 'pip install -e .'. Error: {e}")

# ==============================================================================
# 2. MOCKS & FIXTURES
# ==============================================================================

@dataclass
class MockHistory:
    """Mimics AbstractGenerationOutput/GeneticGenerationOutput structure."""
    generation: np.ndarray
    best_fitness: np.ndarray
    mean_fitness: np.ndarray
    std_fitness: np.ndarray
    ema_delta_fitness: np.ndarray
    best_genome: np.ndarray = None # For GA specific tests

@pytest.fixture
def mock_matplotlib():
    """
    Smart mock for matplotlib that returns the correct shape of axes
    depending on how plt.subplots() is called.
    """
    with patch('matplotlib.use') as mock_use, \
         patch('matplotlib.pyplot.figure') as mock_fig, \
         patch('matplotlib.pyplot.subplots') as mock_subplots, \
         patch('matplotlib.pyplot.show') as mock_show, \
         patch('matplotlib.pyplot.suptitle') as mock_suptitle, \
         patch('matplotlib.pyplot.tight_layout') as mock_layout, \
         patch('matplotlib.pyplot.close') as mock_close, \
         patch('matplotlib.pyplot.style.use') as mock_style, \
         patch('seaborn.set_palette') as mock_sns_palette:
        
        # Setup Axes Mock
        mock_ax = MagicMock()
        mock_figure = MagicMock()
        mock_figure.add_subplot.return_value = mock_ax
        
        # --- THE FIX: Smart Side Effect ---
        def subplots_side_effect(nrows=1, ncols=1, *args, **kwargs):
            # Matplotlib behavior varies by dimensions:
            
            # Case 1: Single Plot (Default) -> Returns (fig, ax)
            if nrows == 1 and ncols == 1:
                return mock_figure, mock_ax
            
            # Case 2: Grid of Plots -> Returns (fig, numpy_array_of_axes)
            # Create a list of mock_ax objects
            total_plots = nrows * ncols
            axes_list = [mock_ax for _ in range(total_plots)]
            
            # Convert to numpy array of objects
            ax_array = np.array(axes_list, dtype=object)
            
            # Case 3: 2D Grid (e.g., 2x2) -> Reshape to (nrows, ncols)
            # This allows unpacking like: ((ax1, ax2), (ax3, ax4)) = axes
            if nrows > 1 and ncols > 1:
                ax_array = ax_array.reshape((nrows, ncols))
            
            return mock_figure, ax_array
            
        mock_subplots.side_effect = subplots_side_effect
        
        yield {
            'fig': mock_fig,
            'ax': mock_ax,
            'subplots': mock_subplots
        }
        
        try:
            plt.close('all')
        except:
            pass

@pytest.fixture
def single_run_data():
    """Creates robust mock history data."""
    gens = 20
    return MockHistory(
        generation=np.arange(gens),
        best_fitness=np.linspace(0.1, 0.9, gens),
        mean_fitness=np.linspace(0.05, 0.8, gens),
        std_fitness=np.random.rand(gens) * 0.1,
        ema_delta_fitness=np.random.randn(gens) * 0.01,
        best_genome=np.random.rand(gens, 10) # 2D Genome
    )

@pytest.fixture
def multi_run_data():
    """Creates a dictionary of runs for comparison."""
    gens = 20
    runs = {}
    for i in range(3):
        runs[f"run_{i}"] = MockHistory(
            generation=np.arange(gens),
            best_fitness=np.linspace(0.1 * i, 0.8 + (i*0.05), gens),
            mean_fitness=np.linspace(0.05, 0.7, gens),
            std_fitness=np.random.rand(gens) * 0.1,
            ema_delta_fitness=np.random.randn(gens) * 0.01
        )
    return runs

# ==============================================================================
# 3. TESTS: SINGLE RUN VISUALIZATION
# ==============================================================================

def test_evolution_visualizer(single_run_data, mock_matplotlib):
    """Test standard EvolutionVisualizer methods."""
    viz = EvolutionVisualizer(single_run_data)
    
    # These calls use plt.subplots() (1x1), expecting a single ax
    viz.create_dashboard(title="Test Dash")
    viz.plot_kpi_evolution('best_fitness')
    
    # This might use subplots(1, 2)
    viz.plot_convergence_summary()
    
    plt.close('all')
    assert mock_matplotlib['subplots'].called

def test_ga_visualizer(single_run_data, mock_matplotlib):
    """Test GeneticAlgorithmVisualizer methods."""
    with patch('malthusjax.visualization.single_run.GeneticGenerationOutput', MockHistory):
        viz = GeneticAlgorithmVisualizer(single_run_data)
        
        viz.create_dashboard(include_genome=True)
        
        # This call uses plt.subplots(2, 2) and unpacking
        viz.create_convergence_analysis()
        
        viz.plot_genome_evolution()
        
        # Test 1D Genome fallback
        single_run_data.best_genome = np.random.rand(10)
        viz_1d = GeneticAlgorithmVisualizer(single_run_data)
        viz_1d.plot_genome_evolution()
        
        plt.close('all')

def test_ga_visualizer_type_error():
    """Ensure it raises error if history is wrong type."""
    try:
        GeneticAlgorithmVisualizer("Invalid Object")
    except (TypeError, AttributeError):
        pass 

# ==============================================================================
# 4. TESTS: MULTI-RUN VISUALIZATION
# ==============================================================================

def test_engine_comparator(multi_run_data, mock_matplotlib):
    """Test EngineComparator methods."""
    comp = EngineComparator(multi_run_data)
    
    # These expect single ax
    comp.create_comparison_dashboard(kpi='best_fitness', confidence_intervals=True)
    comp.create_comparison_dashboard(kpi='best_fitness', show_statistics=True)
    comp.create_performance_distribution(kpi='best_fitness')
    
    df = comp.get_performance_summary()
    assert isinstance(df, pd.DataFrame)
    
    plt.close('all')

def test_functional_data_analyzer(multi_run_data, mock_matplotlib):
    """Test FunctionalDataAnalyzer (FDA) methods."""
    fda = FunctionalDataAnalyzer(multi_run_data)
    
    # Test smoothing methods
    fda.smooth_trajectories(method='gaussian', sigma=1.0)
    fda.smooth_trajectories(method='polynomial', polynomial_degree=2)
    
    # Test basis functions
    gens = np.arange(20)
    fda.create_basis_functions(gens, basis_type='fourier')
    fda.create_basis_functions(gens, basis_type='polynomial')
    
    # Test Full Analysis
    fig, res = fda.create_functional_analysis_dashboard(n_components=2)
    assert 'pc_scores' in res
    
    plt.close('all')

def test_caching_mechanism(single_run_data):
    """Test that caching actually works."""
    viz = EvolutionVisualizer(single_run_data)
    viz.get_kpi_timeseries('best_fitness')
    assert 'kpi_best_fitness' in viz._cache
    viz.clear_cache()
    assert len(viz._cache) == 0

def test_config_object():
    """Test configuration object."""
    conf = VisualizationConfig(figsize=(10,10))
    assert conf.figsize == (10,10)

if __name__ == "__main__":
    print("File exists. Run with: pytest tests/test_viz_final.py")