"""
MalthusJAX Level 4: Abstract Visualization Base Classes

Following MalthusJAX design patterns, these abstract classes provide the foundation
for all visualization components with proper state management and caching.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import functools

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..engine.base import AbstractGenerationOutput


@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for visualization components."""
    figsize: Tuple[int, int] = (15, 10)
    style: str = 'seaborn-v0_8'
    color_palette: str = 'husl'
    dpi: int = 100
    save_format: str = 'png'


class AbstractVisualizer(ABC):
    """
    Abstract base class for all MalthusJAX visualizers.
    
    Follows MalthusJAX design patterns:
    - Stateful initialization with evolution data
    - Cached computations for efficiency
    - Clean, composable interface
    - Configuration-driven styling
    """
    
    def __init__(self, 
                 history: AbstractGenerationOutput,
                 config: Optional[VisualizationConfig] = None):
        """
        Initialize visualizer with evolution history.
        
        Args:
            history: Evolution history from any AbstractEngine
            config: Visualization configuration
        """
        self.history = history
        self.config = config or VisualizationConfig()
        self._cache: Dict[str, Any] = {}
        
        # Extract basic metrics
        self._generations = jnp.arange(len(history.generation))
        self._available_kpis = self._discover_kpis()
        
        # Apply styling
        self._setup_style()
    
    def _setup_style(self):
        """Apply consistent styling."""
        plt.style.use(self.config.style)
        try:
            import seaborn as sns
            sns.set_palette(self.config.color_palette)
        except ImportError:
            pass
    
    def _discover_kpis(self) -> List[str]:
        """Dynamically discover available KPIs from history."""
        fields = list(self.history.__dataclass_fields__.keys())
        # Filter out non-plottable fields
        return [k for k in fields if k not in ['best_genome', 'generation']]
    
    def _get_cached_or_compute(self, key: str, compute_fn: Callable) -> Any:
        """Get cached result or compute and cache it."""
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]
    
    def get_kpi_timeseries(self, kpi_name: str) -> jnp.ndarray:
        """Extract KPI as time series with caching."""
        def compute():
            return getattr(self.history, kpi_name)
        
        return self._get_cached_or_compute(f'kpi_{kpi_name}', compute)
    
    @property
    def available_kpis(self) -> List[str]:
        """Get list of available KPIs."""
        return self._available_kpis
    
    @property
    def generations(self) -> jnp.ndarray:
        """Get generation numbers."""
        return self._generations
    
    def clear_cache(self):
        """Clear visualization cache."""
        self._cache.clear()
    
    @abstractmethod
    def create_dashboard(self, **kwargs) -> plt.Figure:
        """Create main visualization dashboard."""
        pass


class AbstractMultiRunVisualizer(ABC):
    """
    Abstract base class for multi-run visualizers.
    
    Handles multiple evolution histories with efficient caching
    and comparison capabilities.
    """
    
    def __init__(self, 
                 results_dict: Dict[str, AbstractGenerationOutput],
                 config: Optional[VisualizationConfig] = None):
        """
        Initialize with multiple evolution results.
        
        Args:
            results_dict: Dictionary mapping run names to evolution histories
            config: Visualization configuration
        """
        self.results_dict = results_dict
        self.config = config or VisualizationConfig()
        self._cache: Dict[str, Any] = {}
        
        # Validate inputs
        if not results_dict:
            raise ValueError("Results dictionary cannot be empty")
        
        # Extract common properties
        self._run_names = list(results_dict.keys())
        self._n_runs = len(self._run_names)
        self._max_generations = max(len(hist.generation) for hist in results_dict.values())
        
        # Setup styling
        self._setup_style()
    
    def _setup_style(self):
        """Apply consistent styling."""
        plt.style.use(self.config.style)
        try:
            sns.set_palette(self.config.color_palette)
        except ImportError:
            pass
    
    def _get_cached_or_compute(self, key: str, compute_fn: Callable) -> Any:
        """Get cached result or compute and cache it."""
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]
    
    @property
    def run_names(self) -> List[str]:
        """Get list of run names."""
        return self._run_names
    
    @property
    def n_runs(self) -> int:
        """Get number of runs."""
        return self._n_runs
    
    def clear_cache(self):
        """Clear visualization cache."""
        self._cache.clear()
    
    @abstractmethod
    def create_comparison_dashboard(self, **kwargs) -> plt.Figure:
        """Create comparison dashboard across all runs."""
        pass


class VisualizationMixin:
    """Mixin providing common visualization utilities."""
    
    @staticmethod
    def _get_color_palette(n_colors: int) -> np.ndarray:
        """Get appropriate color palette for number of items."""
        if n_colors <= 10:
            return plt.cm.Set3(np.linspace(0, 1, n_colors))
        else:
            return plt.cm.viridis(np.linspace(0, 1, n_colors))
    
    @staticmethod
    def _format_kpi_name(kpi_name: str) -> str:
        """Format KPI name for display."""
        return kpi_name.replace('_', ' ').title()

    @staticmethod
    def _normalize_axes(axes, total_plots: int):
        """
        Normalize the `axes` object returned by `plt.subplots` into a flat list
        of length `total_plots`. This handles cases where tests mock `subplots`
        and return MagicMocks, numpy arrays with odd shapes, or single Axes.
        """
        import numpy as _np
        from collections.abc import Iterable

        def _unwrap_candidate(x):
            # If it's a numpy array, try to extract a contained element
            if isinstance(x, _np.ndarray):
                try:
                    if x.size > 0:
                        return x.flatten()[0]
                    else:
                        return None
                except Exception:
                    return None
            # If it's a list/tuple, return first element
            if isinstance(x, (list, tuple)):
                return x[0] if len(x) > 0 else None
            return x

        flat = []

        # Direct handling for common types
        if isinstance(axes, _np.ndarray):
            try:
                flat = list(axes.ravel())
            except Exception:
                flat = [axes]
        elif isinstance(axes, (list, tuple)):
            flat = list(axes)
        elif hasattr(axes, 'flatten'):
            try:
                flat = list(axes.flatten())
            except Exception:
                # Fallback to treating as single object
                flat = [axes]
        elif isinstance(axes, Iterable) and not isinstance(axes, (str, bytes)):
            try:
                flat = list(axes)
            except Exception:
                flat = [axes]
        else:
            flat = [axes]

        # Post-process entries: unwrap numpy arrays or nested structures
        processed = []
        for item in flat:
            cand = _unwrap_candidate(item)
            if cand is None:
                # keep original if unwrap failed
                processed.append(item)
            else:
                processed.append(cand)

        # Remove falsy trailing items
        processed = [p for p in processed if p is not None]

        # If still empty, provide a placeholder None
        if len(processed) == 0:
            processed = [None]

        # Expand or truncate to match requested total_plots
        if len(processed) < total_plots:
            processed = processed + [processed[-1]] * (total_plots - len(processed))

        return processed[:total_plots]
    
    @staticmethod
    def _add_statistics_box(ax: plt.Axes, values: List[float], 
                           kpi_name: str, position: Tuple[float, float] = (0.02, 0.98)):
        """Add statistics text box to axes."""
        stats_text = (f'Final {VisualizationMixin._format_kpi_name(kpi_name)}:\n'
                     f'Mean: {np.mean(values):.3f}\n'
                     f'Std: {np.std(values):.3f}\n'
                     f'Best: {np.max(values):.3f}\n'
                     f'Worst: {np.min(values):.3f}')
        
        ax.text(position[0], position[1], stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))