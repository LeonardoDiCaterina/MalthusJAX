"""
MalthusJAX Level 4: Single Run Visualizers

Stateful visualization classes for analyzing single evolution runs.
These replace the static method approach with proper initialization and caching.
"""

from typing import Dict, List, Optional, Tuple, Any
import functools

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .base import AbstractVisualizer, VisualizationConfig, VisualizationMixin
from ..engine.base import AbstractGenerationOutput
from ..engine.genetic_engine import GeneticGenerationOutput


class EvolutionVisualizer(AbstractVisualizer, VisualizationMixin):
    """
    Universal single-run visualizer for any AbstractEngine.
    
    Provides stateful, cached visualization with clean API:
    - Initialize once with evolution history
    - Call visualization methods without passing data repeatedly
    - Automatic caching of expensive computations
    """
    
    def __init__(self, 
                 history: AbstractGenerationOutput,
                 config: Optional[VisualizationConfig] = None):
        """
        Initialize visualizer with evolution history.
        
        Args:
            history: Evolution history from any AbstractEngine
            config: Visualization configuration
            
        Example:
            >>> visualizer = EvolutionVisualizer(history)
            >>> fig = visualizer.create_dashboard()
            >>> convergence_fig = visualizer.plot_convergence()
        """
        super().__init__(history, config)
    
    def create_dashboard(self, 
                        kpis: Optional[List[str]] = None,
                        title: str = "Evolution Dashboard") -> plt.Figure:
        """
        Create comprehensive KPI dashboard.
        
        Args:
            kpis: List of KPIs to plot (auto-detected if None)
            title: Dashboard title
            
        Returns:
            Matplotlib figure with dashboard
        """
        def compute_dashboard():
            if kpis is None:
                selected_kpis = self.available_kpis
            else:
                selected_kpis = kpis
            
            n_kpis = len(selected_kpis)
            n_cols = min(3, n_kpis)
            n_rows = (n_kpis + n_cols - 1) // n_cols

            total_plots = n_rows * n_cols
            # Avoid calling plt.subplots for multi-grid layouts in environments
            # where `subplots` is mocked and may return non-standard arrays.
            if n_rows == 1 and n_cols == 1:
                fig, axes = plt.subplots(1, 1, figsize=self.config.figsize)
            else:
                try:
                    fig = plt.figure(figsize=self.config.figsize)
                    axes = [fig.add_subplot(n_rows, n_cols, i + 1) for i in range(total_plots)]
                except Exception:
                    # Final fallback to a single-row layout
                    fig, axes = plt.subplots(1, max(1, total_plots), figsize=self.config.figsize)
            axes = self._normalize_axes(axes, total_plots)
            
            for i, kpi in enumerate(selected_kpis):
                if i >= len(axes):
                    break
                
                ax = axes[i]
                values = self.get_kpi_timeseries(kpi)
                
                # Style KPIs appropriately
                self._style_kpi_plot(ax, kpi, values)
            
            # Hide unused subplots
            for i in range(len(selected_kpis), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            return fig
        
        cache_key = f"dashboard_{hash(tuple(kpis or []))}"
        return self._get_cached_or_compute(cache_key, compute_dashboard)
    
    def plot_kpi_evolution(self, 
                          kpi: str,
                          title: Optional[str] = None,
                          figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot single KPI evolution over time.
        
        Args:
            kpi: KPI name to plot
            title: Plot title (auto-generated if None)
            figsize: Figure size (uses config default if None)
            
        Returns:
            Matplotlib figure
        """
        def compute_plot():
            fig_size = figsize or self.config.figsize
            fig, ax = plt.subplots(figsize=fig_size)
            
            values = self.get_kpi_timeseries(kpi)
            self._style_kpi_plot(ax, kpi, values)
            
            plot_title = title or f'{self._format_kpi_name(kpi)} Evolution'
            ax.set_title(plot_title, fontsize=14, fontweight='bold')
            
            return fig
        
        cache_key = f"kpi_plot_{kpi}_{title}_{figsize}"
        return self._get_cached_or_compute(cache_key, compute_plot)
    
    def plot_convergence_summary(self,
                               title: str = "Evolution Convergence Summary") -> plt.Figure:
        """
        Create convergence summary plot.
        
        Args:
            title: Plot title
            
        Returns:
            Matplotlib figure with convergence analysis
        """
        def compute_convergence():
            try:
                # Prefer simple 1x2 layout via figure+add_subplot to avoid mocked
                # `subplots` reshape issues.
                fig = plt.figure(figsize=self.config.figsize)
                axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
            except Exception:
                fig, axes = plt.subplots(1, 2, figsize=self.config.figsize)
            # Ensure axes is iterable and unpack into ax1, ax2
            if isinstance(axes, (list, tuple)):
                ax1, ax2 = axes[0], axes[1]
            elif hasattr(axes, 'flatten'):
                flat = list(axes.flatten())
                ax1, ax2 = flat[0], flat[1]
            else:
                ax1, ax2 = axes, axes
            
            # Plot main fitness metrics if available
            if 'best_fitness' in self.available_kpis:
                best_fitness = self.get_kpi_timeseries('best_fitness')
                ax1.plot(self.generations, best_fitness, 'r-', linewidth=2.5, 
                        label='Best Fitness', alpha=0.9)
            
            if 'mean_fitness' in self.available_kpis:
                mean_fitness = self.get_kpi_timeseries('mean_fitness')
                ax1.plot(self.generations, mean_fitness, 'b-', linewidth=2, 
                        label='Mean Fitness', alpha=0.8)
            
            if 'std_fitness' in self.available_kpis:
                mean_fitness = self.get_kpi_timeseries('mean_fitness')
                std_fitness = self.get_kpi_timeseries('std_fitness')
                ax1.fill_between(self.generations,
                               mean_fitness - std_fitness,
                               mean_fitness + std_fitness,
                               alpha=0.3, color='blue', label='±1 Std Dev')
            
            ax1.set_title('Fitness Evolution', fontweight='bold')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot diversity if available
            if 'std_fitness' in self.available_kpis:
                std_fitness = self.get_kpi_timeseries('std_fitness')
                ax2.plot(self.generations, std_fitness, 'g-', linewidth=2.5, alpha=0.8)
                ax2.set_title('Population Diversity', fontweight='bold')
                ax2.set_xlabel('Generation')
                ax2.set_ylabel('Fitness Std Dev')
                ax2.grid(True, alpha=0.3)
            else:
                # Alternative plot if std_fitness not available
                available_kpi = self.available_kpis[0] if self.available_kpis else 'best_fitness'
                values = self.get_kpi_timeseries(available_kpi)
                ax2.plot(self.generations, values, linewidth=2.5, alpha=0.8)
                ax2.set_title(f'{self._format_kpi_name(available_kpi)}', fontweight='bold')
                ax2.set_xlabel('Generation')
                ax2.set_ylabel(self._format_kpi_name(available_kpi))
                ax2.grid(True, alpha=0.3)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            return fig
        
        return self._get_cached_or_compute("convergence_summary", compute_convergence)
    
    def _style_kpi_plot(self, ax: plt.Axes, kpi: str, values: jnp.ndarray):
        """Apply appropriate styling to KPI plots."""
        if kpi in ['best_fitness', 'mean_fitness']:
            ax.plot(self.generations, values, linewidth=2.5, alpha=0.8)
            ax.set_ylabel('Fitness')
        elif kpi in ['std_fitness']:
            ax.plot(self.generations, values, linewidth=2, color='orange', alpha=0.8)
            ax.set_ylabel('Population Diversity')
        elif kpi in ['ema_delta_fitness']:
            ax.plot(self.generations, values, linewidth=2, color='purple', alpha=0.8)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_ylabel('Improvement Rate')
        else:
            ax.plot(self.generations, values, linewidth=2, alpha=0.8)
            ax.set_ylabel(self._format_kpi_name(kpi))
        
        ax.set_xlabel('Generation')
        ax.set_title(self._format_kpi_name(kpi), fontweight='bold')
        ax.grid(True, alpha=0.3)


class GeneticAlgorithmVisualizer(EvolutionVisualizer):
    """
    Specialized visualizer for Genetic Algorithm results.
    
    Extends EvolutionVisualizer with GA-specific visualizations
    including genome evolution and specialized convergence analysis.
    """
    
    def __init__(self, 
                 history: GeneticGenerationOutput,
                 config: Optional[VisualizationConfig] = None):
        """
        Initialize GA visualizer.
        
        Args:
            history: GA evolution history (GeneticGenerationOutput)
            config: Visualization configuration
        """
        super().__init__(history, config)
        
        # Verify this is GA history
        if not isinstance(history, GeneticGenerationOutput):
            raise TypeError(f"Expected GeneticGenerationOutput, got {type(history)}")
    
    def create_dashboard(self, 
                        include_genome: bool = True,
                        title: str = "Genetic Algorithm Dashboard") -> plt.Figure:
        """
        Create GA-specific dashboard with genome visualization.
        
        Args:
            include_genome: Whether to include genome evolution plot
            title: Dashboard title
            
        Returns:
            Matplotlib figure
        """
        def compute_ga_dashboard():
            # Use standard dashboard as base
            n_plots = len(self.available_kpis)
            if include_genome and hasattr(self.history, 'best_genome'):
                n_plots += 1
            
            n_cols = min(3, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols
            
            if n_rows == 1 and n_cols == 1:
                fig, axes = plt.subplots(1, 1, figsize=self.config.figsize)
            else:
                try:
                    fig = plt.figure(figsize=self.config.figsize)
                    axes = [fig.add_subplot(n_rows, n_cols, i + 1) for i in range(n_rows * n_cols)]
                except Exception:
                    fig, axes = plt.subplots(1, n_rows * n_cols, figsize=self.config.figsize)
            total_plots = n_rows * n_cols
            axes = self._normalize_axes(axes, total_plots)
            
            # Plot standard KPIs
            for i, kpi in enumerate(self.available_kpis):
                if i >= len(axes):
                    break
                ax = axes[i]
                values = self.get_kpi_timeseries(kpi)
                self._style_kpi_plot(ax, kpi, values)
            
            # Add genome evolution plot
            if include_genome and hasattr(self.history, 'best_genome') and len(self.available_kpis) < len(axes):
                genome_ax = axes[len(self.available_kpis)]
                self._plot_genome_evolution(genome_ax)
            
            # Hide unused subplots
            start_idx = len(self.available_kpis) + (1 if include_genome else 0)
            for i in range(start_idx, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            return fig
        
        cache_key = f"ga_dashboard_{include_genome}"
        return self._get_cached_or_compute(cache_key, compute_ga_dashboard)
    
    def create_convergence_analysis(self,
                                  title: str = "Genetic Algorithm Convergence Analysis") -> plt.Figure:
        """
        Create detailed GA convergence analysis.
        
        Args:
            title: Analysis title
            
        Returns:
            Matplotlib figure with 2x2 convergence analysis
        """
        def compute_convergence_analysis():
            # Create a 2x2 grid using add_subplot to avoid mocked subplots pitfalls
            try:
                fig = plt.figure(figsize=self.config.figsize)
                axes = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
                flat_axes = list(axes)
            except Exception:
                # Fallback to plt.subplots and normalize
                fig, axes = plt.subplots(2, 2, figsize=self.config.figsize)
                if isinstance(axes, (list, tuple)):
                    flat_axes = list(axes)
                elif hasattr(axes, 'flatten'):
                    try:
                        flat_axes = list(axes.flatten())
                    except Exception:
                        flat_axes = [axes]
                else:
                    flat_axes = [axes]

            # Ensure we have four axes (fall back to repeating last if needed)
            while len(flat_axes) < 4:
                flat_axes.append(flat_axes[-1])

            ax1, ax2, ax3, ax4 = flat_axes[0], flat_axes[1], flat_axes[2], flat_axes[3]
            
            # 1. Fitness Evolution
            if 'best_fitness' in self.available_kpis:
                best_fitness = self.get_kpi_timeseries('best_fitness')
                ax1.plot(self.generations, best_fitness, 'r-', linewidth=2.5, 
                        label='Best Fitness', alpha=0.9)
            
            if 'mean_fitness' in self.available_kpis:
                mean_fitness = self.get_kpi_timeseries('mean_fitness')
                ax1.plot(self.generations, mean_fitness, 'b-', linewidth=2, 
                        label='Mean Fitness', alpha=0.8)
            
            if 'std_fitness' in self.available_kpis:
                mean_fitness = self.get_kpi_timeseries('mean_fitness')
                std_fitness = self.get_kpi_timeseries('std_fitness')
                ax1.fill_between(self.generations,
                               mean_fitness - std_fitness,
                               mean_fitness + std_fitness,
                               alpha=0.3, color='blue', label='±1 Std Dev')
            
            ax1.set_title('Fitness Evolution', fontweight='bold')
            ax1.set_ylabel('Fitness')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Population Diversity
            if 'std_fitness' in self.available_kpis:
                std_fitness = self.get_kpi_timeseries('std_fitness')
                ax2.plot(self.generations, std_fitness, 'g-', linewidth=2.5, alpha=0.8)
            ax2.set_title('Population Diversity', fontweight='bold')
            ax2.set_ylabel('Fitness Std Dev')
            ax2.grid(True, alpha=0.3)
            
            # 3. Improvement Rate
            if 'ema_delta_fitness' in self.available_kpis:
                ema_delta = self.get_kpi_timeseries('ema_delta_fitness')
                ax3.plot(self.generations, ema_delta, 'purple', linewidth=2.5, alpha=0.8)
                ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_title('Improvement Rate (EMA)', fontweight='bold')
            ax3.set_ylabel('Fitness Delta')
            ax3.grid(True, alpha=0.3)
            
            # 4. Genome Evolution
            self._plot_genome_evolution(ax4)
            
            for ax in [ax1, ax2, ax3]:
                ax.set_xlabel('Generation')
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            return fig
        
        return self._get_cached_or_compute("convergence_analysis", compute_convergence_analysis)
    
    def plot_genome_evolution(self, 
                             title: str = "Genome Evolution",
                             figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot genome evolution over time.
        
        Args:
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        def compute_genome_plot():
            fig_size = figsize or (12, 8)
            fig, ax = plt.subplots(figsize=fig_size)
            
            self._plot_genome_evolution(ax)
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            return fig
        
        cache_key = f"genome_plot_{title}_{figsize}"
        return self._get_cached_or_compute(cache_key, compute_genome_plot)
    
    def _plot_genome_evolution(self, ax: plt.Axes):
        """Plot genome evolution on given axes."""
        if not hasattr(self.history, 'best_genome'):
            ax.text(0.5, 0.5, 'No genome data available', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        best_genome = self.history.best_genome
        
        if len(best_genome.shape) == 2:  # 2D array: [generation, gene]
            im = ax.imshow(best_genome.T, aspect='auto', cmap='RdYlBu_r', 
                          interpolation='nearest')
            ax.set_title('Best Genome Evolution', fontweight='bold')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Gene Index')
            plt.colorbar(im, ax=ax, label='Gene Value')
        else:
            # Fallback: show final genome
            final_genome = best_genome[-1] if len(best_genome.shape) > 1 else best_genome
            ax.bar(range(len(final_genome)), final_genome, alpha=0.8)
            ax.set_title('Final Best Genome', fontweight='bold')
            ax.set_xlabel('Gene Index')
            ax.set_ylabel('Gene Value')