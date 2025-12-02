"""
MalthusJAX Level 4: Multi-Run Visualizers

Stateful visualization classes for analyzing multiple evolution runs.
Includes advanced functional data analysis with caching for efficiency.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import functools

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.interpolate import BSpline



from .base import AbstractMultiRunVisualizer, VisualizationConfig, VisualizationMixin
from ..engine.base import AbstractGenerationOutput


class EngineComparator(AbstractMultiRunVisualizer, VisualizationMixin):
    """
    Stateful multi-run comparison and analysis.
    
    Provides comprehensive comparison tools with caching:
    - Initialize once with multiple run results
    - Generate comparisons without re-passing data
    - Cached statistical computations
    - Clean API for performance analysis
    """
    
    def __init__(self, 
                 results_dict: Dict[str, AbstractGenerationOutput],
                 config: Optional[VisualizationConfig] = None):
        """
        Initialize with multiple evolution results.
        
        Args:
            results_dict: Dictionary mapping run names to evolution histories
            config: Visualization configuration
            
        Example:
            >>> comparator = EngineComparator(results_dict)
            >>> fig = comparator.create_comparison_dashboard()
            >>> summary = comparator.get_performance_summary()
        """
        super().__init__(results_dict, config)
        
        # Cache commonly used computations
        self._performance_summary = None
        self._final_values_cache: Dict[str, List[float]] = {}
    
    def create_comparison_dashboard(self, 
                                  kpi: str = 'best_fitness',
                                  confidence_intervals: bool = False,
                                  show_statistics: bool = True,
                                  title: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive comparison dashboard.
        
        Args:
            kpi: KPI to compare across runs
            confidence_intervals: Whether to show confidence bands
            show_statistics: Whether to display statistics box
            title: Dashboard title
            
        Returns:
            Matplotlib figure with comparison plots
        """
        def compute_comparison():
            fig, ax = plt.subplots(figsize=self.config.figsize)
            
            # Get color palette
            colors = self._get_color_palette(self.n_runs)
            
            all_values = []
            
            # Plot individual runs
            for i, (run_name, history) in enumerate(self.results_dict.items()):
                generations = jnp.arange(len(history.generation))
                values = getattr(history, kpi)
                all_values.append(values)
                
                ax.plot(generations, values, linewidth=2, label=run_name, 
                       color=colors[i], alpha=0.8)
            
            # Add confidence intervals if requested
            if confidence_intervals and len(all_values) > 1:
                all_values_array = jnp.array(all_values)
                mean_values = jnp.mean(all_values_array, axis=0)
                std_values = jnp.std(all_values_array, axis=0)
                
                ax.fill_between(range(len(mean_values)), 
                               mean_values - std_values, 
                               mean_values + std_values,
                               alpha=0.2, color='gray', label='±1 Std Dev')
                ax.plot(range(len(mean_values)), mean_values, 'k--', linewidth=2, 
                       label='Mean', alpha=0.7)
            
            # Formatting
            ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
            ax.set_ylabel(self._format_kpi_name(kpi), fontsize=12, fontweight='bold')
            
            plot_title = title or f'{self._format_kpi_name(kpi)} Comparison Across Runs'
            ax.set_title(plot_title, fontsize=14, fontweight='bold')
            
            # Legend placement
            if self.n_runs <= 8:
                ax.legend(fontsize=10, loc='best')
            else:
                ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.subplots_adjust(right=0.85)
            
            ax.grid(True, alpha=0.3)
            
            # Add statistics if requested
            if show_statistics and len(all_values) > 1:
                final_values = self.get_final_values(kpi)
                self._add_statistics_box(ax, final_values, kpi)
            
            return fig
        
        cache_key = f"comparison_{kpi}_{confidence_intervals}_{show_statistics}"
        return self._get_cached_or_compute(cache_key, compute_comparison)
    
    def create_performance_distribution(self, 
                                      kpi: str = 'best_fitness',
                                      title: Optional[str] = None) -> plt.Figure:
        """
        Create distribution analysis of final performance.
        
        Args:
            kpi: KPI to analyze
            title: Plot title
            
        Returns:
            Matplotlib figure with histogram and box plot
        """
        def compute_distribution():
            try:
                fig = plt.figure(figsize=self.config.figsize)
                ax1 = fig.add_subplot(1, 2, 1)
                ax2 = fig.add_subplot(1, 2, 2)
                axes = [ax1, ax2]
            except Exception:
                fig, axes = plt.subplots(1, 2, figsize=self.config.figsize)
                axes = self._normalize_axes(axes, 2)
                ax1, ax2 = axes[0], axes[1]

            final_values = self.get_final_values(kpi)

            # Histogram
            ax1.hist(final_values, bins=min(15, max(1, len(final_values)//2)), 
                    alpha=0.7, edgecolor='black', color='skyblue')
            ax1.axvline(np.mean(final_values), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(final_values):.3f}')
            ax1.axvline(np.median(final_values), color='orange', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(final_values):.3f}')
            ax1.set_xlabel(f'Final {self._format_kpi_name(kpi)}', fontweight='bold')
            ax1.set_ylabel('Frequency', fontweight='bold')
            ax1.set_title(f'Distribution of Final {self._format_kpi_name(kpi)}', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(final_values, tick_labels=[self._format_kpi_name(kpi)])
            ax2.set_ylabel(f'Final {self._format_kpi_name(kpi)}', fontweight='bold')
            ax2.set_title(f'Box Plot of Final {self._format_kpi_name(kpi)}', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            if title:
                plt.suptitle(title, fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            return fig
        
        cache_key = f"distribution_{kpi}"
        return self._get_cached_or_compute(cache_key, compute_distribution)
    
    def get_performance_summary(self) -> pd.DataFrame:
        """
        Generate comprehensive performance summary table.
        
        Returns:
            Pandas DataFrame with summary statistics
        """
        if self._performance_summary is not None:
            return self._performance_summary
        
        summary_data = []
        
        for run_name, history in self.results_dict.items():
            final_best = float(getattr(history, 'best_fitness', [0])[-1])
            initial_best = float(getattr(history, 'best_fitness', [0])[0])
            total_improvement = final_best - initial_best
            
            # Find convergence point
            if hasattr(history, 'best_fitness') and len(history.best_fitness) > 1:
                improvements = jnp.diff(history.best_fitness)
                converged_gens = jnp.where(jnp.abs(improvements) < 0.001)[0]
                convergence_gen = int(converged_gens[0]) if len(converged_gens) > 0 else len(history.generation)
                improvement_rate = total_improvement / len(history.best_fitness)
            else:
                convergence_gen = 0
                improvement_rate = 0.0
            
            # Additional metrics if available
            final_mean = float(getattr(history, 'mean_fitness', [final_best])[-1])
            final_diversity = float(getattr(history, 'std_fitness', [0])[-1])
            
            summary_data.append({
                'Run': run_name,
                'Final Best': f'{final_best:.4f}',
                'Final Mean': f'{final_mean:.4f}',
                'Total Improvement': f'{total_improvement:.4f}',
                'Improvement Rate': f'{improvement_rate:.6f}',
                'Convergence Generation': convergence_gen,
                'Final Diversity': f'{final_diversity:.4f}'
            })
        
        self._performance_summary = pd.DataFrame(summary_data)
        return self._performance_summary
    
    def get_final_values(self, kpi: str) -> List[float]:
        """
        Get final values for a KPI across all runs with caching.
        
        Args:
            kpi: KPI name
            
        Returns:
            List of final values
        """
        if kpi not in self._final_values_cache:
            final_values = []
            for history in self.results_dict.values():
                if hasattr(history, kpi):
                    values = getattr(history, kpi)
                    if isinstance(values, (list, jnp.ndarray, np.ndarray)) and len(values) > 0:
                        final_values.append(float(values[-1]))
            self._final_values_cache[kpi] = final_values
        
        return self._final_values_cache[kpi]
    
    def get_statistical_summary(self, kpi: str = 'best_fitness') -> Dict[str, float]:
        """
        Get statistical summary for a KPI.
        
        Args:
            kpi: KPI name
            
        Returns:
            Dictionary with statistical measures
        """
        final_values = self.get_final_values(kpi)
        improvements = [self.get_improvement(run_name, kpi) for run_name in self.run_names]
        
        return {
            'mean_final': np.mean(final_values),
            'std_final': np.std(final_values),
            'best_run': np.max(final_values),
            'worst_run': np.min(final_values),
            'median_final': np.median(final_values),
            'cv_percent': np.std(final_values) / np.mean(final_values) * 100,
            'mean_improvement': np.mean(improvements),
            'std_improvement': np.std(improvements),
            'success_rate': np.sum(np.array(improvements) > 0) / len(improvements) * 100
        }
    
    def get_improvement(self, run_name: str, kpi: str = 'best_fitness') -> float:
        """
        Calculate improvement for a specific run and KPI.
        
        Args:
            run_name: Name of the run
            kpi: KPI name
            
        Returns:
            Improvement value (final - initial)
        """
        history = self.results_dict[run_name]
        if hasattr(history, kpi):
            values = getattr(history, kpi)
            if isinstance(values, (list, jnp.ndarray, np.ndarray)) and len(values) > 0:
                return float(values[-1] - values[0])
        return 0.0
    
    def clear_cache(self):
        """Clear all cached computations."""
        super().clear_cache()
        self._performance_summary = None
        self._final_values_cache.clear()


class FunctionalDataAnalyzer(AbstractMultiRunVisualizer, VisualizationMixin):
    """
    Specialized class for functional data analysis of evolutionary trajectories.
    
    Provides advanced FDA capabilities with caching:
    - Trajectory smoothing with multiple methods
    - Functional Principal Component Analysis (fPCA)
    - Multiple basis function types
    - Comprehensive visualization suite
    """
    
    def __init__(self, 
                 results_dict: Dict[str, AbstractGenerationOutput],
                 config: Optional[VisualizationConfig] = None):
        """
        Initialize functional data analyzer.
        
        Args:
            results_dict: Dictionary of evolution results
            config: Visualization configuration
            
        Example:
            >>> fda = FunctionalDataAnalyzer(results_dict)
            >>> fig = fda.create_functional_analysis_dashboard()
            >>> fpca_results = fda.perform_functional_pca()
        """
        super().__init__(results_dict, config)
        
        # FDA-specific caches
        self._smoothed_data_cache: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        self._fpca_results_cache: Dict[str, Dict[str, Any]] = {}
        self._basis_cache: Dict[str, np.ndarray] = {}
    
    def smooth_trajectories(self, 
                           kpi: str = 'best_fitness',
                           method: str = 'gaussian',
                           sigma: float = 2.0,
                           polynomial_degree: int = 3) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Smooth evolutionary trajectories for functional analysis.
        
        Args:
            kpi: KPI to smooth
            method: Smoothing method ('gaussian', 'polynomial', 'spline', 'savgol')
            sigma: Gaussian smoothing parameter
            polynomial_degree: Polynomial degree for polynomial/savgol methods
            
        Returns:
            Dictionary of smoothed trajectories {run_name: (generations, smoothed_values)}
        """
        cache_key = f"{kpi}_{method}_{sigma}_{polynomial_degree}"
        
        if cache_key not in self._smoothed_data_cache:
            from scipy import ndimage, interpolate
            try:
                from scipy.signal import savgol_filter
            except ImportError:
                savgol_filter = None
            
            smoothed_data = {}
            
            for run_name, history in self.results_dict.items():
                generations = np.arange(len(history.generation))
                values = np.array(getattr(history, kpi))
                
                if method == 'gaussian':
                    smoothed_values = ndimage.gaussian_filter1d(values, sigma=sigma)
                elif method == 'polynomial':
                    coeffs = np.polyfit(generations, values, polynomial_degree)
                    smoothed_values = np.polyval(coeffs, generations)
                elif method == 'spline':
                    spline = interpolate.UnivariateSpline(generations, values, s=len(values))
                    smoothed_values = spline(generations)
                elif method == 'savgol' and savgol_filter is not None:
                    window_length = min(max(5, len(values) // 10), len(values) - 1)
                    if window_length % 2 == 0:
                        window_length += 1
                    smoothed_values = savgol_filter(values, window_length, polynomial_degree)
                else:
                    smoothed_values = values
                
                smoothed_data[run_name] = (generations, smoothed_values)
            
            self._smoothed_data_cache[cache_key] = smoothed_data
        
        return self._smoothed_data_cache[cache_key]
    
    def create_basis_functions(self, 
                              generations: np.ndarray,
                              basis_type: str = 'fourier',
                              n_basis: int = 10) -> np.ndarray:
        """
        Create basis functions for functional analysis.
        
        Args:
            generations: Time points (generation numbers)
            basis_type: Type of basis ('fourier', 'polynomial', 'bspline', 'gaussian')
            n_basis: Number of basis functions
            
        Returns:
            Basis matrix of shape (len(generations), n_basis)
        """
        cache_key = f"{len(generations)}_{basis_type}_{n_basis}_{generations[0]}_{generations[-1]}"
        
        if cache_key not in self._basis_cache:
            n_points = len(generations)
            basis_matrix = np.zeros((n_points, n_basis))
            t_norm = (generations - generations.min()) / (generations.max() - generations.min())
            
            if basis_type == 'fourier':
                basis_matrix[:, 0] = 1.0  # Constant term
                for k in range(1, n_basis):
                    if k % 2 == 1:  # Sine functions
                        freq = (k + 1) // 2
                        basis_matrix[:, k] = np.sin(2 * np.pi * freq * t_norm)
                    else:  # Cosine functions
                        freq = k // 2
                        basis_matrix[:, k] = np.cos(2 * np.pi * freq * t_norm)
            
            elif basis_type == 'polynomial':
                for k in range(n_basis):
                    basis_matrix[:, k] = t_norm ** k
            
            elif basis_type == 'bspline':
                try:
                    knots = np.linspace(0, 1, n_basis - 2)
                    full_knots = np.concatenate([np.zeros(3), knots, np.ones(3)])
                    for k in range(n_basis):
                        coeff = np.zeros(n_basis)
                        coeff[k] = 1
                        spline = BSpline(full_knots, coeff, 3)
                        basis_matrix[:, k] = spline(t_norm)
                except ImportError:
                    # Fallback to polynomial
                    for k in range(n_basis):
                        basis_matrix[:, k] = t_norm ** k
            
            elif basis_type == 'gaussian':
                centers = np.linspace(0, 1, n_basis)
                sigma = 1.0 / (2 * n_basis)
                for k in range(n_basis):
                    basis_matrix[:, k] = np.exp(-0.5 * ((t_norm - centers[k]) / sigma) ** 2)
            
            self._basis_cache[cache_key] = basis_matrix
        
        return self._basis_cache[cache_key]
    
    def perform_functional_pca(self, 
                              kpi: str = 'best_fitness',
                              smoothing_method: str = 'gaussian',
                              basis_type: str = 'fourier',
                              n_basis: int = 15,
                              n_components: int = 3) -> Dict[str, Any]:
        """
        Perform functional Principal Component Analysis.
        
        Args:
            kpi: KPI to analyze
            smoothing_method: Smoothing method for trajectories
            basis_type: Type of basis functions
            n_basis: Number of basis functions
            n_components: Number of principal components
            
        Returns:
            Dictionary containing fPCA results
        """
        cache_key = f"{kpi}_{smoothing_method}_{basis_type}_{n_basis}_{n_components}"
        
        if cache_key not in self._fpca_results_cache:
            
            # Get smoothed data
            smoothed_data = self.smooth_trajectories(kpi, smoothing_method)
            
            # Extract common time grid
            all_generations = []
            for generations, _ in smoothed_data.values():
                all_generations.extend(generations)
            
            time_grid = np.linspace(min(all_generations), max(all_generations), 
                                   max([len(g) for g, _ in smoothed_data.values()]))
            
            # Create basis functions
            basis_matrix = self.create_basis_functions(time_grid, basis_type, n_basis)
            
            # Project trajectories onto basis
            trajectory_coefficients = []
            trajectory_names = []
            
            for run_name, (generations, values) in smoothed_data.items():
                # Interpolate to common time grid
                interp_values = np.interp(time_grid, generations, values)
                
                # Project onto basis functions
                coeffs = np.linalg.lstsq(basis_matrix, interp_values, rcond=None)[0]
                trajectory_coefficients.append(coeffs)
                trajectory_names.append(run_name)
            
            # Perform PCA on coefficients
            coeff_matrix = np.array(trajectory_coefficients)
            pca = PCA(n_components=n_components)
            pc_scores = pca.fit_transform(coeff_matrix)
            
            # Reconstruct principal component functions
            pc_functions = []
            for i in range(n_components):
                pc_coeffs = pca.components_[i, :]
                pc_function = basis_matrix @ pc_coeffs
                pc_functions.append(pc_function)
            
            # Calculate mean function
            mean_coeffs = np.mean(coeff_matrix, axis=0)
            mean_function = basis_matrix @ mean_coeffs
            
            results = {
                'pc_scores': pc_scores,
                'pc_functions': pc_functions,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'mean_function': mean_function,
                'time_grid': time_grid,
                'basis_matrix': basis_matrix,
                'trajectory_coefficients': coeff_matrix,
                'trajectory_names': trajectory_names,
                'pca_model': pca,
                'basis_type': basis_type,
                'smoothed_data': smoothed_data
            }
            
            self._fpca_results_cache[cache_key] = results
        
        return self._fpca_results_cache[cache_key]
    
    def create_functional_analysis_dashboard(self, 
                                           kpi: str = 'best_fitness',
                                           smoothing_method: str = 'gaussian',
                                           basis_type: str = 'fourier',
                                           n_basis: int = 15,
                                           n_components: int = 3,
                                           title: Optional[str] = None) -> Tuple[plt.Figure, Dict[str, Any]]:
        """
        Create comprehensive functional data analysis dashboard.
        
        Args:
            kpi: KPI to analyze
            smoothing_method: Smoothing method (possible: 'gaussian', 'polynomial', 'spline', 'savgol')
            basis_type: Basis function type (possible: 'fourier', 'polynomial', 'bspline', 'gaussian')
            n_basis: Number of basis functions 
            n_components: Number of principal components
            title: Dashboard title
            
        Returns:
            Tuple of (matplotlib figure, fPCA results dictionary)
        """
        def compute_dashboard():
            # Perform functional analysis
            fpca_results = self.perform_functional_pca(
                kpi, smoothing_method, basis_type, n_basis, n_components
            )
            
            # Create comprehensive visualization
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # 1. Raw vs Smoothed trajectories
            self._plot_trajectories(fig, gs, fpca_results, kpi)
            
            # 2. Principal component functions
            self._plot_pc_functions(fig, gs, fpca_results)
            
            # 3. PC scores and variance
            self._plot_pc_analysis(fig, gs, fpca_results)
            
            # 4. Basis functions
            self._plot_basis_functions(fig, gs, fpca_results)
            
            # 5. Reconstruction quality
            self._plot_reconstruction(fig, gs, fpca_results, kpi)
            
            dashboard_title = title or f'Functional Data Analysis: {self._format_kpi_name(kpi)}'
            plt.suptitle(f'{dashboard_title}\nSmoothing: {smoothing_method}, '
                        f'Basis: {basis_type}, Components: {n_basis}→{n_components}', 
                        fontsize=16, fontweight='bold')
            
            return fig, fpca_results
        
        cache_key = f"fda_dashboard_{kpi}_{smoothing_method}_{basis_type}_{n_basis}_{n_components}"
        return self._get_cached_or_compute(cache_key, compute_dashboard)
    
    def _plot_trajectories(self, fig, gs, fpca_results, kpi):
        """Plot raw vs smoothed trajectories."""
        ax1 = fig.add_subplot(gs[0, :2])
        colors = self._get_color_palette(self.n_runs)
        
        for i, (run_name, history) in enumerate(self.results_dict.items()):
            generations = np.arange(len(history.generation))
            raw_values = getattr(history, kpi)
            smooth_gen, smooth_vals = fpca_results['smoothed_data'][run_name]
            
            ax1.plot(generations, raw_values, alpha=0.3, color=colors[i], linewidth=1)
            ax1.plot(smooth_gen, smooth_vals, color=colors[i], linewidth=2, 
                    label=f'{run_name} (smoothed)')
        
        ax1.set_title(f'Raw vs Smoothed {self._format_kpi_name(kpi)} Trajectories', fontweight='bold')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel(self._format_kpi_name(kpi))
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    def _plot_pc_functions(self, fig, gs, fpca_results):
        """Plot principal component functions."""
        ax2 = fig.add_subplot(gs[0, 2:])
        time_grid = fpca_results['time_grid']
        mean_function = fpca_results['mean_function']
        
        ax2.plot(time_grid, mean_function, 'k-', linewidth=3, label='Mean Function')
        pc_colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, pc_func in enumerate(fpca_results['pc_functions']):
            var_explained = fpca_results['explained_variance_ratio'][i] * 100
            ax2.plot(time_grid, mean_function + 2*np.sqrt(var_explained/100)*pc_func, 
                    color=pc_colors[i], linestyle='--', 
                    label=f'PC{i+1} (+2σ, {var_explained:.1f}%)')
            ax2.plot(time_grid, mean_function - 2*np.sqrt(var_explained/100)*pc_func,
                    color=pc_colors[i], linestyle='--')
        
        ax2.set_title('Principal Component Functions', fontweight='bold')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Function Value')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    def _plot_pc_analysis(self, fig, gs, fpca_results):
        """Plot PC scores and explained variance."""
        # PC scores scatter plot
        ax3 = fig.add_subplot(gs[1, 0])
        pc_scores = fpca_results['pc_scores']
        scatter = ax3.scatter(pc_scores[:, 0], pc_scores[:, 1], 
                            c=range(len(pc_scores)), cmap='viridis', s=60)
        ax3.set_xlabel(f'PC1 ({fpca_results["explained_variance_ratio"][0]*100:.1f}%)')
        ax3.set_ylabel(f'PC2 ({fpca_results["explained_variance_ratio"][1]*100:.1f}%)')
        ax3.set_title('PC Scores (Trajectory Clustering)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add trajectory labels
        for i, name in enumerate(fpca_results['trajectory_names']):
            ax3.annotate(f'R{i+1}', (pc_scores[i, 0], pc_scores[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Explained variance plot
        ax4 = fig.add_subplot(gs[1, 1])
        explained_var = fpca_results['explained_variance_ratio'] * 100
        cumulative_var = np.cumsum(explained_var)
        
        bars = ax4.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, color='skyblue')
        ax4.plot(range(1, len(explained_var) + 1), cumulative_var, 'ro-', linewidth=2)
        ax4.set_xlabel('Principal Component')
        ax4.set_ylabel('Explained Variance (%)')
        ax4.set_title('Variance Explained by PCs', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, explained_var)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', fontsize=9)
    
    def _plot_basis_functions(self, fig, gs, fpca_results):
        """Plot basis functions."""
        ax5 = fig.add_subplot(gs[1, 2:])
        basis_matrix = fpca_results['basis_matrix']
        time_grid = fpca_results['time_grid']
        
        for i in range(min(8, basis_matrix.shape[1])):
            ax5.plot(time_grid, basis_matrix[:, i], label=f'{fpca_results["basis_type"].title()} {i+1}')
        
        ax5.set_title(f'{fpca_results["basis_type"].title()} Basis Functions (First 8)', fontweight='bold')
        ax5.set_xlabel('Generation')
        ax5.set_ylabel('Basis Value')
        ax5.grid(True, alpha=0.3)
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    def _plot_reconstruction(self, fig, gs, fpca_results, kpi):
        """Plot reconstruction quality."""
        ax6 = fig.add_subplot(gs[2, :2])
        basis_matrix = fpca_results['basis_matrix']
        time_grid = fpca_results['time_grid']
        
        # Show reconstruction for first few trajectories
        for i in range(min(3, len(fpca_results['trajectory_names']))):
            run_name = fpca_results['trajectory_names'][i]
            smooth_gen, smooth_vals = fpca_results['smoothed_data'][run_name]
            
            # Reconstruct using PC coefficients
            reconstructed = (fpca_results['mean_function'] + 
                           basis_matrix @ (fpca_results['pc_scores'][i] @ fpca_results['pca_model'].components_))
            
            ax6.plot(smooth_gen, smooth_vals, 'o-', alpha=0.7, markersize=3, 
                    label=f'{run_name} (Original)')
            ax6.plot(time_grid, reconstructed, '--', linewidth=2,
                    label=f'{run_name} (Reconstructed)')
        
        ax6.set_title('Reconstruction Quality (First 3 Trajectories)', fontweight='bold')
        ax6.set_xlabel('Generation')
        ax6.set_ylabel(self._format_kpi_name(kpi))
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    def create_comparison_dashboard(self, 
                                  kpi: str = 'best_fitness',
                                  confidence_intervals: bool = False,
                                  show_statistics: bool = True,
                                  title: Optional[str] = None) -> plt.Figure:
        """
        Create comparison dashboard with functional data analysis.
        
        This implementation provides functional analysis instead of basic comparison.
        For standard comparison, use EngineComparator class.
        
        Args:
            kpi: KPI to compare across runs
            confidence_intervals: Whether to show confidence bands (used in FDA)
            show_statistics: Whether to display statistics
            title: Dashboard title
            
        Returns:
            Matplotlib figure with functional analysis dashboard
        """
        # Delegate to functional analysis dashboard
        fig, _ = self.create_functional_analysis_dashboard(
            kpi=kpi,
            title=title or f'Functional Analysis: {self._format_kpi_name(kpi)}'
        )
        return fig
    
    def clear_cache(self):
        """Clear all FDA caches."""
        super().clear_cache()
        self._smoothed_data_cache.clear()
        self._fpca_results_cache.clear()
        self._basis_cache.clear()