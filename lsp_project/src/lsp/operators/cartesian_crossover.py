"""Cartesian Subgraph Crossover utilities.

Provides two algorithms for computing the active subgraph mask:
1. get_active_mask_scan: O(N) reverse-scan (memory efficient)
2. get_active_mask_matrix: O(N^2) triangular solve (matrix algebra)
"""

import chex
import jax
import jax.numpy as jnp

def get_active_mask_scan(args: chex.Array, out_nodes: chex.Array, num_inputs: int, num_nodes: int) -> chex.Array:
    """Computes the active nodes mask using an O(N) reverse jax.lax.scan.
    
    Args:
        args: Array of shape (num_nodes, max_arity) containing the argument pointers for each node.
        out_nodes: Array of shape (num_outputs,) containing pointers to the final output nodes.
        num_inputs: Number of primary inputs.
        num_nodes: Number of computational nodes.
        
    Returns:
        Boolean mask of shape (num_nodes,) indicating which nodes are active.
    """
    N = num_inputs + num_nodes
    
    # Initialize mask with output nodes
    mask = jnp.zeros(N, dtype=jnp.bool_)
    mask = mask.at[out_nodes].set(True)
    
    # Reverse scan over the computational nodes
    def step_fn(current_mask, node_info):
        node_idx, node_args = node_info
        is_active = current_mask[node_idx]
        
        # Only propagate if the current node is active
        def _activate(m):
            return m.at[node_args].set(True)
            
        new_mask = jax.lax.cond(is_active, _activate, lambda m: m, current_mask)
        return new_mask, None
        
    node_indices = jnp.arange(num_inputs, N)
    
    final_mask, _ = jax.lax.scan(
        step_fn,
        mask,
        (node_indices, args),
        reverse=True
    )
    
    return final_mask[num_inputs:]


def get_active_mask_matrix(args: chex.Array, out_nodes: chex.Array, num_inputs: int, num_nodes: int) -> chex.Array:
    """Computes the active nodes mask using an O(N^2) triangular solve.
    
    Args:
        args: Array of shape (num_nodes, max_arity) containing the argument pointers for each node.
        out_nodes: Array of shape (num_outputs,) containing pointers to the final output nodes.
        num_inputs: Number of primary inputs.
        num_nodes: Number of computational nodes.
        
    Returns:
        Boolean mask of shape (num_nodes,) indicating which nodes are active.
    """
    N = num_inputs + num_nodes
    
    # 1. Build Adjacency Matrix A (shape N x N)
    # A[i, j] = 1 if node i reads from node j
    A = jnp.zeros((N, N), dtype=jnp.float32)
    node_indices = jnp.arange(num_inputs, N)
    A = A.at[node_indices[:, None], args].set(1.0)
    
    # 2. Build Output Vector O (shape N x 1)
    O = jnp.zeros((N, 1), dtype=jnp.float32)
    O = O.at[out_nodes].set(1.0)
    
    # 3. Solve (I - A.T) X = O
    # Since node i only reads from j < i, A is strictly lower triangular.
    # Therefore A.T is strictly upper triangular.
    I = jnp.eye(N, dtype=jnp.float32)
    
    # solve_triangular expects 2D matrices, lower=False because (I - A.T) is upper triangular
    X = jax.scipy.linalg.solve_triangular(I - A.T, O, lower=False)
    
    # 4. Any node with a reachability value > 0 is part of the active subgraph
    active_mask = (X > 0.0).flatten()
    
    return active_mask[num_inputs:]

from typing import Any
from flax import struct
from malthusjax.core.genome.cartesian_genome import CartesianGenome, CartesianGenomeConfig
from malthusjax.operators.base import BaseCrossover

@struct.dataclass
class SubgraphCrossover(BaseCrossover[CartesianGenome, CartesianGenomeConfig]):
    """Subgraph Crossover for CGP.
    
    Identifies the active computational subgraph for both parents (mitigating positional bias 
    as described by Kalkreuth & Kocherovsky). Nodes are uniformly exchanged, but the 
    crossover preferentially operates on nodes that are active in at least one parent.
    
    Attributes:
        crossover_rate: Probability of swapping an active node from parent 2.
        mask_algorithm: Either 'scan' (O(N) time/memory) or 'matrix' (O(N^2) time/memory).
    """
    
    crossover_rate: float = struct.field(pytree_node=False, default=0.5)
    mask_algorithm: str = struct.field(pytree_node=False, default="scan")

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _generate_noise(
        self, keys: chex.Array, config: CartesianGenomeConfig, generation: int = 0
    ) -> chex.Array:
        """Tier 2 — Bernoulli gene-selection mask of shape (num_nodes,)."""
        return jax.random.bernoulli(keys[0], p=self.crossover_rate, shape=(config.num_nodes,))

    def _recombine_one(
        self,
        p1: CartesianGenome,
        p2: CartesianGenome,
        noise_data: chex.Array,
        config: CartesianGenomeConfig,
        **_kwargs: Any,
    ) -> CartesianGenome:
        """Tier 1 — Extract active subgraphs and uniformly recombine."""
        
        # 1. Compute Active Subgraphs
        if self.mask_algorithm == "scan":
            mask1 = get_active_mask_scan(p1.args, p1.out_nodes, config.num_inputs, config.num_nodes)
            mask2 = get_active_mask_scan(p2.args, p2.out_nodes, config.num_inputs, config.num_nodes)
        else:
            mask1 = get_active_mask_matrix(p1.args, p1.out_nodes, config.num_inputs, config.num_nodes)
            mask2 = get_active_mask_matrix(p2.args, p2.out_nodes, config.num_inputs, config.num_nodes)

        # 2. Crossover Mask
        cross = noise_data  # (num_nodes,) bool

        # 3. Only swap nodes if they are active in at least one parent.
        # This prevents the crossover from destructively shuffling junk/inactive regions 
        # which can accidentally activate and shatter the homologous active paths.
        active_union = mask1 | mask2
        take_p2 = cross & active_union
        
        offspring_ops = jnp.where(take_p2, p2.ops, p1.ops)
        offspring_args = jnp.where(take_p2[:, None], p2.args, p1.args)
        
        return p1.replace(ops=offspring_ops, args=offspring_args)
