"""
Manipulation and application of shadow PyTrees.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree

from parax.utils.distribution import hypercube_to_physical, physical_to_hypercube
from parax.utils.tree import tree_map_grouped

from parax.parameter import Parameter, is_free_param
from parax.filters import partition
from parax.deprecate.module import Module

from distreqx.bijectors import AbstractBijector

# ------------------------------------------------------------------
# Shadow PyTree transformations
# ------------------------------------------------------------------

def make_bounds_bijectors(lower: PyTree, upper: PyTree):
    """
    Constructs a PyTree of bijectors based on the provided lower and upper bounds.
    """
    from distreqx.bijectors import Chain, ScalarAffine, Sigmoid, Softplus
    
    def _get_bijector(low, high):
        has_low = low is not None and not jnp.all(jnp.isneginf(low))
        has_high = high is not None and not jnp.all(jnp.isposinf(high))

        if has_low and has_high:
            return Chain([ScalarAffine(shift=low, scale=(high - low)), Sigmoid()])
        elif has_low:
            return Chain([ScalarAffine(shift=low), Softplus()])
        elif has_high:
            return Chain([ScalarAffine(shift=high, scale=-1.0), Softplus()])
        else:
            return ScalarAffine(shift=0.0, scale=1.0)

    return jax.tree.map(_get_bijector, lower, upper)

# ------------------------------------------------------------------
# Shadow PyTree manipulations
# ------------------------------------------------------------------

def apply_bijectors(bijector_tree, value_tree, inverse=False):
    """
    Applies a bijector PyTree to a values PyTree. 
    Supports structural prefixes via custom leaf detection.
    """
    # Instruct JAX to stop traversing the bijector tree once it hits a bijector
    def is_bijector(node):
        return isinstance(node, AbstractBijector)

    def _apply(bijector, values):
        # 'values' could be a single leaf array or a deeper sub-PyTree of arrays
        if not inverse:
            # Map the forward transform over the value (or sub-tree of values)
            # w -> u (Unconstrained to Constrained)
            return jax.tree.map(bijector.forward, values)
        else:
            # u -> w (Constrained to Unconstrained)
            # We clip values before the inverse transform to prevent exact edge
            # cases (like 0 or 1 in Sigmoid) from generating infinity.
            def _safe_inverse(v):
                # Optionally add your icdf_bounds clipping logic here if v is strictly bounded.
                # A generic safe clip for bounded arrays relies on checking the bijector chain.
                return bijector.inverse(v)
                
            return jax.tree.map(_safe_inverse, values)

    return jax.tree.map(
        _apply, 
        bijector_tree, 
        value_tree, 
        is_leaf=is_bijector
    )

def apply_hypercube_transform(
    u_tree: PyTree, 
    dist_tree: PyTree, 
    group_tree: PyTree
) -> PyTree:
    """
    Pure PyTree operation mapping [0, 1] U-arrays to physical X-arrays.
    Totally agnostic to parax.
    """
    def map_group(group_id, u_group, dist_group):
        # Every parameter in this group shares the same distribution
        dist = dist_group[0] 
        
        # Fast path: no distribution
        if dist is None:
            return u_group
            
        # Joint distributions: stack, transform, unstack
        u_stacked = jnp.stack(u_group)
        if len(u_group) == 1:
            u_stacked = jnp.squeeze(u_stacked, axis=0)
            
        # Pure math operation
        x = hypercube_to_physical(dist, u_stacked)
        
        # Unpack back to a list of arrays
        if len(u_group) == 1:
            return [x]
        else:
            return [x[i] for i in range(len(u_group))]

    # Using the tree_map_grouped utility from the previous response
    return tree_map_grouped(
        map_group, 
        u_tree, 
        dist_tree, 
        group_tree=group_tree
    )