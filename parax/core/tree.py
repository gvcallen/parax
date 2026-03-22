from typing import TypeVar, Any
import jax
import equinox as eqx
from parax.core.parameter import is_valid_param

T = TypeVar("T")

def partition(
    pytree: T, 
    include_fixed: bool = False, 
    include_arrays: bool = False,
    param_objects: bool = False,
) -> tuple[T, T]:
    """
    Partitions an arbitrary PyTree into (dynamic, static) halves.

    By default, this acts as a "strict" parameter partitioner: ONLY non-fixed 
    `Parameter` objects are routed to the dynamic tree. Raw JAX arrays are 
    treated as static data.
    
    Args:
        pytree: The PyTree to partition.
        include_fixed: If True, includes Parameter objects where fixed=True.
        include_arrays: If True, standard JAX floating-point arrays (not wrapped in a 
                        Parameter) are ALSO routed to the dynamic tree.
        param_objects: If True, the entire Parameter object is routed to the dynamic tree. 
                       If False, ONLY the .value array is routed to the dynamic tree.
                       
    Returns:
        tuple: (dynamic, static) PyTrees.
    """
    
    def build_mask(node):
        # 1. Parameter Logic
        if is_valid_param(node):
            if not include_fixed and getattr(node, "fixed", False):
                return False 
            
            if param_objects:
                return True
            else:
                false_param = jax.tree_util.tree_map(lambda _: False, node)
                return eqx.tree_at(lambda p: p.latent_value, false_param, True)
        
        # 2. Raw Array Logic (The Escape Hatch)
        if include_arrays and eqx.is_array(node):
            # Only treat floating point arrays as dynamic (standard JAX/Equinox behavior)
            return jax.numpy.issubdtype(node.dtype, jax.numpy.inexact)
            
        # 3. Everything else is static
        return False

    # Build the filter spec
    filter_spec = jax.tree_util.tree_map(build_mask, pytree, is_leaf=is_valid_param)
    
    # Preserve Parameter objects if requested
    leaf_fn = is_valid_param if param_objects else None
    
    return eqx.partition(pytree, filter_spec, is_leaf=leaf_fn)