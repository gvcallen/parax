import dataclasses
from typing import TypeVar, Any, Callable
import jax
import jax.numpy as jnp
import equinox as eqx
from parax.parameter import is_valid_param, is_free_param

T = TypeVar("T")


def where_free_param_value(pytree):
    """Generates a boolean filter mask identifying the `latent_value` of free parameters.

    This evaluates the given `pytree` and returns a boolean mask of the same structural 
    prefix. It is designed to isolate trainable/free components: the `latent_value` of 
    any node satisfying `is_free_param` becomes `True`, while fixed parameters, other 
    attributes (like the `fixed` boolean itself), and standard arrays become `False`.

    Intended for direct use as a filter spec with [`eqx.partition`][] or [`eqx.filter`][].

    **Arguments:**

    - `pytree`: Any JAX PyTree (typically your full Equinox model).

    **Returns:**

    A PyTree of booleans matching the structure of `pytree`.

    !!! Example

        ```python
        # Partition a model to extract only the free, trainable arrays
        mask = where_free_param_value(model)
        params, static = eqx.partition(model, mask)
        
        # 'params' now contains only the latent_values, everything else is None.
        ```
    """
    def build_mask(node):
        if is_free_param(node):
            false_param = jax.tree_util.tree_map(lambda _: False, node)
            return eqx.tree_at(lambda p: p.latent_value, false_param, True)
        return False

    return jax.tree_util.tree_map(build_mask, pytree, is_leaf=is_valid_param)


def when_free_param_value(pytree, replace_with: Any):
    """
    Creates a PyTree structural mask where the `latent_value` of all free parameters 
    is replaced with `replace_with`. All other nodes (fixed params, standard arrays) 
    are set to `None`.

    Ideal for generating `in_axes` specs for `eqx.filter_vmap` or sharding specs.

    !!! Example

        ```python
        # Map free parameter values to axis 0 for vmap, ignoring everything else
        axes_spec = map_free_param_values(model, replace_with=0)
        
        # Pass the pre-computed tree spec directly to in_axes
        vmapped_fn = eqx.filter_vmap(my_fn, in_axes=axes_spec)
        ```
    """
    return jax.tree_util.tree_map(
        lambda is_free: replace_with if is_free else None,
        where_free_param_value(pytree),
    )
    

def partition(
    pytree: T, 
    include_fixed: bool = False, 
    include_arrays: bool = False,
    param_objects: bool = False,
) -> tuple[T, T]:
    """
    Partitions an arbitrary PyTree into (dynamic, static) halves.

    By default, this acts as a "strict" parameter partitioner: ONLY non-fixed 
    [`~parax.Parameter`][] objects are routed to the dynamic tree. Raw JAX arrays are 
    treated as static data unless explicitly requested.
    
    Parameters
    ----------
    pytree : T
        The PyTree to partition.
    include_fixed : bool, default=False
        If True, includes [`~parax.Parameter`][] objects where `fixed=True`.
    include_arrays : bool, default=False
        If True, standard JAX floating-point arrays (not wrapped in a 
        [`~parax.Parameter`][]) are ALSO routed to the dynamic tree.
    param_objects : bool, default=False
        If True, the entire [`~parax.Parameter`][] object is routed to the dynamic tree. 
        If False, ONLY the underlying `.latent_value` array is routed to the dynamic tree.
                    
    Returns
    -------
    tuple of T
        A tuple containing `(dynamic, static)` PyTrees.
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