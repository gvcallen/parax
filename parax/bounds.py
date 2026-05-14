"""
An abstract interface for PyTrees that have "bounds".
"""
from typing import TypeVar, Generic, Any, TypeGuard

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree

from parax.wrappers import is_unwrappable
from parax.constants import is_constant


T = TypeVar("Value")


class AbstractBounded(eqx.Module, Generic[T]):
    """
    The abstract interface for a bounded PyTree.

    Used as a type check for `parax.is_bounded`. 

    Attributes:
        bounds: Returns the current PyTree bounds. Each must have a matching PyTree structure as `self`.
    """
    bounds: eqx.AbstractVar[tuple[T, T]]


def is_bounded(x: Any) -> TypeGuard[AbstractBounded]:
    """
    Returns True if `x` is an instance of `parax.AbstractBounded`.
    """
    return isinstance(x, AbstractBounded)


def tree_lower(tree: PyTree) -> PyTree:
    """
    Extracts the lower bounds of a potentially bounded PyTree. 
    
    Standard arrays default to (-inf, inf).

    Note that this function does not allow non-array/bounded leaf nodes.
    If you have leaves in your tree that are neither arrays nor derive
    from `parax.bounds.AbstractBounded`, be sure to mark
    them as static or filter them out using e.g. `eqx.filter` first.    

    Args:
        tree: The PyTree model to extract lower bounds from.

    Returns:
        A PyTree representing the lower bounds.
    """
    def _get_lower(path, x):
        if is_bounded(x):
            return x.bounds[0]
        if eqx.is_inexact_array(x):
            return jnp.full_like(x, -jnp.inf)
        raise ValueError(f"Found a leaf node of type {type(x)} that is neither bounded nor an array in `parax.bounds.tree_lower`. Value: {x}, path: {path}")

    lower = jax.tree.map_with_path(_get_lower, tree, is_leaf=is_bounded)
    return lower


def tree_upper(tree: PyTree) -> PyTree:
    """
    Extracts the upper bounds of a potentially bounded PyTree. 
    
    Standard arrays default to (-inf, inf).

    Note that this function does not allow non-array/bounded leaf nodes.
    If you have leaves in your tree that are neither arrays nor derive
    from `parax.bounds.AbstractBounded`, be sure to mark
    them as static or filter them out using e.g. `eqx.filter` first.    

    Args:
        tree: The PyTree model to extract upper bounds from.

    Returns:
        A PyTree representing the upper bounds.
    """
    def _get_upper(path, x):
        if is_bounded(x):
            return x.bounds[1]
        if eqx.is_inexact_array(x):
            return jnp.full_like(x, jnp.inf)
        raise ValueError(f"Found a leaf node of type {type(x)} that is neither bounded nor an array in `parax.bounds.tree_upper`. Value: {x}, path: {path}")

    upper = jax.tree.map_with_path(_get_upper, tree, is_leaf=is_bounded)
    return upper


def tree_bounds(tree: PyTree) -> tuple[PyTree, PyTree]:
    """
    Extracts two PyTrees (lower and upper) representing the boundaries. 
    
    Standard arrays default to (-inf, inf).

    Note that this function does not allow non-array/bounded leaf nodes.
    If you have leaves in your tree that are neither arrays nor derive
    from `parax.bounds.AbstractBounded`, be sure to mark
    them as static or filter them out using e.g. `eqx.filter` first.    

    Args:
        tree: The PyTree model to extract bounds from.

    Returns:
        A tuple of two PyTrees `(lower_bounds, upper_bounds)`.
    """
    return tree_lower(tree), tree_upper(tree)


def _is_unwrappable_bounded(x):
    return is_bounded(x) and is_unwrappable(x) 

def is_leaf(x):
    """
    Defines the tree traversal boundaries for bounded partitioning.

    In the Parax ecosystem, certain custom nodes (like unwrappable bounded 
    nodes) contain internal metadata. If Equinox traverses inside 
    these nodes, it will strip their differentiable arrays away from their metadata, 
    causing structural mismatches during recombination.

    This function tells JAX/Equinox to treat these specific Parax objects as 
    opaque, indivisible leaves. 

    Args:
        x: Any node encountered during PyTree traversal.

    Returns:
        bool: True if the node should NOT be traversed into. Matches:
            1. Unwrappable bounded nodes (preserves their wrapper structure).
            2. Constant nodes (protects static configuration objects).
    """
    return _is_unwrappable_bounded(x) or is_constant(x)

def is_dynamic(x):
    """
    Identifies parameters that should be updated during bounded inference.

    This function acts as the primary filter for `eqx.partition`, determining 
    which nodes are routed to the `dynamic` (differentiable/optimizable) tree 
    and which are left behind in the `static` tree.

    Because `parax.probability.is_leaf` protects unwrappable nodes from being 
    split open, this function captures those nodes completely whole, allowing 
    them to be safely unwrapped *after* partitioning. Therefore, if you would
    like to pass the full, wrapped nodes through a jit boundary, you should
    include additional conditions or partitioning steps.

    Args:
        x: Any leaf node in the PyTree (as defined by `is_leaf`).

    Returns:
        bool: True if the node is meant for the inference engine. Matches:
            1. Standard JAX inexact arrays (floating-point tensors).
            2. Entire unwrappable bounded nodes.
        Note: Explicitly returns False for `parax.constant` nodes, forcing 
        them into the static tree.
    """    
    if is_constant(x): 
        return False
    if _is_unwrappable_bounded(x): 
        return True
    return eqx.is_inexact_array(x)