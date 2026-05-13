"""
An abstract interface for PyTrees that have "bounds".
"""
from abc import abstractmethod
from typing import TypeVar, Generic, Any, TypeGuard

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree


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
    from `parax.bounded.AbstractBounded`, be sure to mark
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
        raise ValueError(f"Found a leaf node of type {type(x)} that is neither bounded nor an array in `parax.bounded.tree_lower`. Value: {x}, path: {path}")

    lower = jax.tree.map_with_path(_get_lower, tree, is_leaf=is_bounded)
    return lower


def tree_upper(tree: PyTree) -> PyTree:
    """
    Extracts the upper bounds of a potentially bounded PyTree. 
    
    Standard arrays default to (-inf, inf).

    Note that this function does not allow non-array/bounded leaf nodes.
    If you have leaves in your tree that are neither arrays nor derive
    from `parax.bounded.AbstractBounded`, be sure to mark
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
        raise ValueError(f"Found a leaf node of type {type(x)} that is neither bounded nor an array in `parax.bounded.tree_upper`. Value: {x}, path: {path}")

    upper = jax.tree.map_with_path(_get_upper, tree, is_leaf=is_bounded)
    return upper


def tree_bounds(tree: PyTree) -> tuple[PyTree, PyTree]:
    """
    Extracts two PyTrees (lower and upper) representing the boundaries. 
    
    Standard arrays default to (-inf, inf).

    Note that this function does not allow non-array/bounded leaf nodes.
    If you have leaves in your tree that are neither arrays nor derive
    from `parax.bounded.AbstractBounded`, be sure to mark
    them as static or filter them out using e.g. `eqx.filter` first.    

    Args:
        tree: The PyTree model to extract bounds from.

    Returns:
        A tuple of two PyTrees `(lower_bounds, upper_bounds)`.
    """
    return tree_lower(tree), tree_upper(tree)