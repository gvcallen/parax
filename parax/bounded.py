"""
An abstract interface for PyTrees that have "bounds".
"""
from abc import abstractmethod
from typing import TypeVar, Generic, Any, TypeGuard

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree


Base = TypeVar("Base")


class AbstractBounded(eqx.Module, Generic[Base]):
    """
    The abstract interface for a bounded PyTree.

    Makes use of the concept of a "base" space
    where bounded optimizers operate.
    
    Used as a type check for `parax.is_bounded`. 
    """
    @property
    @abstractmethod
    def base(self) -> Base:
        """Returns the current PyTree in base space."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def bounds(self) -> tuple[Base, Base]:
        """
        Returns the current PyTree bounds in base space.
        
        Must have a matching PyTree structure as `self.base`.
        """
        raise NotImplementedError    
  
    @abstractmethod
    def update(self, base: Base) -> "AbstractBounded":
        """
        Returns a new instance of this object updated with a new base PyTree.

        Args:
            base: The new base-space PyTree representing the updated state.

        Returns:
            A new instance of the bounded object, updated to reflect the new base.
        """
        pass


def is_bounded(x: Any) -> TypeGuard[AbstractBounded]:
    """
    Returns True if `x` is an instance of `parax.AbstractBounded`.
    """
    return isinstance(x, AbstractBounded)
    

def tree_base(tree: PyTree) -> PyTree:
    """
    Extracts a PyTree of base values from a tree. 
    
    Standard inexact arrays are left intact.

    Args:
        tree: The original PyTree tree potentially containing bounded nodes.

    Returns:
        A PyTree containing the extracted base values.
    """
    def _extract(x):
        if not is_bounded(x):
            return x
        return x.base

    return jax.tree_util.tree_map(_extract, tree, is_leaf=is_bounded)


def tree_lower(tree: PyTree) -> PyTree:
    """
    Extracts the lower bounds of a potentially bounded PyTree in base space. 
    
    Standard arrays default to (-inf, inf).

    Note that this function does not allow non-array/bounded leaf nodes.
    If you have leaves in your tree that are neither arrays nor derive
    from `parax.bounded.AbstractBounded`, be sure to mark
    them as static or filter them out using e.g. `eqx.filter` first.    

    Args:
        tree: The PyTree model to extract lower bounds from.

    Returns:
        A PyTree representing the lower bounds in base space.
    """
    def _get_lower(x):
        if is_bounded(x):
            return x.bounds[0]
        if eqx.is_inexact_array(x):
            return jnp.full_like(x, -jnp.inf)
        raise ValueError(f"Found a leaf node of type {type(x)} that is neither bounded nor an array in `parax.bounded.tree_lower`. Value: {x}")

    lower = jax.tree_util.tree_map(_get_lower, tree, is_leaf=is_bounded)
    return lower


def tree_upper(tree: PyTree) -> PyTree:
    """
    Extracts the upper bounds of a potentially bounded PyTree in base space. 
    
    Standard arrays default to (-inf, inf).

    Note that this function does not allow non-array/bounded leaf nodes.
    If you have leaves in your tree that are neither arrays nor derive
    from `parax.bounded.AbstractBounded`, be sure to mark
    them as static or filter them out using e.g. `eqx.filter` first.    

    Args:
        tree: The PyTree model to extract upper bounds from.

    Returns:
        A PyTree representing the upper bounds in base space.
    """
    def _get_upper(x):
        if is_bounded(x):
            return x.bounds[1]
        if eqx.is_inexact_array(x):
            return jnp.full_like(x, jnp.inf)
        raise ValueError(f"Found a leaf node of type {type(x)} that is neither bounded nor an array in `parax.bounded.tree_upper`. Value: {x}")

    upper = jax.tree_util.tree_map(_get_upper, tree, is_leaf=is_bounded)
    return upper


def tree_bounds(tree: PyTree) -> tuple[PyTree, PyTree]:
    """
    Extracts two PyTrees (lower and upper) representing the boundaries of 
    the base space. 
    
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


def tree_update(tree: PyTree, base_tree: PyTree) -> PyTree:
    """
    Takes an updated base-space PyTree and injects it back into the 
    original bounded tree structure using `parax.AbstractBounded.update`.

    Args:
        tree: The original PyTree tree containing the bounded nodes.
        base_tree: The updated PyTree containing the new base values.

    Returns:
        A new PyTree tree with its internal states reconstructed to reflect 
        the updated base values.
    """
    def _rebuild(orig, base):
        if is_bounded(orig):
            return orig.update(base)
        return base
        
    return jax.tree_util.tree_map(_rebuild, tree, base_tree, is_leaf=is_bounded)