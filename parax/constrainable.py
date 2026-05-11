"""
An abstract interface for PyTrees that support dynamic constraints.
"""
from abc import abstractmethod
from typing import TypeVar, Any, TypeGuard, Self

import jax
import equinox as eqx
from jaxtyping import PyTree

from parax.bounded import AbstractBounded
from parax.constraints import AbstractConstraint, RealLine

T = TypeVar("Value")


class AbstractConstrainable(AbstractBounded[T]):
    """
    The abstract interface for a constrainable PyTree.

    Variables implementing this interface support the dynamic injection
    and updating of constraints.

    Used as a type check for `parax.is_constrainable`.

    Attributes:
        constraint: Returns the active constraint of the PyTree.
        bounds: Returns the current PyTree bounds. Each must have a matching PyTree structure as `self`.
    """
    constraint: eqx.AbstractVar[AbstractConstraint]
    bounds: eqx.AbstractVar[tuple[T, T]]

    @abstractmethod
    def constrain(self, constraint: AbstractConstraint) -> Self:
        """
        Returns a new instance of the PyTree with the updated constraint,
        ensuring internal state (like unconstrained raw values) is 
        recalculated if necessary.

        Args:
            constraint: The new constraint to apply.

        Returns:
            A new instance of the constrainable PyTree.
        """
        raise NotImplementedError


def is_constrainable(x: Any) -> TypeGuard[AbstractConstrainable]:
    """
    Returns True if `x` is an instance of `parax.AbstractConstrainable`.
    
    Args:
        x: The object to check.
        
    Returns:
        True if `x` implements `AbstractConstrainable`, False otherwise.
    """
    return isinstance(x, AbstractConstrainable)


def tree_constraint(tree: PyTree) -> PyTree:
    """
    Extracts the constraints of a potentially constrainable PyTree. 
    
    Standard arrays default to `parax.constraints.RealLine`.

    Note that this function does not allow non-array/constrainable leaf nodes.
    If you have leaves in your tree that are neither arrays nor derive
    from `parax.constrainable.AbstractConstrainable`, be sure to mark
    them as static or filter them out using e.g. `eqx.filter` first.    

    Args:
        tree: The PyTree model to extract constraints from.

    Returns:
        A PyTree representing the active constraints.
    """
    def _get_constraint(x):
        if is_constrainable(x):
            return x.constraint
        if eqx.is_inexact_array(x):
            return RealLine(shape=x.shape)
        raise ValueError(
            f"Found a leaf node of type {type(x)} that is neither constrainable "
            f"nor an array in `parax.constrainable.tree_constraint`. Value: {x}"
        )

    return jax.tree_util.tree_map(_get_constraint, tree, is_leaf=is_constrainable)


def tree_constrain(tree: PyTree, constraints: PyTree) -> PyTree:
    """
    Applies a PyTree of constraints to a PyTree of constrainable objects.
    
    Standard arrays will be returned untouched if the matching constraint 
    is a `RealLine`. Attempting to apply a bounded constraint directly 
    to a standard array will raise an error.

    Args:
        tree: The PyTree model to update. Must have a matching PyTree structure 
            to `constraints`.
        constraints: A PyTree of `parax.AbstractConstraint` objects.

    Returns:
        A new PyTree with the constraints applied.
    """
    def _apply_constraint(x, c):
        if is_constrainable(x):
            return x.constrain(c)
        if eqx.is_inexact_array(x):
            if isinstance(c, RealLine):
                return x
            raise TypeError(
                "Cannot apply a bounded constraint to a raw JAX array directly. "
                "Ensure the array is wrapped in a `parax.Constrained` variable first."
            )
        raise ValueError(
            f"Found a leaf node of type {type(x)} that is neither constrainable "
            f"nor an array in `parax.constrainable.tree_constrain`. Value: {x}"
        )

    return jax.tree_util.tree_map(
        _apply_constraint, tree, constraints, is_leaf=is_constrainable
    )