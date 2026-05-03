"""
Filter functions and TypeGuards for PyTree traversal.

These functions are specifically designed to be passed as the `is_leaf` 
argument to JAX and Equinox tree utilities (e.g., `jax.tree_map`, `eqx.partition`, 
`eqx.filter`). They use `typing.TypeGuard` to ensure static type checkers 
can correctly infer types after filtering.
"""

from typing import Any, TypeGuard

import equinox as eqx

from parax.constant import AbstractConstant
from parax.variables import AbstractVariable, AbstractConstrained
from parax.unwrappables import AbstractUnwrappable
from parax.constraints import AbstractConstraint
from parax.metadata import AbstractHasMetadata

from distreqx.distributions import AbstractDistribution
from distreqx.bijectors import AbstractBijector


def is_constant(x: Any) -> TypeGuard[AbstractConstant]:
    """
    Returns True if `x` is an instance of `parax.AbstractConstant`.
    
    Useful for partitioning a model to freeze standard parameters.
    """
    return isinstance(x, AbstractConstant)


def is_not_constant(x: Any) -> bool:
    """
    Returns True if `x` is not an instance of `parax.AbstractConstant`.
    
    Useful as the inverse filter for identifying free/active parameters.
    """
    return not isinstance(x, AbstractConstant)


def is_unwrappable(x: Any) -> TypeGuard[AbstractUnwrappable]:
    """
    Returns True if `x` is an instance of `parax.AbstractUnwrappable`.
    """
    return isinstance(x, AbstractUnwrappable)


def is_variable(x: Any) -> TypeGuard[AbstractVariable]:
    """
    Returns True if `x` is an instance of `parax.AbstractVariable`.
    """
    return isinstance(x, AbstractVariable)


def is_constrained(x: Any) -> TypeGuard[AbstractConstrained]:
    """
    Returns True if `x` is an instance of `parax.AbstractConstrained`.
    
    Useful for filtering out parameters that require physical bounded 
    optimization steps.
    """
    return isinstance(x, AbstractConstrained)


def is_constraint(x: Any) -> TypeGuard[AbstractConstraint]:
    """
    Returns True if `x` is an instance of `parax.AbstractConstraint`.
    """
    return isinstance(x, AbstractConstraint)


def is_distribution(x: Any) -> TypeGuard[AbstractDistribution]:
    """
    Returns True if `x` is an instance of `distreqx.AbstractDistribution`.
    """
    return isinstance(x, AbstractDistribution)


def is_bijector(x: Any) -> TypeGuard[AbstractBijector]:
    """
    Returns True if `x` is an instance of `distreqx.AbstractBijector`.
    """
    return isinstance(x, AbstractBijector)


def has_metadata(x: Any) -> TypeGuard[AbstractHasMetadata]:
    """Returns True if `x` is an instance of  `distreqx.AbstractHasMetadata`."""
    return isinstance(x, AbstractHasMetadata)