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
from parax.transforms import AbstractTransform
from parax.bounded import AbstractBounded
from parax.probabilistic import AbstractProbabilistic
from parax.variables import AbstractVariable
from parax.unwrappables import AbstractUnwrappable
from parax.constraints import AbstractConstraint
from parax.annotated import AbstractAnnotated

from distreqx.distributions import AbstractDistribution
from distreqx.bijectors import AbstractBijector


def is_constant(x: Any) -> TypeGuard[AbstractConstant]:
    """
    Returns True if `x` is an instance of `parax.AbstractConstant`.
    
    Useful as `is_leaf` when partitioning a model to freeze standard parameters.
    """
    return isinstance(x, AbstractConstant)


def is_annotated(x: Any) -> TypeGuard[AbstractAnnotated]:
    """
    Returns True if `x` is an instance of `parax.AbstractAnnotated`
    (i.e. has metadata).
    """
    return isinstance(x, AbstractAnnotated)


def is_variable(x: Any) -> TypeGuard[AbstractVariable]:
    """
    Returns True if `x` is an instance of `parax.AbstractVariable`.
    """
    return isinstance(x, AbstractVariable)


def is_param(x: Any) -> bool:
    """
    Returns True if `x` is an instance of `parax.AbstractVariable`
    or returns True for `eqx.is_inexact_array`.
    """
    return isinstance(x, AbstractVariable) or eqx.is_inexact_array(x)


def is_bounded(x: Any) -> TypeGuard[AbstractBounded]:
    """
    Returns True if `x` is an instance of `parax.AbstractBounded`.
    """
    return isinstance(x, AbstractBounded)


def is_probabilistic(x: Any) -> TypeGuard[AbstractProbabilistic]:
    """
    Returns True if `x` is an instance of `parax.AbstractProbabilistic`.
    """
    return isinstance(x, AbstractProbabilistic)


def is_unwrappable(x: Any) -> TypeGuard[AbstractUnwrappable]:
    """
    Returns True if `x` is an instance of `parax.AbstractUnwrappable`.
    """
    return isinstance(x, AbstractUnwrappable)


def is_constraint(x: Any) -> TypeGuard[AbstractConstraint]:
    """
    Returns True if `x` is an instance of `parax.AbstractConstraint`.
    """
    return isinstance(x, AbstractConstraint)


def is_transform(x: Any) -> TypeGuard[AbstractTransform]:
    """
    Returns True if `x` is an instance of `parax.AbstractTransform`.
    """
    return isinstance(x, AbstractTransform)


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