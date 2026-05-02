from typing import Any, TypeGuard

import jax.numpy as jnp
import jax
import equinox as eqx

from parax.variables import AbstractVariable, AbstractFreeVariable
from parax.constraints import AbstractConstraint
from distreqx.distributions import AbstractDistribution
from distreqx.bijectors import AbstractBijector


def is_variable(x: Any) -> TypeGuard[AbstractVariable]:
    """Returns True if `x` is a `parax.AbstractVariable`."""
    return isinstance(x, AbstractVariable)


def is_free_variable(x: Any) -> TypeGuard[AbstractFreeVariable]:
    """
    Returns True if `x` is a `parax.AbstractFreeVariable`.
    """
    return isinstance(x, AbstractFreeVariable)


def is_constraint(x: Any) -> TypeGuard[AbstractConstraint]:
    """Returns True if `x` is a `parax.AbstractConstraint`."""
    return isinstance(x, AbstractConstraint)


def is_distribution(x: Any) -> TypeGuard[AbstractDistribution]:
    """Returns True if `x` is a `distreqx.AbstractDistribution`."""
    return isinstance(x, AbstractDistribution)


def is_bijector(x: Any) -> TypeGuard[AbstractBijector]:
    """Returns True if `x` is a `distreqx.AbstractBijector`."""
    return isinstance(x, AbstractBijector)


def where_free_array(pytree: Any) -> Any:
    """
    Takes a PyTree and returns a boolean mask of the exact same structure.
    
    The mask is `True` ONLY for array-like leaves that are nested inside 
    an `AbstractFreeVariable` and arent inside non-free variables.
    All other leaves are `False`.
    """
    def mask_fn(x: Any) -> Any:
        if not is_variable(x):
            return False

        if is_free_variable(x):
            return jax.tree_util.tree_map(
                lambda leaf: True if eqx.is_inexact_array(leaf) else False,
                x
            )
        return False

    return jax.tree_util.tree_map(mask_fn, pytree, is_leaf=is_variable)


def when_free_array(pytree: Any, replace_val: Any) -> Any:
    """
    Takes a PyTree and replaces all array-like leaves nested inside 
    an `AbstractFreeVariable` that aren't inside non-free variables with `replace_val`. 
    
    All other leaves (outside free variables, or non-arrays) remain unchanged.
    """
    def _replace_fn(x: Any) -> Any:
        if not is_variable(x):
            return False

        if is_free_variable(x):
            return jax.tree_util.tree_map(
                lambda leaf: jnp.full_like(leaf, replace_val) if eqx.is_inexact_array(leaf) else leaf,
                x
            )
        return x

    return jax.tree_util.tree_map(_replace_fn, pytree, is_leaf=is_variable)
