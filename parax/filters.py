from typing import Any, TypeGuard
import jax
import equinox as eqx
from jaxtyping import PyTree

from parax.parameter import Parameter
from parax.constraints import AbstractConstraint
from distreqx.distributions import AbstractDistribution


def is_param(x: Any) -> TypeGuard[Parameter]:
    """Returns True if `x` is a `parax.Parameter`."""
    return isinstance(x, Parameter)


def is_free_param(x: Any) -> TypeGuard[Parameter]:
    """Returns True if `x` is a `parax.Parameter` AND is not fixed."""
    return is_param(x) and not x.fixed


def is_fixed_param(x: Any) -> TypeGuard[Parameter]:
    """Returns True if `x` is a `parax.Parameter` AND is not fixed."""
    return is_param(x) and x.fixed


def is_constraint(x: Any) -> TypeGuard[AbstractConstraint]:
    """Returns True if `x` is a `parax.AbstractConstraint`."""
    return isinstance(x, AbstractConstraint)


def is_distribution(x: Any) -> TypeGuard[AbstractDistribution]:
    """Returns True if `x` is a `distreqx.AbstractDistribution`."""
    return isinstance(x, AbstractDistribution)


def where_free_raw_value(pytree: PyTree) -> PyTree:
    """
    Generates a boolean filter mask identifying the `raw_value` of free parameters.

    Designed to isolate trainable/free components: the `raw_value` of any node 
    satisfying `is_free_param` becomes `True`, while fixed parameters and other 
    attributes become `False`.

    Intended for direct use as a filter spec with `eqx.partition` or `eqx.filter`
    for optimization and inference.

    Args:
        pytree: Any JAX PyTree.

    Returns:
        A PyTree of booleans matching the structure of `pytree`.
    """
    def build_mask(node: Any) -> Any:
        if is_free_param(node):
            false_param = jax.tree_util.tree_map(lambda _: False, node)
            return eqx.tree_at(lambda p: p.raw_value, false_param, True)
        return False

    return jax.tree_util.tree_map(build_mask, pytree, is_leaf=is_param)


def when_free_raw_value(pytree: PyTree, replace_with: Any) -> PyTree:
    """
    Creates a PyTree structural mask where the `raw_value` of all free parameters 
    is replaced with `replace_with`. All other nodes are set to `None`.

    Ideal for generating `in_axes` specs for `eqx.filter_vmap`.

    Args:
        pytree: Any JAX PyTree.
        replace_with: The value to insert at the location of free `raw_value` arrays.

    Returns:
        A PyTree mask populated with `replace_with` and `None`.
    """
    return jax.tree_util.tree_map(
        lambda is_free: replace_with if is_free else None,
        where_free_raw_value(pytree),
    )