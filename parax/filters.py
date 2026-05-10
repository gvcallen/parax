"""
General filter functions.
"""

from typing import Any, TypeGuard, Callable

from jaxtyping import PyTree
import equinox as eqx

from distreqx.distributions import AbstractDistribution
from distreqx.bijectors import AbstractBijector


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


def remove(pytree: PyTree, condition: Callable[[Any], bool], *, stop_at: Callable[[Any], bool] = None) -> Any:
    """Removes nodes from a PyTree that match a given condition.

    Replaces matching nodes with None. Halts traversal at matching nodes,
    as well as any nodes matching `stop_at`.

    Args:
        pytree (PyTree): The input PyTree to filter.
        condition (Callable[[Any], bool]): A function that evaluates to True 
            for nodes that should be removed.
        stop_at (Callable[[Any], bool]): A function that evaluates to True 
            for nodes that shouldn't be traversed into.

    Returns:
        Any: A copy of the PyTree with the matched nodes replaced by None.
    """
    if stop_at is None:
        stop_at = lambda _: False

    return eqx.filter(
        pytree, 
        filter_spec=condition, 
        is_leaf=lambda x: stop_at(x) or condition(x), 
        inverse=True
    )