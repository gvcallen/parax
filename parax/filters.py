"""
General filter functions.
"""

from typing import Any, TypeGuard

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