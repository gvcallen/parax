"""
An abstract interface for PyTrees that have an associated probability distribution.
"""
from typing import Generic, TypeVar

from jaxtyping import PyTree
import jax
import jax.numpy as jnp
import equinox as eqx

from distreqx.distributions import AbstractDistribution

T = TypeVar("T")

class AbstractProbabilistic(eqx.Module, Generic[T]):
    """
    The abstract interface for a probabilistic PyTree.

    Probabilistic PyTrees have a probability distribution
    associated with them. That is, samples from the resultant
    distribution should match the PyTree structure of `self`.

    Currently Parax depends on `distreqx` for probability distributions,
    however this may be generalized in the future.
    
    Used as a type check for `parax.is_probabilistic`. 

    Attributes:
        distribution: The probability distribution associated with this PyTree node.
    """
    distribution: eqx.AbstractVar[AbstractDistribution]


def tree_distribution(tree: PyTree) -> PyTree:
    """
    Extracts the probability distributions of a PyTree.
    
    Standard arrays default to `distreqx.ImproperUniform`.

    Args:
        tree: The PyTree model containing probabilistic nodes or standard arrays.

    Returns:
        A PyTree of the exact same structure containing the extracted 
        probability distributions.
    """
    from parax.filters import is_probabilistic

    def _get_distribution(x):
        if is_probabilistic(x):
            return x.distribution
        if eqx.is_inexact_array(x):
            from distreqx.distributions import ImproperUniform
            return ImproperUniform(shape=jnp.shape(x))
        return x

    distributions = jax.tree_util.tree_map(_get_distribution, tree, is_leaf=is_probabilistic)
    return distributions