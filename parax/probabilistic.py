"""
An abstract interface for PyTrees that have an associated probability distribution.
"""
from abc import abstractmethod
from typing import Generic, TypeVar, Any, TypeGuard

from jaxtyping import PyTree
import jax
import jax.numpy as jnp
import equinox as eqx

from distreqx.distributions import AbstractDistribution, Joint
from parax.unwrappables import unwrap

Base = TypeVar("Base")


class AbstractProbabilistic(eqx.Module, Generic[Base]):
    """
    The abstract interface for a probabilistic PyTree.

    Probabilistic PyTrees have a probability distribution
    associated with them. That is, samples from the resultant
    distribution should match the PyTree structure of `self`.

    Makes use of the concept of a "base" space
    where inference algorithms operate.
    
    Used as a type check for `parax.is_probabilistic`. 

    Attributes:
        distribution: The probability distribution associated with this PyTree node.
    """
    distribution: eqx.AbstractVar[AbstractDistribution]

    @property
    @abstractmethod
    def base(self) -> Base:
        """Returns the current PyTree in the probability base space."""
        raise NotImplementedError

    @abstractmethod
    def update(self, base: Base) -> "AbstractProbabilistic":
        """
        Returns a new instance of this object updated with a new base PyTree.

        Args:
            base: The new base-space PyTree representing the sampled state.

        Returns:
            A new instance of the probabilistic object, updated to reflect the new base.
        """
        pass


def is_probabilistic(x: Any) -> TypeGuard[AbstractProbabilistic]:
    """
    Returns True if `x` is an instance of `parax.AbstractProbabilistic`.
    """
    return isinstance(x, AbstractProbabilistic)


def tree_base(tree: PyTree) -> PyTree:
    """
    Extracts a PyTree of base values from a probabilistic tree. 
    
    Standard inexact arrays are left intact.

    Args:
        tree: The original PyTree tree potentially containing probabilistic nodes.

    Returns:
        A PyTree containing the extracted base values.
    """
    def _extract(x):
        if not is_probabilistic(x):
            return x
        return unwrap(x.base)

    return jax.tree_util.tree_map(_extract, tree, is_leaf=is_probabilistic)


def tree_distribution(tree: PyTree) -> PyTree:
    """
    Extracts the probability distributions of a PyTree.
    
    Standard arrays default to `distreqx.ImproperUniform`.

    Args:
        tree: The PyTree containing probabilistic nodes or standard arrays.

    Returns:
        A PyTree of the exact same structure containing the extracted 
        probability distributions.
    """
    from distreqx.distributions import ImproperUniform

    def _get_distribution(x):
        if is_probabilistic(x):
            return unwrap(x.distribution)
        if eqx.is_inexact_array(x):
            return ImproperUniform(shape=jnp.shape(x))
        raise ValueError(f"Found a leaf node of type {type(x)} that is neither static, probabilistic nor an array in `parax.probabilistic.tree_distribution`. Value: {x}")

    distributions = jax.tree_util.tree_map(_get_distribution, tree, is_leaf=is_probabilistic)
    return distributions


def tree_joint(tree: PyTree) -> Joint:
    """
    Extracts the single joint probability distributions of a PyTree.
    
    Wraps the output of `parax.probabilistic.tree_distribution`
    in a `distreqx.distributions.Joint` distribution to define
    a single distribution that matches the shape of `tree`.

    Args:
        tree: The PyTree model containing probabilistic nodes or standard arrays.

    Returns:
        A single joint distribution whose event shape matches the structure of `tree`.
    """
    return Joint(tree_distribution(tree))


def tree_update(tree: PyTree, base_tree: PyTree) -> PyTree:
    """
    Takes an updated base-space PyTree and injects it back into the 
    original probabilistic tree structure using `update`.

    Args:
        tree: The original PyTree tree containing the probabilistic nodes.
        base_tree: The updated PyTree containing the new base values.

    Returns:
        A new PyTree tree with its internal states reconstructed to reflect 
        the updated base values.
    """
    def _rebuild(orig, base):
        if is_probabilistic(orig):
            return orig.update(base)
        return base
        
    return jax.tree_util.tree_map(_rebuild, tree, base_tree, is_leaf=is_probabilistic)