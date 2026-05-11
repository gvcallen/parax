"""
An abstract interface for PyTrees that have an associated probability distribution.
"""
from abc import abstractmethod
from typing import Generic, TypeVar, Any, TypeGuard, Self

from jaxtyping import PyTree
import jax
import jax.numpy as jnp
import equinox as eqx


from distreqx.bijectors import Inverse, Leafwise as LeafwiseBijector
from distreqx.distributions import AbstractDistribution, Joint, Transformed
from parax.constraints import AbstractConstraint, Leafwise as LeafwiseConstraint, RealLine
from parax.constrainable import AbstractConstrainable
from parax.unwrappable import unwrap

T = TypeVar("T")


class AbstractProbabilistic(AbstractConstrainable[T]):
    """
    The abstract interface for a probabilistic PyTree.

    Probabilistic PyTrees have a probability distribution
    associated with them. That is, the event shape of the
    distribution matches the PyTree structure of `self`.

    Used as a type check for `parax.is_probabilistic`. 

    Attributes:
        distribution: The probability distribution associated with this PyTree node.
        constraint: Returns the active constraint of the PyTree.
        bounds: Returns the current PyTree bounds. Each must have a matching PyTree structure as `self`.
    """
    distribution: eqx.AbstractVar[AbstractDistribution]
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


def is_probabilistic(x: Any) -> TypeGuard[AbstractProbabilistic]:
    """
    Returns True if `x` is an instance of `parax.AbstractProbabilistic`.
    """
    return isinstance(x, AbstractProbabilistic)


def tree_distributions(tree: PyTree) -> PyTree:
    """
    Extracts the individual probability distributions of a PyTree.
    
    Standard arrays default to `distreqx.distributions.ImproperUniform`.

    Note that this function does not allow non-array/probabilistic leaf nodes.
    If you have leaves in your tree that are neither arrays nor derive
    from `parax.probabilistic.AbstractProbabilistic`, be sure to mark
    them as static or filter them out using e.g. `eqx.filter` first.

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
        raise ValueError(f"Found a leaf node of type {type(x)} that is neither probabilistic nor an array in `parax.probabilistic.tree_distributions`. Value: {x}")

    distributions = jax.tree.map(_get_distribution, tree, is_leaf=is_probabilistic)
    return distributions


def tree_constraints(tree: PyTree) -> PyTree:
    """
    Extracts the individual constraints of a PyTree.
    
    Standard arrays default to `parax.constraints.RealLine`.

    Note that this function does not allow non-array/probabilistic leaf nodes.
    If you have leaves in your tree that are neither arrays nor derive
    from `parax.probabilistic.AbstractProbabilistic`, be sure to mark
    them as static or filter them out using e.g. `eqx.filter` first.

    Args:
        tree: The PyTree containing probabilistic nodes or standard arrays.

    Returns:
        A PyTree of the exact same structure containing the extracted 
        constraints.
    """
    def _get_constraint(x):
        if is_probabilistic(x):
            return unwrap(x.constraint)
        if eqx.is_inexact_array(x):
            return RealLine(shape=jnp.shape(x))
        raise ValueError(f"Found a leaf node of type {type(x)} that is neither probabilistic nor an array in `parax.probabilistic.tree_constraint`. Value: {x}")

    constraints = jax.tree.map(_get_constraint, tree, is_leaf=is_probabilistic)
    return constraints


def tree_joint_distribution(tree: PyTree) -> Joint:
    """
    Extracts the single joint probability distributions of a PyTree.
    
    Wraps the output of `parax.probabilistic.tree_distributions`
    in a `distreqx.distributions.Joint` distribution to define
    a single distribution that matches the shape of `tree`.

    Note that this distribution is defined over the constrained space
    of any `parax.probabilistic.AbstractProbabilistic` variables.
    For interoperability with unconstrained algorithms, see
    `parax.probabilistic.tree_unconstrained_distribution`.

    Args:
        tree: The PyTree model containing probabilistic nodes or standard arrays.

    Returns:
        A single joint distribution whose event shape matches the structure of `tree`.
    """
    return Joint(tree_distributions(tree))


def tree_leafwise_constraint(tree: PyTree) -> LeafwiseConstraint:
    """
    Extracts the single leafwise constraint of a PyTree.
    
    Wraps the output of `parax.probabilistic.tree_constraint`
    in a `parax.constraints.LeafwiseConstraint` constraint to define
    a single constraint that matches the shape of `tree`.

    Args:
        tree: The PyTree model containing probabilistic nodes or standard arrays.

    Returns:
        A single constraint whose shape matches the structure of `tree`.
    """
    return LeafwiseConstraint(tree_constraints(tree))


def tree_leafwise_bijector(tree: PyTree) -> LeafwiseBijector:
    """
    Extracts the constraint bijector of a PyTree.
    
    Returns the bijector of `parax.probabilistic.tree_leafwise_constraint`,
    which transforms unconstrained values to constrained values.

    Args:
        tree: The PyTree model containing probabilistic nodes or standard arrays.

    Returns:
        A single bijector whose shape matches the structure of `tree`.
    """
    return tree_leafwise_constraint(tree).bijector    


def tree_unconstrained_distribution(tree: PyTree) -> Joint:
    """
    Extracts the joint, unconstrained probability distribution of a PyTree.
    
    Transforms the joint distribution of `parax.probabilistic.tree_joint_distribution`
    by the inverse bijector of `parax.probabilistic.tree_leafwise_constraint`.

    This is very useful for interoperability with unconstrained algorithms.

    Args:
        tree: The PyTree model containing probabilistic nodes or standard arrays.

    Returns:
        A single joint distribution for the unconstrained event space of `tree`.
    """
    joint = tree_joint_distribution(tree)
    bijector = tree_leafwise_bijector(tree)
    return Transformed(joint, Inverse(bijector))