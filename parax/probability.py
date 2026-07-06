"""
An abstract interface for PyTrees that have an associated probability distribution.
"""
from typing import TypeVar, Any, TypeGuard

from jaxtyping import PyTree
import jax
import jax.numpy as jnp
import equinox as eqx


from distreqx.distributions import AbstractDistribution, Transformed, Normal, Uniform
from parax.constraints import AbstractConstraint, AbstractConstrained

T = TypeVar("T")


class AbstractProbabilistic(AbstractConstrained[T]):
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
    from `parax.probability.AbstractProbabilistic`, be sure to mark
    them as static or filter them out using e.g. `eqx.filter` first.

    Args:
        tree: The PyTree containing probabilistic nodes or standard arrays.

    Returns:
        A PyTree of the exact same structure containing the extracted 
        probability distributions.
    """
    from distreqx.distributions import ImproperUniform
    from parax.wrappers import as_unwrapped

    def _get_distribution(path, x):
        if is_probabilistic(x):
            return as_unwrapped(x.distribution)
        if eqx.is_inexact_array(x):
            return ImproperUniform(shape=jnp.shape(x))
        raise ValueError(f"Found a leaf node of type {type(x)} that is neither probabilistic nor an array in `parax.probability.tree_distributions`. Value: {x}, path: {path}")

    distributions = jax.tree.map_with_path(_get_distribution, tree, is_leaf=is_probabilistic)
    return distributions


def tree_joint_distribution(tree: PyTree) -> AbstractDistribution:
    """
    Extracts the single joint probability distributions of a PyTree.
    
    Wraps the output of `parax.probability.tree_distributions`
    in a `distreqx.distributions.Joint` distribution to define
    a single distribution that matches the shape of `tree`.

    Note that this distribution is defined over the constrained space
    of any `parax.probability.AbstractProbabilistic` variables.
    For interoperability with unconstrained algorithms, see
    `parax.probability.tree_unconstrained_distribution`.

    Args:
        tree: The PyTree model containing probabilistic nodes or standard arrays.

    Returns:
        A single joint distribution whose event shape matches the structure of `tree`.
    """
    from distreqx.distributions import Joint
    return Joint(tree_distributions(tree))


def tree_unconstrained_distribution(tree: PyTree) -> AbstractDistribution:
    """
    Extracts the joint, unconstrained probability distribution of a PyTree.
    
    Transforms the joint distribution of `parax.probability.tree_joint_distribution`
    by the inverse bijector of `parax.probability.tree_leafwise_constraint`.

    This is very useful for interoperability with unconstrained algorithms.

    Args:
        tree: The PyTree model containing probabilistic nodes or standard arrays.

    Returns:
        A single joint distribution for the unconstrained event space of `tree`.
    """
    from parax.constraints import tree_leafwise_constraint
    
    joint = tree_joint_distribution(tree)
    bijector = tree_leafwise_constraint(tree).bijector
    
    from distreqx.bijectors import Inverse
    return Transformed(joint, Inverse(bijector))


def truncate_distribution(
    dist: AbstractDistribution, 
    new_lower: Any, 
    new_upper: Any
) -> AbstractDistribution:
    """
    Attempts to return a truncated version of the distribution.
    Raise an exception if unavailable.
    """
    if isinstance(dist, Normal):
        from distreqx.distributions import TruncatedNormal
        return TruncatedNormal(loc=dist.loc, scale=dist.scale, low=new_lower, high=new_upper)
    elif isinstance(dist, Uniform):
        return Uniform(new_lower, new_upper)
    else:
        raise ValueError(f"Distribution of type {type(dist)} cannot be truncated")
    

def _is_unwrappable_probabilistic(x):
    from parax.wrappers import is_unwrappable
    return is_probabilistic(x) and is_unwrappable(x) 

def is_leaf(x):
    """
    Defines the tree traversal boundaries for probabilistic partitioning.

    In the Parax ecosystem, certain custom nodes (like unwrappable probabilistic 
    priors or posteriors) contain internal metadata. If Equinox traverses inside 
    these nodes, it will strip their differentiable arrays away from their metadata, 
    causing structural mismatches during recombination.

    This function tells JAX/Equinox to treat these specific Parax objects as 
    opaque, indivisible leaves. 

    Args:
        x: Any node encountered during PyTree traversal.

    Returns:
        bool: True if the node should NOT be traversed into. Matches:
            1. Unwrappable probabilistic nodes (preserves their wrapper structure).
            2. Constant nodes (protects static configuration objects).
    """
    from parax.constants import is_constant
    return _is_unwrappable_probabilistic(x) or is_constant(x)

def is_dynamic(x):
    """
    Identifies parameters that should be updated during probabilistic inference.

    This function acts as the primary filter for `eqx.partition`, determining 
    which nodes are routed to the `dynamic` (differentiable/optimizable) tree 
    and which are left behind in the `static` tree.

    Because `parax.probability.is_leaf` protects unwrappable nodes from being 
    split open, this function captures those nodes completely whole, allowing 
    them to be safely unwrapped *after* partitioning. Therefore, if you would
    like to pass the full, wrapped nodes through a jit boundary, you should
    include additional conditions or partitioning steps.

    Args:
        x: Any leaf node in the PyTree (as defined by `is_leaf`).

    Returns:
        bool: True if the node is meant for the inference engine. Matches:
            1. Standard JAX inexact arrays (floating-point tensors).
            2. Entire unwrappable probabilistic nodes.
        Note: Explicitly returns False for `parax.constant` nodes, forcing 
        them into the static tree.
    """    
    from parax.constants import is_constant
    if is_constant(x): 
        return False
    if _is_unwrappable_probabilistic(x): 
        return True
    return eqx.is_inexact_array(x)