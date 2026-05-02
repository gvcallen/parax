"""
Tree mapping/unwrapping utilities for "Parameter PyTrees",
called "ParamTrees" for short.

These functions operate on PyTrees with `parax.Parameter` leaves.
They are design to map parametric structures to new PyTrees
that represent a specific value or metadata from the original PyTree.

For example, raw_values, physical values, the "fixed" attribute etc.
can be extracted. Further, joint probability distributions and constraints
can be extracted that represent a combined distribution/constraint
over the entire tree.

Note that these functions only allow parameter leaves. To operate
on mixed PyTrees, first partition using e.g.,
`eqx.partition(pytree, prx.is_free_param, is_leaf=prx.is_param)`,
then perform the mapping, and then re-combine using `eqx.combine`.
"""
import dataclasses
from typing import Sequence, Any
from functools import reduce

import jax
from jaxtyping import PyTree, Array
import equinox as eqx
from distreqx.distributions import Joint
from distreqx.bijectors import TreeMap

from parax.constrained import Param
from parax.filters import is_free_variable
from parax.constraints import TreeConstraint


def base_values(pytree: PyTree[Param]) -> PyTree[Array]:
    """
    Extracts physical values from every Parameter in a ParamTree.

    Args:
        pytree: Any JAX PyTree containing `Parameter` leaves.

    Returns:
        A PyTree containing the `base_value` attribute in place of `Parameter` wrappers. 
    """
    return jax.tree.map(lambda p: p.raw_value if is_free_variable(p) else p, pytree, is_leaf=is_free_variable)


def physical_values(pytree: PyTree[Param]) -> PyTree[Array]:
    """
    Extracts physical values from every Parameter in a ParamTree.

    Args:
        pytree: Any JAX PyTree containing `Parameter` leaves.

    Returns:
        A PyTree containing the `physical_value` attribute in place of `Parameter` wrappers. 
    """
    return jax.tree.map(lambda p: p.physical_value, pytree, is_leaf=is_free_variable)


def scales(pytree: PyTree[Param]) -> PyTree[Array]:
    """
    Extracts the scale from every Parameter in a ParamTree.

    Args:
        pytree: Any JAX PyTree containing `Parameter` objects.

    Returns:
        A PyTree where `Parameter` leaves are replaced by their `scale` property.
    """
    return jax.tree.map(lambda p: p.scale, pytree, is_leaf=is_free_variable)


def fixed(pytree: PyTree[Param]) -> PyTree[Array]:
    """
    Extracts the fixed boolean flag from a ParamTree.

    Args:
        pytree: Any JAX PyTree containing `Parameter` objects.

    Returns:
        A PyTree containing the `fixed` boolean property in place of `Parameter` wrappers. 
    """
    return jax.tree.map(lambda p: p.fixed, pytree, is_leaf=is_free_variable)


def distribution(pytree: PyTree[Param]) -> Joint:
    """
    Constructs a joint probability distribution from a ParamTree.

    The resultant distribution is of type `distreqx.distributions.Joint`
    and applies over the base values for all parameters in the pytree.
    It can be used to sample and calculate the log probability of values 
    over the entire tree structure simultaneously.

    Args:
        pytree: Any JAX PyTree containing `Parameter` objects.

    Returns:
        A `distreqx.distributions.Joint` distribution representing the prior.
    """
    distribution_tree = jax.tree.map(lambda p: p.distribution, pytree, is_leaf=is_free_variable)
    return Joint(distribution_tree)


def constraint(pytree: PyTree[Param]) -> TreeConstraint:
    """
    Constructs a combined tree constraint from a ParamTree.

    The resultant constraint is of type `parax.constraints.TreeConstraint`
    and applies over the raw values for all parameters in the pytree.

    Note that the constraint's bijector transforms raw values to base values
    and not phyiscal values. To transform straight to physical values,
    either multiply the base values by scales element-wise,
    or use `parax.tree.raw_to_physical_bijector` directly.

    Args:
        pytree: Any JAX PyTree containing `Parameter` objects.

    Returns:
        A `parax.constraints.TreeConstraint` containing the joint constraints.
    """
    constraint_tree = jax.tree.map(lambda p: p.constraint, pytree, is_leaf=is_free_variable)
    return TreeConstraint(constraint_tree)


def raw_to_base_bijector(pytree: PyTree[Param]) -> TreeMap:
    """
    Constructs a combined tree bijector from a ParamTree.
    
    The resultant bijector is of type `distreqx.bijectors.TreeMap`
    and can be used to convert a PyTree of raw values
    to a PyTree of base values.

    Args:
        pytree: Any JAX PyTree containing `Parameter` leaves.

    Returns:
        A `TreeMap` bijector mapping from raw to base space.
    """
    bijector_tree = jax.tree.map(lambda p: p.raw_to_base_bijector, pytree, is_leaf=is_free_variable)
    return TreeMap(bijector_tree)


def raw_to_physical_bijector(pytree: PyTree[Param]) -> TreeMap:
    """
    Constructs a combined tree bijector from a ParamTree.
    
    The resultant bijector is of type `distreqx.bijectors.TreeMap`
    and can be used to convert a PyTree of raw values
    to a PyTree of physical values.

    Args:
        pytree: Any JAX PyTree containing `Parameter` leaves.

    Returns:
        A `TreeMap` bijector mapping from raw to physical space.
    """
    bijector_tree = jax.tree.map(lambda p: p.raw_to_physical_bijector, pytree, is_leaf=is_free_variable)
    return TreeMap(bijector_tree)


def base_to_physical_bijector(pytree: PyTree[Param]) -> TreeMap:
    """
    Constructs a combined tree bijector from a ParamTree.
    
    The resultant bijector is of type `distreqx.bijectors.TreeMap`
    and can be used to convert a PyTree of base values
    to a PyTree of physical values.

    Args:
        pytree: Any JAX PyTree containing `Parameter` leaves.

    Returns:
        A `TreeMap` bijector mapping from base to physical space.
    """
    bijector_tree = jax.tree.map(lambda p: p.base_to_physical_bijector, pytree, is_leaf=is_free_variable)
    return TreeMap(bijector_tree)


def merge_update(pytrees: Sequence[PyTree[Param]]) -> PyTree:
    """Merge a sequence of ParamTree into one based on free parameters.
    
    This is useful to combine separate PyTrees obtained from fitting
    the same initial PyTrees with different free parameters.
    
    Starting with the first PyTree, any free parameter (fixed == False) 
    in the next PyTree overrides the parameter in the accumulated PyTree.
    
    Note that this function only works in eager mode.

    Parameters
    ----------
    pytrees : Sequence[PyTree]
        The PyTrees to merge. Must contain at least one PyTree.

    Returns
    -------
    PyTree
        The merged PyTree.
    """
    if not pytrees:
        raise ValueError("Must provide at least one PyTree to merge.")

    def merge_two(tree1: PyTree, tree2: PyTree) -> PyTree:
        def _merge_leaf(p1: Param, p2: Param) -> Param:
            return p2 if not p2.fixed else p1

        return jax.tree.map(
            _merge_leaf, 
            tree1, 
            tree2, 
            is_leaf=is_free_variable
        )

    return reduce(merge_two, pytrees)