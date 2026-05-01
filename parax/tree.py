"""
Tree mapping/unwrapping utilities for Parax.

These functions operate on PyTrees with `parax.Parameter` leaves.
They are design to map parametric structures to new PyTrees
that represent a specific value or metadata from the original PyTree.

For example, raw_values, physical values, the "fixed" attribute etc.
can be extracted. Further, joint probability distributions and constraints
can be extracted that represent a combined distribution/constraint
over the entire tree.

Note that these functions only allow parameter leaves. To operate
on mixed PyTrees, first partition using e.g. `eqx.partition(pytree, prx.is_param)`,
then perform the mapping, and then combine using `eqx.combine`.
"""
import dataclasses

import jax
from jaxtyping import PyTree, Array
from distreqx.distributions import Joint
from distreqx.bijectors import TreeMap

from parax.parameter import Parameter
from parax.filters import is_param
from parax.constraints import TreeConstraint


def raw_values(pytree: PyTree[Parameter]) -> PyTree[Array]:
    """
    Extracts raw values from a PyTree of parameters.

    Args:
        pytree: Any JAX PyTree containing `Parameter` leaves.

    Returns:
        A PyTree containing the `raw_value` arrays in place of `Parameter` wrappers. 
    """
    return jax.tree.map(lambda p: p.raw_value, pytree, is_leaf=is_param)


def base_values(pytree: PyTree[Parameter]) -> PyTree[Array]:
    """
    Extracts base values from a PyTree of parameters.

    Args:
        pytree: Any JAX PyTree containing `Parameter` leaves.

    Returns:
        A PyTree containing the `base_value` attribute in place of `Parameter` wrappers. 
    """
    return jax.tree.map(lambda p: p.base_value, pytree, is_leaf=is_param)


def physical_values(pytree: PyTree[Parameter]) -> PyTree[Array]:
    """
    Extracts physical values from a PyTree of parameters.

    Args:
        pytree: Any JAX PyTree containing `Parameter` leaves.

    Returns:
        A PyTree containing the `physical_value` attribute in place of `Parameter` wrappers. 
    """
    return jax.tree.map(lambda p: p.physical_value, pytree, is_leaf=is_param)


def scales(pytree: PyTree[Parameter]) -> PyTree[Array]:
    """
    Extracts the scale from every Parameter in the PyTree.

    Args:
        pytree: Any JAX PyTree containing `Parameter` objects.

    Returns:
        A PyTree where `Parameter` leaves are replaced by their `scale` property.
    """
    return jax.tree.map(lambda p: p.scale, pytree, is_leaf=is_param)


def fixed(pytree: PyTree[Parameter]) -> PyTree[Array]:
    """
    Extracts the fixed boolean flag from a parameter PyTree.

    Args:
        pytree: Any JAX PyTree containing `Parameter` objects.

    Returns:
        A PyTree containing the `fixed` boolean property in place of `Parameter` wrappers. 
    """
    return jax.tree.map(lambda p: p.fixed, pytree, is_leaf=is_param)


def distribution(pytree: PyTree[Parameter]) -> Joint:
    """
    Constructs a joint probability distribution from a parameter PyTree.

    The resultant distribution is of type `distreqx.distributions.Joint`
    and applies over the base values for all parameters in the pytree.
    It can be used to sample and calculate the log probability of values 
    over the entire tree structure simultaneously.

    Args:
        pytree: Any JAX PyTree containing `Parameter` objects.

    Returns:
        A `distreqx.distributions.Joint` distribution representing the prior.
    """
    distribution_tree = jax.tree.map(lambda p: p.distribution, pytree, is_leaf=is_param)
    return Joint(distribution_tree)


def constraint(pytree: PyTree[Parameter]) -> TreeConstraint:
    """
    Constructs a combined tree constraint from a parameter PyTree.

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
    constraint_tree = jax.tree.map(lambda p: p.constraint, pytree, is_leaf=is_param)
    return TreeConstraint(constraint_tree)


def raw_to_base_bijector(pytree: PyTree[Parameter]) -> TreeMap:
    """
    Constructs a combined tree bijector from a parameter PyTree.
    
    The resultant bijector is of type `distreqx.bijectors.TreeMap`
    and can be used to convert a PyTree of raw values
    to a PyTree of base values.

    Args:
        pytree: Any JAX PyTree containing `Parameter` leaves.

    Returns:
        A `TreeMap` bijector mapping from raw to base space.
    """
    bijector_tree = jax.tree.map(lambda p: p.raw_to_base_bijector, pytree, is_leaf=is_param)
    return TreeMap(bijector_tree)


def raw_to_physical_bijector(pytree: PyTree[Parameter]) -> TreeMap:
    """
    Constructs a combined tree bijector from a parameter PyTree.
    
    The resultant bijector is of type `distreqx.bijectors.TreeMap`
    and can be used to convert a PyTree of raw values
    to a PyTree of physical values.

    Args:
        pytree: Any JAX PyTree containing `Parameter` leaves.

    Returns:
        A `TreeMap` bijector mapping from raw to physical space.
    """
    bijector_tree = jax.tree.map(lambda p: p.raw_to_physical_bijector, pytree, is_leaf=is_param)
    return TreeMap(bijector_tree)


def base_to_physical_bijector(pytree: PyTree[Parameter]) -> TreeMap:
    """
    Constructs a combined tree bijector from a parameter PyTree.
    
    The resultant bijector is of type `distreqx.bijectors.TreeMap`
    and can be used to convert a PyTree of base values
    to a PyTree of physical values.

    Args:
        pytree: Any JAX PyTree containing `Parameter` leaves.

    Returns:
        A `TreeMap` bijector mapping from base to physical space.
    """
    bijector_tree = jax.tree.map(lambda p: p.base_to_physical_bijector, pytree, is_leaf=is_param)
    return TreeMap(bijector_tree)


def names(pytree: PyTree[Parameter]) -> PyTree[Array]:
    """
    Extracts the name from every Parameter in the PyTree.

    Args:
        pytree: Any JAX PyTree containing `Parameter` objects.

    Returns:
        A PyTree where `Parameter` leaves are replaced by their `name` property.
    """
    return jax.tree.map(lambda p: p.name, pytree, is_leaf=is_param)


def metadata(pytree: PyTree[Parameter]) -> PyTree[Array]:
    """
    Extracts the metadata from every Parameter in the PyTree.

    Args:
        pytree: Any JAX PyTree containing `Parameter` objects.

    Returns:
        A PyTree where `Parameter` leaves are replaced by their `metadata` property.
    """
    return jax.tree.map(lambda p: p.metadata, pytree, is_leaf=is_param)


def replace_raw_values(pytree: PyTree[Parameter], new_raw_values: PyTree[Array]) -> PyTree[Parameter]:
    """
    Injects new raw values into an existing Parameter PyTree.
    
    This is the inverse of `raw_values`. It preserves all metadata (scales, 
    constraints, names) while updating the underlying `raw_value` arrays.
    """
    return jax.tree.map(lambda p, v: dataclasses.replace(p, raw_value=v), pytree, new_raw_values, is_leaf=is_param)