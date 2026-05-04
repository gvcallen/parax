"""
An abstract interface for PyTrees that have "bounds".
"""
from abc import abstractmethod
import jax
import jax.numpy as jnp
from typing import TypeVar, Generic
import equinox as eqx
from jaxtyping import PyTree

Base = TypeVar("Base")
Physical = TypeVar("Physical")

class AbstractBounded(eqx.Module, Generic[Base]):
    """
    The abstract interface for a bounded PyTree.

    Makes use of the concept of a "base" space,
    where bounded optimizers operate, and a
    "physical" space, where the forward pass operates.
    
    Used as a type check for `parax.is_bounded`. 
    """
    @property
    @abstractmethod
    def base(self) -> Base:
        """Returns the current PyTree in base space."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def bounds(self) -> tuple[Base, Base]:
        """
        Returns the current PyTree bounds in base space.
        
        Must have a matching PyTree structure as `self.base`.
        """
        raise NotImplementedError    
  
    @abstractmethod
    def transform_to_physical(self, base: Base) -> Physical:
        """
        Converts a new base PyTree to a physical PyTree.

        Args:
            base: The base-space PyTree to transform.

        Returns:
            The transformed PyTree in the physical (forward-pass) space.
        """
        pass
    
    @abstractmethod
    def update_from_base(self, base: Base) -> "AbstractBounded":
        """
        Returns a new instance of this object with a new base PyTree.

        Args:
            base: The new base-space PyTree representing the updated state.

        Returns:
            A new instance of the bounded object, updated to reflect the new base.
        """
        pass
    

def tree_base(model: PyTree) -> PyTree:
    """
    Extracts a PyTree of base values from a model. 
    
    Standard inexact arrays are left intact.

    Args:
        model: The original PyTree model potentially containing bounded nodes.

    Returns:
        A PyTree containing the extracted base values.
    """
    from parax.filters import is_bounded
    def _extract(x):
        if not is_bounded(x):
            return x
        return x.base

    return jax.tree_util.tree_map(_extract, model, is_leaf=is_bounded)


def tree_lower(tree: PyTree) -> PyTree:
    """
    Extracts the lower bounds of a potentially bounded PyTree in base space. 
    
    Standard arrays default to (-inf, inf).

    Args:
        tree: The PyTree model to extract lower bounds from.

    Returns:
        A PyTree representing the lower bounds in base space.
    """
    from parax.filters import is_bounded

    def _get_lower(x):
        if is_bounded(x):
            return x.bounds[0]
        if eqx.is_inexact_array(x):
            return jnp.full_like(x, -jnp.inf)
        return x

    lower = jax.tree_util.tree_map(_get_lower, tree, is_leaf=is_bounded)
    return lower


def tree_upper(tree: PyTree) -> PyTree:
    """
    Extracts the upper bounds of a potentially bounded PyTree in base space. 
    
    Standard arrays default to (-inf, inf).

    Args:
        tree: The PyTree model to extract upper bounds from.

    Returns:
        A PyTree representing the upper bounds in base space.
    """
    from parax.filters import is_bounded

    def _get_upper(x):
        if is_bounded(x):
            return x.bounds[1]
        if eqx.is_inexact_array(x):
            return jnp.full_like(x, jnp.inf)
        return x

    upper = jax.tree_util.tree_map(_get_upper, tree, is_leaf=is_bounded)
    return upper


def tree_bounds(tree: PyTree) -> tuple[PyTree, PyTree]:
    """
    Extracts two PyTrees (lower and upper) representing the boundaries of 
    the base space. 
    
    Standard arrays default to (-inf, inf).

    Args:
        tree: The PyTree model to extract bounds from.

    Returns:
        A tuple of two PyTrees `(lower_bounds, upper_bounds)`.
    """
    return tree_lower(tree), tree_upper(tree)


def tree_transform_to_physical(base_model: PyTree, original_model: PyTree) -> PyTree:
    """
    Takes a base-space PyTree and projects it to the external physical space.

    Args:
        base_model: The PyTree containing the base-space values (e.g., from an optimizer).
        original_model: The original PyTree model containing the `AbstractBounded` 
            nodes used to perform the transformation.

    Returns:
        A PyTree representing the fully evaluated model in the physical space.
    """
    from parax.filters import is_bounded

    def evaluate_base(orig_node, base_node):
        from parax.filters import is_bounded
        if is_bounded(orig_node):
            return orig_node.transform_to_physical(base_node)
        return base_node
        
    evaluated_model = jax.tree_util.tree_map(
        evaluate_base, original_model, base_model, is_leaf=is_bounded
    )
    return evaluated_model


def tree_update_from_base(model: PyTree, base_model: PyTree) -> PyTree:
    """
    Takes an updated base-space PyTree and injects it back into the 
    original bounded model structure using `update_from_base`.

    Args:
        model: The original PyTree model containing the bounded nodes.
        base_model: The updated PyTree containing the new base values.

    Returns:
        A new PyTree model with its internal states reconstructed to reflect 
        the updated base values.
    """
    from parax.filters import is_bounded

    def _rebuild(orig, base):
        if is_bounded(orig):
            return orig.update_from_base(base)
        return base
        
    return jax.tree_util.tree_map(_rebuild, model, base_model, is_leaf=is_bounded)