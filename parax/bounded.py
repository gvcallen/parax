"""
An abstract interface for PyTrees that have "bounds".

Bounded trees are allowed to work in three distinct spaces which
may or not may be equivalent:
- The "raw" space, where the original model stores its values
- The "base" space, where bounded optimizers operate
- The "unwrapped" space, where computations take plase

"""
from abc import abstractmethod
import jax
import jax.numpy as jnp
from typing import TypeVar, Generic
import equinox as eqx
from jaxtyping import PyTree

from parax.unwrappables import AbstractUnwrappable, as_frozen

T = TypeVar("T")

class AbstractBounded(eqx.Module, Generic[T]):
    """
    The abstract interface for a bounded PyTree.

    Bounded PyTrees are expose a base value of type "T"
    which defines the space which bounds are defined in,
    as well as where bounded optimizers should operate.
    
    This allows for underlying and unwrapped representations
    of the PyTree to be decoupled from the optimization space
    which a bounded solver might operate in.

    Used as a type check for `parax.is_bounded`. 
    """
    @property
    @abstractmethod
    def base(self) -> T:
        """
        Returns the base space value.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def bounds(self) -> tuple[T, T]:
        """
        Returns the PyTree's bounds.

        Must have the same underlying treedef as `self.base`.
        """
        raise NotImplementedError    
  
    @abstractmethod
    def evaluate_base(self, base: T) -> PyTree:
        """Evaluates the final output value from a given base value."""
        pass
    
    @abstractmethod
    def replace_base(self, base: T) -> "AbstractBounded":
        """
        Returns a new instance of this object, immutably updated 
        with the new base value.
        """
        pass
    

def tree_base(model: PyTree) -> PyTree:
    """
    Extracts a PyTree of base values from a model. 
    Standard inexact arrays are left intact.
    """
    from parax.filters import is_bounded
    def _extract(x):
        if not is_bounded(x):
            return x
        return x.base

    return jax.tree_util.tree_map(_extract, model, is_leaf=is_bounded)


def tree_bounds(model: PyTree) -> tuple[PyTree, PyTree]:
    """
    Extracts two PyTrees (lower and upper) representing the boundaries of 
    the base space. Standard arrays default to (-inf, inf).
    """
    from parax.filters import is_bounded

    def _get_lower(x):
        if is_bounded(x):
            return x.bounds[0]
        if eqx.is_inexact_array(x):
            return jnp.full_like(x, -jnp.inf)
        return x

    def _get_upper(x):
        if is_bounded(x):
            return x.bounds[1]
        if eqx.is_inexact_array(x):
            return jnp.full_like(x, jnp.inf)
        return x

    lower = jax.tree_util.tree_map(_get_lower, model, is_leaf=is_bounded)
    upper = jax.tree_util.tree_map(_get_upper, model, is_leaf=is_bounded)
    
    return lower, upper


def tree_replace_base(model: PyTree, base_model: PyTree) -> PyTree:
    """
    Takes an updated base-space PyTree and injects it back into the 
    original bounded model structure using `replace_from_base`.
    """
    from parax.filters import is_bounded

    def _rebuild(orig, base):
        if is_bounded(orig):
            return orig.replace_base(base)
        return base
        
    return jax.tree_util.tree_map(_rebuild, model, base_model, is_leaf=is_bounded)


def tree_evaluate_base(base_model: PyTree, bounded_model: PyTree) -> PyTree:
    """
    Takes a base-space PyTree and evaluates it.
    """
    from parax.filters import is_bounded

    def evaluate_base(orig_node, base_node):
        from parax.filters import is_bounded
        if is_bounded(orig_node):
            return orig_node.evaluate_base(base_node)
        return base_node
        
    evaluated_model = jax.tree_util.tree_map(
        evaluate_base, bounded_model, base_model, is_leaf=is_bounded
    )
    return evaluated_model