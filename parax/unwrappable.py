"""
The general interface for PyTree unwrapping using `parax.unwrap`.
"""

from abc import abstractmethod
from typing import Generic, TypeVar, Callable, Any, Union, TypeGuard

import equinox as eqx
import jax
from jaxtyping import PyTree

T = TypeVar("T")


class AbstractUnwrappable(eqx.Module, Generic[T]):
    """An abstract class representing a deferred or wrapped PyTree node.

    Unwrappables act as placeholders within a PyTree. When `parax.unwrap` 
    is called on the tree, these nodes execute custom logic (like computation 
    or gradient stopping) and replace themselves with their output.
    """

    @abstractmethod
    def unwrap(self) -> T:
        """Returns the unwrapped pytree, assuming no wrapped subnodes exist."""
        pass


def is_unwrappable(x: Any) -> TypeGuard[AbstractUnwrappable]:
    """
    Returns True if `x` is an instance of `parax.AbstractUnwrappable`.
    """
    return isinstance(x, AbstractUnwrappable)


def unwrap(tree: Any, only_if: Callable[[Any], bool] = None) -> Any:
    """
    Map across a PyTree and conditionally resolve `AbstractUnwrappable` nodes.
    
    Unwrapping is done inside-out, but gated by the `unwrap_from` condition.
    A node is only unwrapped if it is a root itself, or if it is a 
    descendant of a root. Nodes above roots are not unwrapped unless 
    they also have a root as a parent.

    By default, all nodes are considered roots.
    """
    def _do_unwrap(node, *, include_self: bool):
        def _map_fn(leaf):
            if not is_unwrappable(leaf):
                return leaf
            # Recursively unwrap the children first (bottom-up)
            resolved_node = _do_unwrap(leaf, include_self=False)
            return resolved_node.unwrap()

        def is_leaf(x):
            included = True if x is not node else include_self
            return is_unwrappable(x) and included

        return jax.tree.map(_map_fn, node, is_leaf=is_leaf)
    
    def _search_and_unwrap(node, *, include_self: bool):
        if include_self and only_if(node):
            return _do_unwrap(node, include_self=True)
            
        def _map_fn(leaf):
            if only_if(leaf):
                return _do_unwrap(leaf, include_self=True)
            
            if is_unwrappable(leaf):
                return _search_and_unwrap(leaf, include_self=False)
            
            return leaf

        def is_leaf(x):
            included = True if x is not node else include_self
            return (is_unwrappable(x) or only_if(x)) and included

        return jax.tree.map(_map_fn, node, is_leaf=is_leaf)
    
    if only_if is None:  # fast path
        return _do_unwrap(tree, include_self=True)    
    return _search_and_unwrap(tree, include_self=True)


def as_unwrapped(tree: Union[T | PyTree[T]]) -> T:
    """
    Calls `tree.unwrap` once and only if it is an `AbstractUnwrappable`, otherwise returns it.

    Args:
        tree: The tree to (potentially) unwrap.

    Returns:
        The unwrapped tree.
    """
    if isinstance(tree, AbstractUnwrappable):
        return tree.unwrap()
    return tree