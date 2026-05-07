"""
The general interface for PyTree unwrapping using `parax.unwrap`.
"""

from abc import abstractmethod
from typing import Generic, TypeVar, Any, Callable, TypeGuard, Self

import jax
import equinox as eqx
from jaxtyping import PyTree


T = TypeVar("T")


class AbstractWrappable(eqx.Module, Generic[T]):
    """An abstract class representing a PyTree node to be wrapped.
    """

    @abstractmethod
    def wrap(self, tree: T) -> Self:
        """Returns the unwrapped pytree, assuming no wrapped subnodes exist."""
        pass


def is_wrappable(x: Any) -> TypeGuard[AbstractWrappable]:
    """
    Returns True if `x` is an instance of `parax.AbstractWrappable`.
    """
    return isinstance(x, AbstractWrappable)


def wrap(template_tree: PyTree, unwrapped_tree: PyTree, only_if: Callable[[Any], bool] = None) -> PyTree:
    """
    Map across an unwrapped PyTree and a template PyTree, recursively resolving 
    all `AbstractWrappable` nodes to reconstruct the wrapped tree.

    Wrapping is done outside-in (top-down), which perfectly inverts the inside-out 
    (bottom-up) process of `unwrap`. A node is only wrapped if it meets the `only_if` 
    condition itself, or is a descendant of a node that met the condition.

    By default, all nodes are eligible for wrapping.
    """
    def _do_wrap(t_node, u_node, *, include_self: bool):
        def _map_fn(t_leaf, u_leaf):
            if not is_wrappable(t_leaf):
                return u_leaf
            
            partially_wrapped = t_leaf.wrap(u_leaf)
            return _do_wrap(t_leaf, partially_wrapped, include_self=False)

        def is_leaf(x):
            included = True if x is not t_node else include_self
            return is_wrappable(x) and included

        return jax.tree.map(_map_fn, t_node, u_node, is_leaf=is_leaf)

    def _search_and_wrap(t_node, u_node, *, include_self: bool):
        if include_self and only_if(t_node):
            return _do_wrap(t_node, u_node, include_self=True)

        def _map_fn(t_leaf, u_leaf):
            if only_if(t_leaf):
                return _do_wrap(t_leaf, u_leaf, include_self=True)

            if is_wrappable(t_leaf):
                # Bypass wrapping this node, but keep searching its children
                return _search_and_wrap(t_leaf, u_leaf, include_self=False)

            return u_leaf

        def is_leaf(x):
            included = True if x is not t_node else include_self
            return (is_wrappable(x) or only_if(x)) and included

        return jax.tree.map(_map_fn, t_node, u_node, is_leaf=is_leaf)

    if only_if is None:
        return _do_wrap(template_tree, unwrapped_tree, include_self=True)
    return _search_and_wrap(template_tree, unwrapped_tree, include_self=True)