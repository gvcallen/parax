"""
The general interface for PyTree wrapping using `parax.wrap`.
"""

from abc import abstractmethod
from typing import Generic, TypeVar, Any, Callable, TypeGuard, Self

import jax
import equinox as eqx
from jaxtyping import PyTree


T = TypeVar("T")


class AbstractWrappable(eqx.Module, Generic[T]):
    """An abstract class representing a PyTree node capable of wrapping another tree.

    This interface is the counterpart to `parax.AbstractUnwrappable`. It is 
    typically used to define how an object should reconstruct or "re-wrap" a 
    tree that was previously unwrapped.
    """

    @abstractmethod
    def wrap(self, tree: T) -> Self:
        """Wraps the provided tree inside this node's structure.

        Args:
            tree: The unwrapped tree or raw value to be wrapped.

        Returns:
            A new instance of this node containing the wrapped tree.
        """
        pass


def is_wrappable(x: Any) -> TypeGuard[AbstractWrappable]:
    """Checks if a given object is a wrappable node.

    Args:
        x: The object to check.

    Returns:
        True if `x` is an instance of `AbstractWrappable`, False otherwise.
    """
    return isinstance(x, AbstractWrappable)


def wrap(template_tree: PyTree, unwrapped_tree: PyTree, only_if: Callable[[Any], bool] = None) -> PyTree:
    """Recursively resolves `AbstractWrappable` nodes to reconstruct a wrapped PyTree.

    This function maps across a `template_tree` and an `unwrapped_tree` simultaneously. 
    Wrapping is performed outside-in (top-down), perfectly inverting the inside-out 
    (bottom-up) process of `parax.unwrap`. 
    
    **Note:** This function is typically used to re-wrap a PyTree that was previously 
    unwrapped via `parax.unwrap` and `parax.AbstractUnwrappable`.

    If the `only_if` predicate is provided, the wrapping process is conditionally gated.
    The tree is searched top-down, and wrapping only triggers for subtrees that 
    satisfy the condition. Once a node satisfies `only_if`, that entire subtree 
    is fully wrapped.

    Behavior with `only_if`:
        1. If `only_if(node)` is True: The node and all of its `AbstractWrappable` 
           descendants are fully wrapped.
        2. If `only_if(node)` is False: The node itself bypasses wrapping, but the 
           search continues recursively into its children.

    Args:
        template_tree: The original (or blueprint) PyTree containing `AbstractWrappable` nodes.
        unwrapped_tree: The PyTree containing the raw, unwrapped values.
        only_if: An optional predicate function `Callable[[Any], bool]`. If provided, 
            only subtrees evaluating to True (and their descendants) are wrapped.

    Returns:
        A new PyTree where the appropriate values from `unwrapped_tree` have been 
        wrapped using the template nodes.
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