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
    is called on the tree, these nodes execute custom logic (such as delayed 
    computation, parameter injection, or gradient stopping) and replace 
    themselves with their output.
    """

    @abstractmethod
    def unwrap(self) -> T:
        """Evaluates and returns the underlying unwrapped PyTree node.
        
        Returns:
            The resolved underlying value, assuming no wrapped subnodes exist.
        """
        pass


def is_unwrappable(x: Any) -> TypeGuard[AbstractUnwrappable]:
    """Checks if a given object is an unwrappable node.
    
    Args:
        x: The object to check.
        
    Returns:
        True if `x` is an instance of `AbstractUnwrappable`, False otherwise.
    """
    return isinstance(x, AbstractUnwrappable)


def unwrap(tree: Any, only_if: Callable[[Any], bool] = None) -> Any:
    """Recursively resolves `AbstractUnwrappable` nodes within a PyTree.

    By default, unwrapping is performed inside-out (bottom-up) across the entire 
    tree. Every `AbstractUnwrappable` node is replaced by the result of its 
    `unwrap()` method.

    If the `only_if` predicate is provided, unwrapping is conditionally gated. 
    The tree is searched top-down, and unwrapping only triggers for subtrees 
    that satisfy the condition. Once a node satisfies `only_if`, that entire 
    subtree is fully unwrapped. 

    Behavior with `only_if`:
        1. If `only_if(node)` is True: The node and all of its `AbstractUnwrappable` 
           descendants are fully resolved.
        2. If `only_if(node)` is False: The node is left wrapped, but the search 
           continues recursively into its children.

    Args:
        tree: The PyTree to unwrap.
        only_if: An optional predicate function `Callable[[Any], bool]`. If provided, 
            only subtrees evaluating to True (and their descendants) are unwrapped.

    Returns:
        A new PyTree with the appropriate `AbstractUnwrappable` nodes resolved.
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


def as_unwrapped(tree: Union[T, PyTree[T]]) -> T:
    """Conditionally unwraps the root node if it is an `AbstractUnwrappable`.
    
    Unlike `unwrap`, this function does not recursively traverse the PyTree. 
    It only evaluates the top-level object.

    Args:
        tree: The tree or node to potentially unwrap.

    Returns:
        The unwrapped result if `tree` was an `AbstractUnwrappable`, 
        otherwise returns the original `tree` unmodified.
    """
    if isinstance(tree, AbstractUnwrappable):
        return tree.unwrap()
    return tree