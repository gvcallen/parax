"""
The general interface for PyTree unwrapping using `parax.unwrap`.

This module provides tools for deferred evaluation, parameterization, 
and gradient stopping within JAX PyTrees.
"""

from abc import abstractmethod
from typing import Generic, TypeVar, Callable, Any, Union

import equinox as eqx
import jax
from jaxtyping import PyTree

from parax.constant import AbstractConstant

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


def unwrap(tree: PyTree):
    """
    Map across a PyTree and recursively resolve all `AbstractUnwrappable` nodes.

    **Corner Case Note:** This function handles nested unwrappables from the 
    inside out. If an `AbstractUnwrappable` contains other unwrappables, the 
    inner nodes are recursively unwrapped *before* the outer node's `.unwrap()` 
    method is called.
    """

    def _unwrap(tree, *, include_self: bool):
        def _map_fn(leaf):
            if isinstance(leaf, AbstractUnwrappable):
                # Unwrap subnodes, then itself
                return _unwrap(leaf, include_self=False).unwrap()
            return leaf

        def is_leaf(x):
            is_unwrappable = isinstance(x, AbstractUnwrappable)
            included = include_self or x is not tree
            return is_unwrappable and included

        return jax.tree_util.tree_map(f=_map_fn, tree=tree, is_leaf=is_leaf)

    return _unwrap(tree, include_self=True)


class Parameterized(AbstractUnwrappable[T]):
    """
    Unwrap into an arbitrary object by calling a function with arguments.
    
    Useful for injecting dynamic generation (like neural network outputs 
    or complex parametrizations) into an otherwise static PyTree structure.
    """

    fn: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, fn: Callable, *args, **kwargs):
        """
        Args:
            fn: The callable to execute upon unwrapping.
            *args: Positional arguments to pass to `fn`.
            **kwargs: Keyword arguments to pass to `fn`.
        """
        self.fn = fn
        self.args = tuple(args)
        self.kwargs = kwargs

    def unwrap(self) -> T:
        return self.fn(*self.args, **self.kwargs)


class Computed(AbstractUnwrappable[T]):
    """
    Unwrap a PyTree by applying a function to its array-like leaves.
    
    **Corner Case Note:** This relies on `eqx.is_array_like`. Non-array 
    leaves (e.g., strings, standard integers, metadata) inside `tree` 
    will be bypassed and left intact, while any array-like objects
    (including booleans) while be mapped.
    """
    
    tree: T
    fn: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, tree: T, fn: Callable, *args, **kwargs):
        """
        Args:
            tree: The target PyTree to map the computation over.
            fn: The function to apply to each array-like leaf in the tree.
            *args: Positional arguments passed to `fn` after the leaf.
            **kwargs: Keyword arguments passed to `fn`.
        """
        self.tree = tree
        self.fn = fn
        self.args = tuple(args)
        self.kwargs = kwargs

    def unwrap(self) -> T:
        def _map_fn(x):
            if not eqx.is_array_like(x):
                return x
            
            return self.fn(x, *self.args, **self.kwargs)

        return jax.tree.map(_map_fn, self.tree, is_leaf=eqx.is_array_like)


class Frozen(AbstractUnwrappable[T], AbstractConstant[T]):
    """
    Applies `jax.lax.stop_gradient` to all array-like leaves before unwrapping.

    Implements the `AbstractConstant` interface so it can be filtered out 
    during optimization partitioning.

    **Corner Case Note:** The `__init__` automatically prevents double-wrapping. 
    If you pass a `Frozen` object into `Frozen`, it safely absorbs it rather 
    than nesting them.
    """
    tree: T

    def __init__(self, tree: T):
        """
        Args:
            tree: The PyTree to freeze.
        """
        if isinstance(tree, Frozen):
            tree = tree.tree
        self.tree = tree

    def as_free(self) -> T:
        return self.tree

    def unwrap(self) -> T:
        differentiable, static = eqx.partition(self.tree, eqx.is_array_like)
        return eqx.combine(jax.lax.stop_gradient(differentiable), static)
    
    def __getattr__(self, name):
        if hasattr(self.tree, name):
            return getattr(self.tree, name)
        return super().__getattribute__(name)    
