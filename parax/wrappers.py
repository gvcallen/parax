"""
The general interface for PyTree unwrapping using `parax.unwrap`.

This module provides tools for deferred evaluation, parameterization, 
and gradient stopping within JAX PyTrees.
"""

from abc import abstractmethod
from typing import Generic, TypeVar, Callable, Any, Union, TypeGuard, Self

import equinox as eqx
import jax
from jaxtyping import PyTree

from parax.constant import AbstractConstant
from parax.unwrappable import AbstractUnwrappable
from parax.wrappable import AbstractWrappable

T = TypeVar("T")


class Parameterized(AbstractUnwrappable[T]):
    """
    Unwrap into an arbitrary object by calling a function with arguments.
    
    Useful for injecting dynamic generation (like neural network outputs 
    or complex parametrizations) into an otherwise static PyTree structure.
    """

    fn: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, fn: Callable, *args: Any, **kwargs: Any):
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
    
    fn: Callable[..., T]
    tree: T
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, fn: Callable, tree: T, *args: Any, **kwargs: Any):
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


class Frozen(AbstractUnwrappable[T], AbstractWrappable[T], AbstractConstant[T]):
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
    
    def wrap(self, tree: T) -> Self:
        return Frozen(tree)
    

def as_frozen(tree: Union[T | Frozen[T]]) -> T:
    """
    Returns `tree` wrapped in a `parax.Frozen` module, creating one if needed.

    Args:
        tree: An arbitrary tree.

    Returns:
        A frozen version of the tree. If it is already frozen, returns it directly.
    """    
    if isinstance(tree, Frozen):
        return tree
    return Frozen(tree)
    

class Static(AbstractUnwrappable[T], AbstractWrappable[T]):
    """
    Wraps a tree and marks it as static.
    """
    tree: T = eqx.field(static=True)

    def __init__(self, tree: T):
        """
        Args:
            tree: The PyTree to freeze.
        """
        if isinstance(tree, Static):
            tree = tree.tree
        self.tree = tree

    def unwrap(self) -> T:
        return self.tree
    
    def wrap(self, tree: T) -> Self:
        return Static(tree)
    

def as_static(tree: Union[T | Static[T]]) -> T:
    """
    Returns `tree` wrapped in a `parax.Static` module, creating one if needed.

    Args:
        tree: An arbitrary tree.

    Returns:
        A static version of the tree. If it is already static, returns it directly.
    """    
    if isinstance(tree, Static):
        return tree
    return Static(tree)


def as_frozen_or_static(tree: Union[T | Static[T]]) -> Union[Frozen, Static]:
    """
    Returns `tree` wrapped in either a `parax.Static` or `parax.Frozen` module, creating one if needed.

    If `tree` is a JAX array or a structured PyTree, it is wrapped in `parax.Frozen`. 
    If `tree` is an unregistered Python object (an opaque leaf e.g. a lambda), it is wrapped in `parax.Static` 
    to safely bypass JAX transformations.

    Args:
        tree: An arbitrary PyTree, array, or Python object.

    Returns:
        A static or frozen version of the tree. If it is already static or frozen, returns it directly.
    """
    if isinstance(tree, Frozen | Static):
        return tree
    
    # Ask JAX how it views this object
    # If JAX can't unpack it, it returns a list containing exactly the object itself.
    leaves, _ = jax.tree_util.tree_flatten(tree)
    is_opaque_leaf = (len(leaves) == 1) and (leaves[0] is tree)
    
    if is_opaque_leaf and not eqx.is_array(tree):
        return as_static(tree)
    return as_frozen(tree)