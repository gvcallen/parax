"""
The general interface for PyTree unwrapping using `parax.unwrap`.
"""

from abc import abstractmethod
from typing import Generic, TypeVar, Callable, Any, Union

import equinox as eqx
import jax
from jaxtyping import PyTree

from parax.constant import AbstractConstant

T = TypeVar("T")


class AbstractUnwrappable(eqx.Module, Generic[T]):
    """An abstract class representing an unwrappable object.

    Unwrappables replace PyTree nodes, applying custom behavior upon unwrapping.
    """

    @abstractmethod
    def unwrap(self) -> T:
        """Returns the unwrapped pytree, assuming no wrapped subnodes exist."""
        pass


def unwrap(tree: PyTree):
    """Map across a PyTree and unwrap all `AbstractUnwrappable` nodes."""

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



class Computed(AbstractUnwrappable[T]):
    """Unwrap an object by calling fn with args and kwargs."""

    fn: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, fn: Callable, *args, **kwargs):
        self.fn = fn
        self.args = tuple(args)
        self.kwargs = kwargs

    def unwrap(self) -> T:
        return self.fn(*self.args, **self.kwargs)


class Frozen(AbstractUnwrappable[T], AbstractConstant[T]):
    """
    Applies stop gradient to all arraylike leaves before unwrapping.

    Implements the `AbstractConstant` interface.
    """
    tree: T

    def __init__(self, tree: T):
        if isinstance(tree, Frozen):
            tree = tree.tree
        self.tree = tree

    def as_free(self) -> T:
        return self.tree

    def unwrap(self) -> T:
        differentiable, static = eqx.partition(self.tree, eqx.is_array_like)
        return eqx.combine(jax.lax.stop_gradient(differentiable), static)
    

def as_frozen(pytree: Union[T | Frozen[T]]) -> T:
    """
    Returns `value` as a `parax.Frozen` module by creating one if needed.

    Args:
        value: An arbitrary pytree.

    Returns:
        A frozen version of the PyTree.
    """    
    if isinstance(pytree, Frozen):
        return pytree
    return Frozen(pytree)