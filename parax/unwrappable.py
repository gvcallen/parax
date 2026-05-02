"""
The general interface for PyTree unwrapping using `parax.unwrap`.
"""

from abc import abstractmethod
from typing import Generic, TypeVar, Callable, Any

import equinox as eqx
import jax
from jaxtyping import PyTree


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
    """Map across a PyTree and unwrap all :class:`AbstractUnwrappable` nodes.

    This leaves all other nodes unchanged. If nested, the innermost
    ``AbstractUnwrappable`` nodes are unwrapped first.

    Example:
        Enforcing positivity.

        .. doctest::

            >>> import parax
            >>> import jax.numpy as jnp
            >>> params = parax.Parameterize(jnp.exp, jnp.zeros(3))
            >>> parax.unwrap(("abc", 1, params))
            ('abc', 1, Array([1., 1., 1.], dtype=float32))
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



class DerivedTree(AbstractUnwrappable[T]):
    """Unwrap an object by calling fn with args and kwargs.

    All of fn, args and kwargs may contain trainable parameters.

    .. note::

        Unwrapping typically occurs after model initialization. Therefore, if the
        ``Parameterize`` object may be created in a vectorized context, we recommend
        ensuring that ``fn`` still unwraps correctly, e.g. by supporting broadcasting.

    Example:
        .. doctest::

            >>> from parax.wrappers import Parameterize, unwrap
            >>> import jax.numpy as jnp
            >>> positive = Parameterize(jnp.exp, jnp.zeros(3))
            >>> unwrap(positive)  # Aplies exp on unwrapping
            Array([1., 1., 1.], dtype=float32)

    Args:
        fn: Callable to call with args, and kwargs.
        *args: Positional arguments to pass to fn.
        **kwargs: Keyword arguments to pass to fn.
    """

    fn: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, fn: Callable, *args, **kwargs):
        self.fn = fn
        self.args = tuple(args)
        self.kwargs = kwargs

    def unwrap(self) -> T:
        return self.fn(*self.args, **self.kwargs)


class FixedTree(AbstractUnwrappable[T]):
    """Applies stop gradient to all arraylike leaves before unwrapping.

    See also :func:`non_trainable`, which is probably a generally prefereable way to
    achieve similar behaviour, which wraps the arraylike leaves directly, rather than
    the tree. Useful to mark pytrees (arrays, submodules, etc) as frozen/non-trainable.
    Note that the underlying parameters may still be impacted by regularization,
    so it is generally advised to use this as a suggestively named class
    for filtering parameters.
    """

    tree: T

    def unwrap(self) -> T:
        differentiable, static = eqx.partition(self.tree, eqx.is_array_like)
        return eqx.combine(jax.lax.stop_gradient(differentiable), static)