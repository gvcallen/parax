from typing import TypeVar, Union, Any

import jax.numpy as jnp

from parax.constant import AbstractConstant
from parax.variables import Param, Fixed
from parax.unwrappables import Frozen
from parax.filters import is_param

T = TypeVar('T')

def as_free(value: Union[AbstractConstant[T], T]) -> T:
    """
    Returns a freed version of `value` by stripping any constant wrappers.
      
    If `value` implements `AbstractConstant`, this calls `value.as_free()`.
    Otherwise, it acts as a safe no-op and returns `value` unchanged. This
    makes it safe to use directly within a `jax.tree_map` over mixed PyTrees.

    Args:
        value: An arbitrary value, potentially wrapped in an `AbstractConstant`.

    Returns:
        The freed parameter, or the original value if it was not fixed.
    """    
    if isinstance(value, AbstractConstant):
        return value.as_free()
    return value


def as_frozen(pytree: Union[T | Frozen[T]]) -> T:
    """
    Returns `pytree` wrapped in a `parax.Frozen` module, creating one if needed.

    Args:
        pytree: An arbitrary PyTree.

    Returns:
        A frozen version of the PyTree. If it is already frozen, returns it directly.
    """    
    if isinstance(pytree, Frozen):
        return pytree
    return Frozen(pytree)


def as_param(value: Any) -> Any:
    """
    Returns `value` as a `parax.Param`, wrapping it if necessary.

    Args:
        value: An arbitrary value or array.

    Returns:
        The instantiated parameter.
    """    
    if is_param(value):
        return value
    return jnp.asarray(value)



def as_fixed(value: Param) -> Fixed:
    """
    Returns `value` as a `parax.Fixed` variable, wrapping it if necessary.

    Args:
        value: An arbitrary variable or array-like object.

    Returns:
        A fixed version of the variable.
    """    
    if isinstance(value, Fixed):
        return value
    return Fixed(value)