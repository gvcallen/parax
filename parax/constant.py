"""
An abstract interface for a constant node in a model.

This is the canonical `tag` in Parax used to filter out
fixed parameters and frozen layers.
"""

from abc import abstractmethod
from typing import Generic, TypeVar, Union

import equinox as eqx

T = TypeVar('T')


class AbstractConstant(eqx.Module, Generic[T]):
    """
    An abstract interface for a constant node in a model.
     
    This class is simply used as a tag for a node that is currently
    constant and can potentially be set as free. It is used as a
    type-check for `parax.is_constant` and `parax.is_free_array`
    as the main filters in Parax for partitioning.
    
    Note that this class does not guarantee that the underlying
    type implements stop gradients. For concrete classes that do,
    for example for freezing modules or fixing Parax variables,
    see `parax.Frozen` and `parax.Fixed`.
    """

    @abstractmethod
    def as_free(self) -> T:
        """Return a freed version of self."""
        pass


def as_free(value: Union[AbstractConstant[T], T]) -> T:
    """
    Returns a version of `value` that is not fixed.
      
    Calls `value.as_free()` if value is `AbstractConstant` else returns `value`.

    Args:
        value: An arbitrary value, potentially wrapped in an AbstractConstant.

    Returns:
        The unwrapped parameter, or the original value if it wasn't wrapped.
    """    
    if isinstance(value, AbstractConstant):
        return value.as_free()
    return value