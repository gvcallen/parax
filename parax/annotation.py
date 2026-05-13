"""
Interfaces for attaching metadata to Equinox modules.
"""

from typing import TypeVar, Generic, TypeGuard, Any

import equinox as eqx


T = TypeVar("T")


class AbstractAnnotated(eqx.Module, Generic[T]):
    """
    An abstract interface for an Equinox module that is annotated
    with arbitrary metadata.

    In Parax, metadata is typically used to store units, descriptions, tags, 
    or optimization hints alongside the underlying JAX array.

    Attributes:
        metadata: Returns the underlying metadata.
    """
    metadata: eqx.AbstractVar[T]



def is_annotated(x: Any) -> TypeGuard[AbstractAnnotated]:
    """
    Returns True if `x` is an instance of `parax.AbstractAnnotated`
    (i.e. has metadata).
    """
    return isinstance(x, AbstractAnnotated)