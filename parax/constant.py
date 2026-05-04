"""
Abstract interfaces for defining constant (fixed/frozen) nodes in a model.

This module provides the canonical `AbstractConstant` tag used by Parax to 
filter out fixed parameters and frozen layers during PyTree partitioning.
"""

from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx

T = TypeVar('T')


class AbstractConstant(eqx.Module, Generic[T]):
    """
    An abstract interface and structural tag for a constant node.
     
    This class is primarily used as a type-check for `parax.is_constant` 
    and `parax.is_free_array` to facilitate PyTree partitioning.

    **Note:** This interface *only* provides the structural tagging required 
    by Parax. It does not automatically apply `jax.lax.stop_gradient` during 
    computations. Concrete implementations (like `parax.Frozen` and `parax.Fixed`) 
    handle the actual JAX-level gradient stopping and unwrapping logic.
    """

    @abstractmethod
    def as_free(self) -> T:
        """Return the underlying value, stripping the constant tag."""
        pass