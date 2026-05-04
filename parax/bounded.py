"""
An abstract interface for PyTrees that have "bounds".

Bounded trees are allowed to work in three distinct spaces which
may or not may be equivalent:
- The "raw" space, where the original model stores its values
- The "base" space, where bounded optimizers operate
- The "unwrapped" space, where computations take plase

"""
from abc import abstractmethod
from typing import TypeVar
import equinox as eqx

from parax.unwrappables import AbstractUnwrappable, as_frozen

Raw = TypeVar("Raw")
Base = TypeVar("Base")
Unwrapped = TypeVar("Unwrapped")

class AbstractBounded(eqx.Module):
    """
    The abstract interface for a bounded PyTree.

    Bounded PyTrees are a type of unwrappable pytree 
    that have a `bounds` property, and also expose conversions
    between their raw, base and unwrapped spaces.

    Used as a type check for `parax.is_bounded`. 
    """
    @property
    @abstractmethod
    def base(self) -> Base:
        """
        Returns the base space value.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def bounds(self) -> tuple[Base, Base]:
        """
        Returns the PyTree's bounds.

        Must have the same underlying treedef as `self.base`.
        """
        raise NotImplementedError    
  
    @abstractmethod
    def unwrap_from_base(self, base: Base) -> Unwrapped:
        """Evaluates the final output value from a given base value."""
        pass
    
    @abstractmethod
    def replace_from_base(self, base: Base) -> "AbstractBounded":
        """
        Returns a new instance of this object, immutably updated 
        with the new base value.
        """
        pass
    

class BaseSpaceWrapper(AbstractUnwrappable[Base]):
    base: Base
    bounded: AbstractBounded = eqx.field(converter=as_frozen)

    def unwrap(self):
        return self.bounded.unwrap_from_base(self.base)
