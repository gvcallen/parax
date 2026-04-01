"""
A class for composable, differentiable, parametric PyTree manipulation.
"""

from __future__ import annotations
import operator
from typing import Any, Generic, TypeVar, Union
import jax.numpy as jnp
import equinox as eqx

# Use typing_extensions for Python < 3.10 compatibility
try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec

OpInputs = ParamSpec("P")
OpOutputs = TypeVar("OpOutputs")

class Operator(eqx.Module, Generic[OpInputs, OpOutputs]):
    """
    A composable callable that applies some operation to input arguments.
    
    Supports standard Python operator overloading to seamlessly compose 
    operators into complex graphs.
    """
    
    def __call__(self, *args: OpInputs.args, **kwargs: OpInputs.kwargs) -> OpOutputs:
        raise NotImplementedError("Operator nodes must implement __call__.")

    # --- Arithmetic Operators ---

    def __add__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.operators import Binary
        return Binary(left=self, right=other, fn=operator.add)

    def __sub__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.operators import Binary
        return Binary(left=self, right=other, fn=operator.sub)

    def __mul__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.operators import Binary
        return Binary(left=self, right=other, fn=operator.mul)

    def __truediv__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.operators import Binary
        return Binary(left=self, right=other, fn=operator.truediv)

    def __pow__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.operators import Binary
        return Binary(left=self, right=other, fn=operator.pow)

    # --- Unary Operators ---
    
    def __neg__(self) -> Operator[OpInputs, OpOutputs]:
        from parax.operators import Map
        return Map(operator=self, fn=operator.neg)

    # --- Reverse Arithmetic (for <scalar> + <Operator>) ---

    def __radd__(self, other: Any) -> Operator[OpInputs, OpOutputs]:
        from parax.operators import Binary
        return Binary(left=other, right=self, fn=operator.add)

    def __rsub__(self, other: Any) -> Operator[OpInputs, OpOutputs]:
        from parax.operators import Binary
        return Binary(left=other, right=self, fn=operator.sub)

    def __rmul__(self, other: Any) -> Operator[OpInputs, OpOutputs]:
        from parax.operators import Binary
        return Binary(left=other, right=self, fn=operator.mul)
        
    def __rtruediv__(self, other: Any) -> Operator[OpInputs, OpOutputs]:
        from parax.operators import Binary
        return Binary(left=other, right=self, fn=operator.truediv)

    # --- Comparison Operators (Useful for custom logic) ---

    def __gt__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.operators import Binary
        return Binary(left=self, right=other, fn=operator.gt)

    def __lt__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.operators import Binary
        return Binary(left=self, right=other, fn=operator.lt)
        
    def __ge__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.operators import Binary
        return Binary(left=self, right=other, fn=operator.ge)

    def __le__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.operators import Binary
        return Binary(left=self, right=other, fn=operator.le)