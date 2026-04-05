from __future__ import annotations
import operator
from typing import Any, Generic, TypeVar, Union, ParamSpec
import equinox as eqx
import jax

OpInputs = ParamSpec("P")
OpOutputs = TypeVar("OpOutputs")

# --- Helper to map operators over PyTrees ---
def tree_op(op_fn):
    """Wraps a standard operator to apply it element-wise across PyTrees."""
    return lambda *args: jax.tree_util.tree_map(op_fn, *args)

class Operator(eqx.Module, Generic[OpInputs, OpOutputs]):
    """
    A composable callable that applies operations element-wise across PyTrees.
    
    Supports standard Python operator overloading (+, -, *, etc.) to seamlessly 
    compose individual operators into complex computation graphs. Operations are 
    automatically mapped across the leaves of arbitrary PyTrees (like tuples or 
    dictionaries).
    """
    
    def __call__(self, *args: OpInputs.args, **kwargs: OpInputs.kwargs) -> OpOutputs:
        raise NotImplementedError("Operator nodes must implement __call__.")

    # --- Arithmetic Operators ---

    def __add__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.add))

    def __sub__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.sub))

    def __mul__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.mul))

    def __truediv__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.truediv))

    def __pow__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.pow))

    # --- Unary Operators ---
    
    def __neg__(self) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Map
        return Map(operator=self, fn=tree_op(operator.neg))

    # --- Reverse Arithmetic (for <scalar> + <Operator>) ---

    def __radd__(self, other: Any) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=other, right=self, fn=tree_op(operator.add))

    def __rsub__(self, other: Any) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=other, right=self, fn=tree_op(operator.sub))

    def __rmul__(self, other: Any) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=other, right=self, fn=tree_op(operator.mul))
        
    def __rtruediv__(self, other: Any) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=other, right=self, fn=tree_op(operator.truediv))

    # --- Comparison Operators ---

    def __gt__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.gt))

    def __lt__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.lt))
        
    def __ge__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.ge))

    def __le__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.le))