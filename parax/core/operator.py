"""
Core evaluators.
"""

from __future__ import annotations
import operator
from typing import Callable, Any, TypeVar, Union

import jax

# Use typing_extensions for Python < 3.11 compatibility
try:
    from typing import TypeVarTuple, Unpack
except ImportError:
    from typing_extensions import TypeVarTuple, Unpack

from parax.core import Operator, field

# Re-define or import the TypeVars used in your base Operator
OpInputs = TypeVarTuple("OpInputs")
OpOutputs = TypeVar("OpOutputs")


class Lambda(Operator[Unpack[OpInputs], OpOutputs]):
    """
    Wraps a standard Python or JAX callable with the same domain as the operator.
    """
    fn: Callable[[Unpack[OpInputs]], OpOutputs]

    def __call__(self, *args: Unpack[OpInputs], **kwargs: Any) -> OpOutputs:
        return self.fn(*args, **kwargs)


class Constant(Operator[Unpack[OpInputs], OpOutputs]):
    """
    Returns a fixed constant array or scalar.
    """
    value: OpOutputs
    
    def __call__(self, *args: Unpack[OpInputs], **kwargs: Any) -> OpOutputs:
        return self.value


class Binary(Operator[Unpack[OpInputs], OpOutputs]):
    """
    Returns the result of a callable that accepts the result of two operators.
    
    The functional callable ``fn`` must have the signature ``f(left, right)``.
    """
    fn: Callable[[Any, Any], OpOutputs]
    # Union handles the case where an operator is composed with a raw scalar/array
    left: Union[Operator[Unpack[OpInputs], Any], Any]
    right: Union[Operator[Unpack[OpInputs], Any], Any]

    def __call__(self, *args: Unpack[OpInputs], **kwargs: Any) -> OpOutputs:
        val_left = self.left(*args, **kwargs) if isinstance(self.left, Operator) else self.left
        val_right = self.right(*args, **kwargs) if isinstance(self.right, Operator) else self.right
        return self.fn(val_left, val_right)


class Where(Operator[Unpack[OpInputs], OpOutputs]):
    """
    A conditional branching node using `jax.lax.cond`.

    Evaluates a boolean condition (from an Operator) and returns the output
    of either `true_branch` or `false_branch` depending on the condition.
    """
    condition: Union[Operator[Unpack[OpInputs], Any], Any]
    true_branch: Union[Operator[Unpack[OpInputs], OpOutputs], OpOutputs]
    false_branch: Union[Operator[Unpack[OpInputs], OpOutputs], OpOutputs]

    def __call__(self, *args: Unpack[OpInputs], **kwargs: Any) -> OpOutputs:
        cond_val = self.condition(*args, **kwargs) if isinstance(self.condition, Operator) else self.condition
        
        def true_fn(_: Any) -> OpOutputs:
            return self.true_branch(*args, **kwargs) if isinstance(self.true_branch, Operator) else self.true_branch
        
        def false_fn(_: Any) -> OpOutputs:
            return self.false_branch(*args, **kwargs) if isinstance(self.false_branch, Operator) else self.false_branch
        
        return jax.lax.cond(cond_val, true_fn, false_fn, operand=None)


class Method(Operator[Unpack[OpInputs], OpOutputs]):
    """
    Dynamically accesses and executes a method on the first argument.
    """
    path: str = field(static=True)

    def __call__(self, *args: Unpack[OpInputs], **kwargs: Any) -> OpOutputs:
        if not args:
            raise ValueError(f"Method operator '{self.path}' requires at least one positional argument.")
            
        obj = args[0] 
        method_args = args[1:]
        
        func = operator.attrgetter(self.path)(obj)
        return func(*method_args, **kwargs)


class Map(Operator[Unpack[OpInputs], OpOutputs]):
    """
    Applies an arbitrary function to a single operator's output.
    """
    fn: Callable[[Any], OpOutputs]
    operator: Union[Operator[Unpack[OpInputs], Any], Any]

    def __call__(self, *args: Unpack[OpInputs], **kwargs: Any) -> OpOutputs:
        val = self.operator(*args, **kwargs) if isinstance(self.operator, Operator) else self.operator
        return self.fn(val)