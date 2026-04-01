"""
Core evaluators.
"""

from __future__ import annotations
import operator
from typing import Callable, Any, Union

import jax

from parax.core import Operator, OpInputs, OpOutputs, field


class Lambda(Operator[OpInputs, OpOutputs]):
    """
    Wraps a standard Python or JAX callable with the same domain as the operator.
    """
    fn: Callable[OpInputs, OpOutputs]

    def __call__(self, *args: OpInputs.args, **kwargs: OpInputs.kwargs) -> OpOutputs:
        return self.fn(*args, **kwargs)


class Constant(Operator[OpInputs, OpOutputs]):
    """
    Returns a fixed constant array or scalar.
    """
    value: OpOutputs
    
    def __call__(self, *args: OpInputs.args, **kwargs: OpInputs.kwargs) -> OpOutputs:
        return self.value


class Binary(Operator[OpInputs, OpOutputs]):
    """
    Returns the result of a callable that accepts the result of two operators.
    
    The functional callable ``fn`` must have the signature ``f(left, right)``.
    """
    fn: Callable[[Any, Any], OpOutputs]
    left: Union[Operator[OpInputs, Any], Any]
    right: Union[Operator[OpInputs, Any], Any]

    def __call__(self, *args: OpInputs.args, **kwargs: OpInputs.kwargs) -> OpOutputs:
        val_left = self.left(*args, **kwargs) if isinstance(self.left, Operator) else self.left
        val_right = self.right(*args, **kwargs) if isinstance(self.right, Operator) else self.right
        return self.fn(val_left, val_right)


class Where(Operator[OpInputs, OpOutputs]):
    """
    A conditional branching node using `jax.lax.cond`.

    Evaluates a boolean condition (from an Operator) and returns the output
    of either `true_branch` or `false_branch` depending on the condition.
    """
    condition: Union[Operator[OpInputs, Any], Any]
    true_branch: Union[Operator[OpInputs, OpOutputs], OpOutputs]
    false_branch: Union[Operator[OpInputs, OpOutputs], OpOutputs]

    def __call__(self, *args: OpInputs.args, **kwargs: OpInputs.kwargs) -> OpOutputs:
        cond_val = self.condition(*args, **kwargs) if isinstance(self.condition, Operator) else self.condition
        
        def true_fn(_: Any) -> OpOutputs:
            return self.true_branch(*args, **kwargs) if isinstance(self.true_branch, Operator) else self.true_branch
        
        def false_fn(_: Any) -> OpOutputs:
            return self.false_branch(*args, **kwargs) if isinstance(self.false_branch, Operator) else self.false_branch
        
        return jax.lax.cond(cond_val, true_fn, false_fn, operand=None)


class Method(Operator[OpInputs, OpOutputs]):
    """
    Dynamically accesses and executes a method on the first argument.
    """
    path: str = field(static=True)

    def __call__(self, *args: OpInputs.args, **kwargs: OpInputs.kwargs) -> OpOutputs:
        if not args:
            raise ValueError(f"Method operator '{self.path}' requires at least one positional argument.")
            
        obj = args[0] 
        method_args = args[1:]
        
        func = operator.attrgetter(self.path)(obj)
        return func(*method_args, **kwargs)


class Map(Operator[OpInputs, OpOutputs]):
    """
    Applies an arbitrary function to a single operator's output.
    """
    fn: Callable[[Any], OpOutputs]
    operator: Union[Operator[OpInputs, Any], Any]

    def __call__(self, *args: OpInputs.args, **kwargs: OpInputs.kwargs) -> OpOutputs:
        val = self.operator(*args, **kwargs) if isinstance(self.operator, Operator) else self.operator
        return self.fn(val)