"""
Core evaluators.
"""

from __future__ import annotations
import operator
from typing import Callable
import jax.numpy as jnp

from typing import Any
import jax.numpy as jnp

from parax.core import Operator, field


class Lambda(Operator):
    """
    Wraps a standard Python or JAX callable.
    """
    fn: Callable

    def __call__(self, *args: Any, **kwargs) -> Any:
        return self.fn(*args, **kwargs)


class Constant(Operator):
    """
    Returns a fixed constant array or scalar.
    """
    value: float | int | complex | Any
    
    def __call__(self, *args: Any, **kwargs) -> Any:
        return jnp.asarray(self.value)


class Binary(Operator):
    """
    Returns a the result of a callable that accepts the result of two operators.
    
    The functional callable ``fn`` must have the signature ``f(left, right)``.
    """
    fn: Callable
    left: Operator
    right: Operator

    def __call__(self, *args: Any, **kwargs) -> Any:
        val_left = self.left(*args, **kwargs)
        val_right = self.right(*args, **kwargs)
        return self.fn(val_left, val_right)


class Where(Operator):
    """
    A conditional branching node using `jnp.where`.

    Evaluates a boolean condition (from an Operator) and returns elements 
    from the `true_branch` or `false_branch` accordingly. 
    
    Useful for applying argument-dependent logic or piecewise penalty functions.
    """
    condition: Operator
    true_branch: Operator
    false_branch: Operator

    def __call__(self, *args: Any, **kwargs) -> Any:
        cond_val = self.condition(*args, **kwargs)
        
        true_val = self.true_branch(*args, **kwargs)
        false_val = self.false_branch(*args, **kwargs)
            
        return jnp.where(cond_val, jnp.asarray(true_val), jnp.asarray(false_val))


class Method(Operator):
    """
    Dynamically accesses and executes a method on the first argument.
    """
    path: str = field(static=True)

    def __call__(self, *args: Any, **kwargs) -> Any:
        if not args:
            raise ValueError(f"Method operator '{self.path}' requires at least one positional argument.")
            
        obj = args[0] 
        method_args = args[1:]
        
        func = operator.attrgetter(self.path)(obj)
        return func(*method_args, **kwargs)


class Map(Operator):
    """
    Applies an arbitrary function to a single operator's output.
    """
    fn: Callable
    operator: Operator

    def __call__(self, *args: Any, **kwargs):
        return self.fn(self.operator(*args, **kwargs))