"""
Core evaluators.
"""

from __future__ import annotations
import operator
from typing import Callable, Any
import jax.numpy as jnp

from jaxtyping import PyTree

from parax.core import Evaluator, field


class Lambda(Evaluator):
    """
    Wraps a standard Python or JAX callable.
    
    This is useful for defining quick, custom, on-the-fly objective 
    functions without needing to subclass Transform.
    """
    fn: Callable[..., jnp.ndarray]

    def __call__(self, *args: PyTree, **kwargs) -> jnp.ndarray:
        return self.fn(*args, **kwargs)


class Constant(Evaluator):
    """
    Returns a fixed constant array or scalar.
    
    Useful for inserting fixed thresholds, reference gold-standard data, 
    or static masks directly into the transform tree.
    """
    value: float | int | complex | jnp.ndarray
    
    def __call__(self, *args: PyTree, **kwargs: Any) -> jnp.ndarray:
        return jnp.asarray(self.value)


class Binary(Evaluator):
    """
    Calculates a metric between two inputs.
    
    The functional callable ``fn`` must have the signature ``f(left, right)``.
    Inputs can be other Transforms, arrays, or scalars.
    """
    fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    left: Evaluator | jnp.ndarray | float | int
    right: Evaluator | jnp.ndarray | float | int

    def __call__(self, *args: PyTree, **kwargs: Any) -> jnp.ndarray:
        # Resolve left branch
        val_left = self.left(*args, **kwargs) if isinstance(self.left, Evaluator) else self.left
        
        # Resolve right branch
        val_right = self.right(*args, **kwargs) if isinstance(self.right, Evaluator) else self.right
            
        return self.fn(jnp.asarray(val_left), jnp.asarray(val_right))


class Where(Evaluator):
    """
    A conditional branching node using `jnp.where`.

    Evaluates a boolean condition (from an Transform) and returns elements 
    from the `true_branch` or `false_branch` accordingly. 
    
    Useful for applying argument-dependent logic or piecewise penalty functions.
    """
    condition: Evaluator
    true_branch: Evaluator | jnp.ndarray | float | int
    false_branch: Evaluator | jnp.ndarray | float | int

    def __call__(self, *args: PyTree, **kwargs: Any) -> jnp.ndarray:
        cond_val = self.condition(*args, **kwargs)
        
        true_val = self.true_branch(*args, **kwargs) if isinstance(self.true_branch, Evaluator) else self.true_branch
        false_val = self.false_branch(*args, **kwargs) if isinstance(self.false_branch, Evaluator) else self.false_branch
            
        return jnp.where(cond_val, jnp.asarray(true_val), jnp.asarray(false_val))


class Method(Evaluator):
    """
    Dynamically accesses and executes a method on the first argument.
    
    This acts as the domain-agnostic "Source Node" for the AST.
    It expects args[0] to be the primary object and args[1:] to be 
    the parameters passed to that object's method.
    """
    path: str = field(static=True)

    def __call__(self, *args: PyTree, **kwargs: Any) -> jnp.ndarray:
        if not args:
            raise ValueError(f"Method transform '{self.path}' requires at least one positional argument.")
            
        obj = args[0] 
        method_args = args[1:]
        
        func = operator.attrgetter(self.path)(obj)
        return func(*method_args, **kwargs)


class Map(Evaluator):
    """
    Applies an arbitrary transformation function to a single transform's output.
    
    Typically used to apply operations like matrix diagonal extraction or 
    math reductions across an entire array.
    """
    fn: Callable[[jnp.ndarray], jnp.ndarray]
    transform: Evaluator

    def __call__(self, *args: PyTree, **kwargs: Any) -> jnp.ndarray:
        return self.fn(self.transform(*args, **kwargs))