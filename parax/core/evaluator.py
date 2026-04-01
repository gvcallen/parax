"""
The foundational class for composable, differentiable, parametric PyTree manipulation.
"""

from __future__ import annotations
import operator
from typing import Any
import jax.numpy as jnp

from jaxtyping import PyTree

from parax.core import Module

class Evaluator(Module):
    """
    A parametric, composable callable that transforms input arguments into arrays.
    
    Supports standard Python operator overloading to seamlessly compose 
    transforms into complex graphs.
    """
    
    def __call__(self, *args: Any, **kwargs: Any) -> jnp.ndarray:
        raise NotImplementedError("Operator nodes must implement __call__.")

    # --- Arithmetic Operators ---

    def __add__(self, other: Any) -> Evaluator:
        from parax.evaluators import Binary
        return Binary(left=self, right=other, fn=operator.add)

    def __sub__(self, other: Any) -> Evaluator:

        return Binary(left=self, right=other, fn=operator.sub)

    def __mul__(self, other: Any) -> Evaluator:

        return Binary(left=self, right=other, fn=operator.mul)

    def __truediv__(self, other: Any) -> Evaluator:

        return Binary(left=self, right=other, fn=operator.truediv)

    def __pow__(self, other: Any) -> Evaluator:

        return Binary(left=self, right=other, fn=operator.pow)

    # --- Unary Operators ---
    
    def __neg__(self) -> Evaluator:
        return Map(operator=self, fn=operator.neg)

    # --- Reverse Arithmetic (for <scalar> + <Operator>) ---

    def __radd__(self, other: Any) -> Evaluator:

        return Binary(left=other, right=self, fn=operator.add)

    def __rsub__(self, other: Any) -> Evaluator:

        return Binary(left=other, right=self, fn=operator.sub)

    def __rmul__(self, other: Any) -> Evaluator:

        return Binary(left=other, right=self, fn=operator.mul)
        
    def __rtruediv__(self, other: Any) -> Evaluator:

        return Binary(left=other, right=self, fn=operator.truediv)

    # --- Comparison Operators (Useful for custom logic) ---

    def __gt__(self, other: Any) -> Evaluator:

        return Binary(left=self, right=other, fn=operator.gt)

    def __lt__(self, other: Any) -> Evaluator:

        return Binary(left=self, right=other, fn=operator.lt)
        
    def __ge__(self, other: Any) -> Evaluator:

        return Binary(left=self, right=other, fn=operator.ge)

    def __le__(self, other: Any) -> Evaluator:

        return Binary(left=self, right=other, fn=operator.le)