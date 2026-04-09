import operator

import jax
import jax.numpy as jnp
from typing import Any, Callable, Union
from parax.operator import Operator, OpInputs, OpOutputs, tree_op
from parax.op import Map
from parax.field import field

class Stack(Operator[OpInputs, OpOutputs]):
    """Stacks the results of multiple operators along an axis."""
    operators: tuple[Operator, ...]
    axis: int = field(default=-1, static=True)

    def __call__(self, *args: OpInputs.args, **kwargs: OpInputs.kwargs) -> OpOutputs:
        results = [op(*args, **kwargs) for op in self.operators]
        return jnp.stack(results, axis=self.axis)

class Index(Operator[OpInputs, OpOutputs]):
    """Slices or indexes the output of another operator."""
    operator: Operator
    indices: Any = field(static=True)

    def __call__(self, *args: OpInputs.args, **kwargs: OpInputs.kwargs) -> OpOutputs:
        return self.operator(*args, **kwargs)[self.indices]

class Mask(Operator[OpInputs, OpOutputs]):
    """Applies a boolean mask to the output of an operator."""
    operator: Operator
    mask: jnp.ndarray

    def __call__(self, *args: OpInputs.args, **kwargs: OpInputs.kwargs) -> OpOutputs:
        data = self.operator(*args, **kwargs)
        # vmap ensures we apply the mask correctly across lead dimensions (e.g. freq)
        return jax.vmap(lambda x: x[self.mask])(data)
    
class Reduce(Operator[OpInputs, OpOutputs]):
    """Applies a reduction (e.g., jnp.max, jnp.mean) over a specific axis."""
    operator: Operator
    fn: Callable = field(static=True)
    axis: Union[int, tuple[int, ...], None] = field(default=None, static=True)

    def __call__(self, *args: OpInputs.args, **kwargs: OpInputs.kwargs) -> OpOutputs:
        return self.fn(self.operator(*args, **kwargs), axis=self.axis)

class Sum(Operator[OpInputs, OpOutputs]):
    """Evaluates multiple operators and sums their outputs together."""
    operators: tuple[Operator, ...] | list[Operator]

    def __call__(self, *args: OpInputs.args, **kwargs: OpInputs.kwargs) -> OpOutputs:
        results = [op(*args, **kwargs) for op in self.operators]
        return sum(results[1:], start=results[0])
    
class Negate(Map):
    """
    Applies an arbitrary function to a single operator's output.
    """
    def __init__(self, other):
        return super().__init__(fn=tree_op(operator.neg), operator=other)

class Derivative(Operator[OpInputs, OpOutputs]):
    """Computes numerical derivative with respect to a context attribute."""
    operator: Operator
    step_attr: str = field(static=True)
    axis: int = field(default=0, static=True)
    order: int = field(default=1, static=True)
    arg_index: int = field(default=1, static=True)

    def __call__(self, *args: OpInputs.args, **kwargs: OpInputs.kwargs) -> OpOutputs:
        data = self.operator(*args, **kwargs)
        # Grabs the step size (e.g. freq.f_scaled) from the context argument
        dx = getattr(args[self.arg_index], self.step_attr)
        
        for _ in range(self.order):
            data = jnp.gradient(data, dx, axis=self.axis)
        return data

class Flatness(Derivative):
    """Enforces gain flatness by computing the first derivative."""
    order: int = field(default=1, static=True)
    
class Diagonal(Operator[OpInputs, OpOutputs]):
    """Extracts the diagonals of matrices."""
    operator: Operator

    def __call__(self, *args: OpInputs.args, **kwargs: OpInputs.kwargs) -> OpOutputs:
        data = self.operator(*args, **kwargs)
        return jax.vmap(jnp.diag)(data)

class OffDiagonal(Mask):
    """Extracts off-diagonal elements."""
    def __init__(self, operator: Operator, n_ports: int, **kwargs):
        mask = ~jnp.eye(n_ports, dtype=bool)
        # We initialize the parent Mask class with the generated eye mask
        super().__init__(operator=operator, mask=mask, **kwargs)