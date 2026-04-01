"""
Transform evaluator factories.

These functions act as convenience builders, wrapping core AST nodes 
(like `Map`, `Binary`, and `Stack`) with specific mathematical transformations 
to simplify objective graph construction.
"""

from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp

from parax.core import Operator
from parax.operators.core import Map, Lambda, Binary


def Stack(*evaluators: Operator, axis: int = -1) -> Operator:
    def fn(*args: Any, **kwargs):
        results = [t(*args, **kwargs) for t in evaluators]
        return jnp.stack(results, axis=axis)
    
    return Lambda(fn=fn)


def Derivative(
    evaluator: Operator, 
    step_attr: str,
    axis: int = 0, 
    order: int = 1, 
    arg_index: int = 1
) -> Operator:
    """
    Computes the discrete numerical derivative of the data.
    
    Parameters
    ----------
    evaluator : Evaluator
        The base evaluator whose output will be differentiated.
    step_attr : str, default='f_scaled'
        The attribute name on the step object used for the gradient spacing.
    axis : int, default=0
        The axis along which to compute the gradient.
    order : int, default=1
        The number of times to apply the derivative.
    arg_index : int, default=1
        The index of the positional argument (in `*args`) that contains 
        the step object (e.g., the `Frequency` object is typically `args[1]`).
        
    Returns
    -------
    Evaluator
        A composed Binary evaluator representing the derivative.
    """
    # 1. The math function (Pure array logic)
    def _grad_fn(data: jnp.ndarray, dx: jnp.ndarray | float) -> jnp.ndarray:
        for _ in range(order):
            data = jnp.gradient(data, dx, axis=axis)
        return data

    # 2. A leaf evaluator that grabs the step size from the correct positional argument
    dx_extractor = Lambda(fn=lambda *args, **kwargs: getattr(args[arg_index], step_attr))

    # 3. Stitch them together safely in the PyTree
    return Binary(
        left=evaluator, 
        right=dx_extractor, 
        fn=_grad_fn
    )


def Sum(evaluators: Sequence[Operator]) -> Operator:
    """
    Sums the outputs of multiple Evaluators into a single scalar or array.
    
    Useful for creating a composite scalar loss function from multiple 
    independent penalty vectors.
    """
    return Map(evaluator=Stack(evaluators=evaluators, axis=0), fn=lambda data: jnp.sum(data, axis=0))


def Flatness(evaluator: Operator, step_attr: str) -> Operator:
    """
    Computes the numerical derivative of the data with respect to frequency.
    
    Used to penalize ripple or enforce gain flatness across a band. It relies on 
    `freq.f_scaled` to prevent catastrophic numerical instability.
    """
    return Derivative(evaluator=evaluator, axis=0, order=1, step_attr=step_attr)


def Reduce(
    evaluator: Operator, 
    fn: Callable[..., jnp.ndarray], 
    axis: int | tuple[int, ...] | None = None
) -> Operator:
    """
    Applies a reduction operation (e.g., jnp.max, jnp.mean) over a specific axis.
    """
    # We use Map to handle the PyTree structure, and lambda handles the pure array math!
    return Map(evaluator=evaluator, fn=lambda data: fn(data, axis=axis))


def Index(evaluator: Operator, indices: Any) -> Operator:
    """
    Slices or indexes the output of another Evaluator.
    
    Useful for extracting specific ports or frequency ranges from a larger
    N-dimensional response array.
    """
    return Map(evaluator=evaluator, fn=lambda data: data[indices])


def Mask(evaluator: Operator, mask: jnp.ndarray) -> Operator:
    """
    Applies a boolean mask to the final dimension of the data.
    
    Utilizes `jax.vmap` to efficiently broadcast the masking operation across 
    the batch/frequency dimensions.
    """
    # JAX compilation will safely bake this mask array into the XLA graph.
    return Map(evaluator=evaluator, fn=lambda data: jax.vmap(lambda m: m[mask])(data))



def Diagonal(evaluator: Operator) -> Operator:
    """
    Extracts the diagonals of N-port scattering matrices.
    """
    return Map(evaluator=evaluator, fn=lambda data: jax.vmap(jnp.diag)(data))


def OffDiagonal(evaluator: Operator, n_ports: int) -> Operator:
    """
    Extracts the off-diagonals (transmission) of N-port matrices.
    """
    mask = ~jnp.eye(n_ports, dtype=bool)
    return Mask(evaluator=evaluator, mask=mask)