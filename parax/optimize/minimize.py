import abc
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Scalar
import equinox as eqx
import optimistix as optx

from parax.unwrappables import unwrap
from parax.filters import is_variable, is_constant, is_constrained
from parax.replace import tree_replace
from parax.variables import map_variables, AbstractVariable
from parax.constraints import RealLine, TreeConstraint

from parax.optimize.results import OptimizeResults

class MinimizePayload(eqx.Module):
    """The core mathematical payload of a minimization run."""
    #: The optimal arrays (y0)
    y: PyTree

    #: The final objective function value
    fn_value: Scalar

    #: Auxiliary data at the optimum
    aux: Any = None


class AbstractMinimizer(eqx.Module):
    """
    An interface for JAX-wrapped minimization algorithms that require a single call.

    Provided to cater for algorithms that `Optimistix` does not support.

    The interface should accept pure PyTrees and return a standardized tuple.
    """
    #: Signifies whether the minimizer supports bounds or not.
    #: If True, PyTree bounds will be passed in `lower` and `upper`
    supports_bounds: eqx.AbstractClassVar[bool]

    @abc.abstractmethod
    def minimize(
        self,
        fn: Callable[[PyTree, Any], Scalar],
        y0: PyTree,
        args: PyTree = None,
        lower: PyTree | None = None,
        upper: PyTree | None = None,
        has_aux: bool = False,
        max_steps: int = 1024,
        **kwargs
    ) -> tuple[MinimizePayload, PyTree]:
        """
        Execute the minimization algorithm.

        Parameters
        ----------
        fn : callable
            The objective function to minimize.
        y0 : PyTree
            The initial parameter guess.
        args : PyTree
            Additional static arguments passed to the objective function.
        lower : PyTree
            Lower bounds for `y0`. Only passed if `support_bounds` is True.
        upper : PyTree
            Upper bounds for `y0`. Only passed if `support_bounds` is True.
        has_aux: bool = False
            Specifies that `fn` returns a tuple of (`loss`, `aux`).
        **kwargs
            Runtime arguments forward to the solver backend.

        Returns
        -------
        tuple
            A tuple of `(MinimizePayload, metrics)`.
        """
        raise NotImplementedError
    

def minimize(
    objective: Callable[[PyTree, Any], Scalar],
    solver: optx.AbstractMinimiser | AbstractMinimizer,
    model: PyTree,
    *,
    args: PyTree = None,
    max_steps: int = 256,
    bounded: bool = None,
    has_aux: bool = False,
    filter_spec: Any = eqx.is_inexact_array,
    is_leaf: Any = is_constant,
    **kwargs
) -> OptimizeResults:
    """
    High-level minimization API for parax models.

    Automatically handles static/dynamic tree partitioning, topological 
    bijector mappings, and physical constraint bounds.
    """
    # Resolve the solver
    if not isinstance(solver, AbstractMinimizer):
        if not isinstance(solver, optx.AbstractMinimiser):
            raise Exception(f"Unknown solver in `parax.minimize`: {solver}")
        from parax.optimize.optimistix import OptimistixMinimise
        solver = OptimistixMinimise(solver)

    # Determine if we are doing a bounded optimization and extract the bounds if so
    if bounded is None:
        bounded = solver.supports_bounds
    elif bounded is True and not solver.supports_bounds:
        raise Exception("Solver does not support bounds but `bounded=True` was requested.")
    use_bounds = bounded and solver.supports_bounds
    if use_bounds:
        raise Exception("Bounded optimization not yet supported")

    # Setup problem and partition
    params, static = eqx.partition(model, filter_spec, is_leaf=is_leaf)
    
    # Define the array-level objective function
    def fn(params, _args):
        unwrapped_model = unwrap(eqx.combine(params, static))
        return objective(unwrapped_model, _args)

    # Run the minimization
    payload, metrics = solver.minimize(
        fn=fn,
        y0=params,
        args=args,
        has_aux=has_aux,
        max_steps=max_steps,
        **kwargs
    )

    # Reconstruct the final optimized model
    optimized_model = eqx.combine(payload.y, static)
    
    return OptimizeResults(
        model=optimized_model,
        objective=payload.fn_value,
        metrics=metrics, 
        aux=payload.aux,
    )