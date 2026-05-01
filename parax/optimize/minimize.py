import abc
from typing import Any, Callable, TypeVar

from jaxtyping import PyTree, Scalar
import equinox as eqx

import parax.tree as tree
from parax.filters import is_free_param
from parax.optimize.results import OptimizeResults

OptimistixAbstractMinimiser = TypeVar('OptimistixAbstractMinimiser')

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
    solver: OptimistixAbstractMinimiser | AbstractMinimizer,
    model: PyTree,
    args: PyTree = None,
    respect_bounds: bool = True,
    max_steps: int = 1024,
    has_aux: bool = False,
    param_filter: Any = is_free_param,
    **kwargs
) -> OptimizeResults:
    """
    High-level minimization API for parax models.

    Automatically handles static/dynamic tree partitioning, topological 
    bijector mappings, and physical constraint bounds.
    """
    if not isinstance(solver, AbstractMinimizer):
        import optimistix as optx
        if not isinstance(solver, optx.AbstractMinimiser):
            raise Exception(f"Unknown solver in `parax.minimize`: {solver}")
        from parax.optimize.optimistix import OptimistixMinimise
        solver = OptimistixMinimise(solver)

    # Setup problem
    params, static = eqx.partition(model, param_filter)
    bounded = respect_bounds and solver.supports_bounds

    # Extract raw values, constraints and scales
    constraint = tree.constraint(params)
    raw_values = tree.raw_values(params)
    base_to_physical_bijector = tree.base_to_physical_bijector(params)

    if bounded:
        lower, upper = constraint.bounds
    else:
        raw_to_base_bijector = constraint.bijector

    def raw_objective_fn(raw_values, _args):
        if not bounded:
            base_values = raw_to_base_bijector.forward(raw_values)
        else:
            base_values = raw_values
        
        physical_values = base_to_physical_bijector.forward(base_values)
        y0_with_physical_values = eqx.combine(physical_values, static)

        out = objective(y0_with_physical_values, _args)
        if has_aux:
            loss, aux = out
            return loss, aux
        return out        

    # Run the minimization
    payload, metrics = solver.minimize(
        fn=raw_objective_fn,
        y0=raw_values,
        args=args,
        lower=lower if bounded else None,
        upper=upper if bounded else None,
        has_aux=has_aux,
        max_steps=max_steps,
        **kwargs
    )

    # Return the result
    optimized_params = tree.replace_raw_values(params, payload.y)
    optimized_model = eqx.combine(optimized_params, static)
    return OptimizeResults(
        model=optimized_model,
        objective_value=payload.fn_value,
        metrics=metrics, 
        aux=payload.aux,
    )