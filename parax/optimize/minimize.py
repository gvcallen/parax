import abc
from typing import Any, Callable

from jaxtyping import PyTree, Scalar
import equinox as eqx
import optimistix as optx

from parax.filters import where_free_array
from parax.replace import tree_replace
from parax.constraints import RealLine

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
    fn: Callable[[PyTree, Any], Scalar],
    solver: optx.AbstractMinimiser | AbstractMinimizer,
    y0: PyTree,
    *,
    args: PyTree = None,
    max_steps: int = 256,
    has_aux: bool = False,
    param_spec: Any = is_free_param,
    **kwargs
) -> OptimizeResults:
    """
    High-level minimization API for parax models.

    Automatically handles static/dynamic tree partitioning, topological 
    bijector mappings, and physical constraint bounds.
    """
    if not isinstance(solver, AbstractMinimizer):
        if not isinstance(solver, optx.AbstractMinimiser):
            raise Exception(f"Unknown solver in `parax.minimize`: {solver}")
        from parax.optimize.optimistix import OptimistixMinimise
        solver = OptimistixMinimise(solver)

    # Setup problem
    params, static = eqx.partition(y0, param_spec, is_leaf=is_free_variable)
    bounded = solver.supports_bounds

    # Extract constraints, scales, and tree bijectors
    constraint = map_params(lambda x: x.constraint, params)
    raw_to_base_bij = ppt.raw_to_base_bijector(params)
    base_to_physical_bij = ppt.base_to_physical_bijector(params)

    # Calculate static arrays
    static_arrays = ppt.physical_values(static)

    if bounded:
        lower, upper = constraint.bounds
        y0 = ppt.base_values(params)
    else:
        lower, upper = None, None
        y0 = ppt.raw_values(params)

    def _array_objective_fn(y, _args):
        if not bounded:
            base_values = raw_to_base_bij.forward(y)
        else:
            base_values = y
        
        physical_params = base_to_physical_bij.forward(base_values)
        model_arrays = eqx.combine(physical_params, static_arrays)

        return fn(model_arrays, _args)

    # Run the minimization
    payload, metrics = solver.minimize(
        fn=_array_objective_fn,
        y0=y0,
        args=args,
        lower=lower,
        upper=upper,
        has_aux=has_aux,
        max_steps=max_steps,
        **kwargs
    )

    if bounded:
        final_raw_values = raw_to_base_bij.inverse(payload.y)
    else:
        final_raw_values = payload.y

    # Return the result
    optimized_params = tree_replace(params, raw_value=final_raw_values, is_leaf=is_free_variable)
    optimized_model = eqx.combine(optimized_params, static, is_leaf=is_free_variable)
    
    return OptimizeResults(
        model=optimized_model,
        final_value=payload.fn_value,
        metrics=metrics, 
        aux=payload.aux,
    )