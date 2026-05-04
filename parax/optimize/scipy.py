from typing import Any, Callable, Dict, Optional, Tuple
from jaxtyping import PyTree

import scipy.optimize as opt
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.flatten_util import ravel_pytree

from parax.unwrappables import unwrap
from parax.bounded import tree_bounded_base, tree_bounded_bounds, tree_bounded_replace, tree_bounded_convert
from parax.filters import is_constant

def minimize_scipy(
    fn: Callable[[Any, tuple], jax.Array], 
    y0: PyTree, 
    args: tuple = (), 
    method: str = "L-BFGS-B",
    use_grad: bool = True,
    options: Optional[Dict[str, Any]] = None,
    filter_spec: Any = eqx.is_inexact_array,
    is_leaf: Optional[Callable[[Any], bool]] = is_constant,
    **kwargs
) -> Tuple[PyTree, opt.OptimizeResult]:
    """
    Minimizes a PyTree using SciPy optimization.

    Performs partitioning and returns the full, optimized PyTree.

    Note that, although this function is general, it has native support
    for nodes that implement the `parax.AbstractBounded` interface. That is,
    it will automatically convert them to base space and extract their bounds
    using `parax.tree_base` and `parax.tree_bounds`, and then reconstruct
    them after optimization. This include both `parax.Constrained` and
    `parax.Physical` variables.

    Args:
        fn (Callable): The objective function to minimize. It must take the 
            unwrapped model and `args` as arguments, and return a scalar JAX array.
        y0 (Any): The initial parameters to optimize.
        args (tuple, optional): Additional arguments passed to the objective function. 
            Defaults to ().
        method (str, optional): The type of solver. Should be one of the methods 
            supported by `scipy.optimize.minimize`. Defaults to "L-BFGS-B".
        use_grad (bool, optional): Whether to use JAX's automatic differentiation to 
            compute exact gradients for the optimizer. If False, SciPy will attempt to 
            approximate gradients numerically. Defaults to True.
        options (dict, optional): A dictionary of solver options (e.g., `maxiter`, `ftol`). 
            Defaults to None.
        filter_spec (Any, optional): The specification used by `equinox.partition` to 
            filter the arrays to optimize. Defaults to `eqx.is_inexact_array`.
        is_leaf (Callable, optional): A function specifying custom leaf nodes during 
            PyTree traversal and partitioning. Defaults to `parax.is_constant`.
        **kwargs: Additional keyword arguments forwarded directly to `scipy.optimize.minimize`.

    Returns:
        tuple: A tuple containing:
            - The fully reconstructed and optimized parameters.
            - The `scipy.optimize.OptimizeResult` object containing solver metrics.
    """
    # 1. Extract the numerical base space and bounds
    base_model = tree_bounded_base(y0)
    lower_model, upper_model = tree_bounded_bounds(y0)

    gradient_free_methods = {'nelder-mead', 'powell', 'cobyla'}
    use_grad = use_grad and (method.lower() not in gradient_free_methods)
    
    # 2. Partition out constants/strings so we only pass target arrays to the optimizer
    params, static = eqx.partition(base_model, filter_spec, is_leaf=is_leaf)
    lower_params, _ = eqx.partition(lower_model, filter_spec, is_leaf=is_leaf)
    upper_params, _ = eqx.partition(upper_model, filter_spec, is_leaf=is_leaf)
    
    # 3. Flatten into 1D vectors for SciPy
    flat_params, unflatten_fn = ravel_pytree(params)
    flat_lower, _ = ravel_pytree(lower_params)
    flat_upper, _ = ravel_pytree(upper_params)
    
    # Format bounds for SciPy: a list of (lower, upper) tuples
    scipy_bounds = list(zip(flat_lower.tolist(), flat_upper.tolist()))
    
    # 4. Define the 1D objective function bridging SciPy and JAX
    @jax.jit
    def flat_objective(flat_x):
        base_p = unflatten_fn(flat_x)
        current_base_model = eqx.combine(base_p, static)
        projected_model = tree_bounded_convert(current_base_model, y0)
        full_model = unwrap(projected_model)
        return fn(full_model, args)
       
    # 5. Compile the objective and (optional) gradient functions
    if use_grad:
        val_and_grad_fn = jax.jit(jax.value_and_grad(flat_objective))
        
        def scipy_fun(x_numpy):
            # Evaluate JAX function and return standard floats/numpy arrays to SciPy
            val, grad = val_and_grad_fn(jnp.asarray(x_numpy))
            return val.item(), jax.device_get(grad)
    else:
        val_fn = jax.jit(flat_objective)
        
        def scipy_fun(x_numpy):
            val = val_fn(jnp.asarray(x_numpy))
            return val.item()
            
    # 6. Run the optimization
    result = opt.minimize(
        fun=scipy_fun,
        x0=jax.device_get(flat_params),
        method=method,
        jac=True if use_grad else None,
        bounds=scipy_bounds,
        options=options,
        **kwargs
    )
    
    # 7. Reconstruct the final optimized model
    opt_base_params = unflatten_fn(jnp.asarray(result.x))
    opt_base_model = eqx.combine(opt_base_params, static)
    opt_model = tree_bounded_replace(y0, opt_base_model)
    
    return opt_model, result