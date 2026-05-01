import logging
from typing import Callable, Any, ClassVar

import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.optimize import minimize as scipy_minimize
import equinox as eqx
from tqdm.auto import tqdm

from parax.optimize.minimize import AbstractMinimizer, MinimizePayload

DEBUG_PRINT = False

class ScipyMinimize(AbstractMinimizer):
    """
    A JAX-wrapped optimizer using :func:`scipy.optimize.minimize`.

    Acts as an adapter layer between PyTrees and SciPy's required flat 1D NumPy arrays.
    Handles automatic differentiation via `jax.value_and_grad`.
    """
    method: str = eqx.field(static=True, default="L-BFGS-B")
    tol: float | None = eqx.field(static=True, default=None)
    options: dict = eqx.field(static=True, default_factory=dict)
    use_grad: bool = eqx.field(static=True, default=True)
    show_progress: bool = eqx.field(static=True, default=True)

    supports_bounds: ClassVar[bool] = True

    def minimize(self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: PyTree = None,
        lower: PyTree | None = None,
        upper: PyTree | None = None,
        has_aux: bool = False,
        max_steps: int = 1024,
        **kwargs
    ) -> tuple[MinimizePayload, PyTree]:
        
        method = self.method
        gradient_free_methods = {'nelder-mead', 'powell', 'cobyla'}
        use_grad = self.use_grad and (method.lower() not in gradient_free_methods)

        # 1. Flatten the PyTree 'y' into a 1D JAX array
        flat_y, unravel_fn = ravel_pytree(y0)
        
        scipy_options = dict(self.options)
        scipy_options.setdefault('maxiter', max_steps)
        
        # 2. Extract and flatten the bounds PyTrees natively
        scipy_bounds = None
        if lower is not None and upper is not None:
            flat_lower, _ = ravel_pytree(lower)
            flat_upper, _ = ravel_pytree(upper)
            scipy_bounds = list(zip(np.array(flat_lower), np.array(flat_upper)))        

        # 3. Define the internal JAX objective
        def flat_fn(_flat_y, _args):
            return fn(unravel_fn(_flat_y), _args)
            
        val_and_grad_fn = jax.jit(jax.value_and_grad(flat_fn, has_aux=has_aux))

        # State containers to capture the final loss and auxiliary data
        current_loss = [np.inf]
        current_aux = [None]
        nan_logged = [False] 

        def objective_with_grad(x_np):
            if has_aux:
                (loss, aux), grad = val_and_grad_fn(jnp.array(x_np), args)
                current_aux[0] = aux
            else:
                loss, grad = val_and_grad_fn(jnp.array(x_np), args)
                
            loss_np = np.asarray(loss, dtype=np.float64)
            grad_np = np.asarray(grad, dtype=np.float64)
            current_loss[0] = loss_np
            
            if not nan_logged[0] and np.any(np.isnan(loss_np)):
                logging.warning("Loss value was NaN")
                nan_logged[0] = True
            
            if not nan_logged[0] and np.any(np.isnan(grad_np)):
                logging.warning("Loss gradients were NaN")
                nan_logged[0] = True
            
            return loss_np, grad_np

        def objective_no_grad(x_np):
            if has_aux:
                loss, aux = flat_fn(jnp.array(x_np), args)
                current_aux[0] = aux
            else:
                loss = flat_fn(jnp.array(x_np), args)
                
            loss_np = np.asarray(loss, dtype=np.float64)
            current_loss[0] = loss_np
            return loss_np

        obj_func = objective_with_grad if use_grad else objective_no_grad

        # 4. Setup the progress bar and callback
        pbar = None
        if self.show_progress:
            maxiter = scipy_options.get("maxiter", None)
            pbar = tqdm(total=maxiter, desc=f"SciPy {method}")

        def callback(*cb_args, **cb_kwargs):
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=f"{current_loss[0]:.3g}")

        # 5. Optimize on the host (CPU)
        try:
            res = scipy_minimize(
                obj_func, 
                np.array(flat_y), 
                jac=use_grad, 
                method=method,
                tol=self.tol,
                bounds=scipy_bounds,  
                options=scipy_options,
                callback=callback,
                **kwargs,
            )
        finally:
            if pbar is not None:
                pbar.close()

        # 6. Reconstruct the PyTree and cast metrics to JAX scalar arrays
        optimized_y = unravel_fn(jnp.array(res.x))

        payload = MinimizePayload(optimized_y, jnp.array(res.fun), aux=current_aux[0])

        metrics = {
            "success": jnp.array(res.success, dtype=bool),
            "num_steps": jnp.array(getattr(res, 'nit', 0), dtype=jnp.int32),
            "num_evals": jnp.array(getattr(res, 'nfev', 0), dtype=jnp.int32),
            "message": str(getattr(res, 'message', '')), 
            "loss": jnp.array(res.fun, dtype=float)
        }


        return payload, metrics