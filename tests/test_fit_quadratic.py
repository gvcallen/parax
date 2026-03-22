import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx

import parax as prx
from parax.parameters import Free, Fixed

def test_fit_quadratic():
    # 1. Define the Parametric Model
    class Quadratic(eqx.Module):
        """A generic quadratic curve: y = a*x^2 + b*x + c"""
        
        a: prx.Parameter
        b: prx.Parameter
        c: prx.Parameter

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            return self.a * (x ** 2) + self.b * x + self.c
        
    model = Quadratic(a=Free(1.5), b=Free(0.5), c=Fixed(10.0))

    # 2. Generate some dummy "ground truth" data with noise
    x_true = jnp.linspace(-5.0, 5.0, 100)
    y_true = 3.0 * (x_true ** 2) - 2.0 * x_true + 10.0 # True a=3.0, b=-2.0
    y_true = y_true + jax.random.normal(jax.random.key(0), x_true.shape)

    # 3. Partition the model into free and fixed parameters
    params, static = prx.partition(model)

    # 4. Define the loss Function
    def loss_fn(params, args=None):
        model = eqx.combine(params, static)
        y_pred = model(x_true)
        return jnp.mean((y_pred - y_true)**2)

    # 5. Run the BFGS optimizer
    solver = optx.LBFGS(rtol=1e-6, atol=1e-6)
    solution = optx.minimise(
        fn=loss_fn,
        y0=params,
        solver=solver,
        args=(x_true, y_true, static),
    )

    # 6. Recombine to get the final fitted model
    fitted_model = eqx.combine(solution.value, static)
    final_loss = loss_fn(fitted_model)
    
    assert final_loss < 1.0