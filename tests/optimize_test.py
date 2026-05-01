# test_minimize.py
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx

# Assuming your library is structured like this:
from parax.parameter import Parameter
from parax.constraints import Interval
from parax.optimize.minimize import minimize
from parax.optimize import ScipyMinimize, OptimistixMinimise

# =====================================================================
# Base Models & Objective Functions for Testing
# =====================================================================

class ParabolaModel(eqx.Module):
    """A simple 2D quadratic bowl for testing optimization."""
    x: Parameter
    y: Parameter

@pytest.fixture
def initial_model():
    """Returns a model initialized far away from the true minimum."""
    return ParabolaModel(
        x=Parameter(0.0, name="x"),
        y=Parameter(0.0, name="y")
    )

@pytest.fixture
def bounded_model():
    """Returns a model where 'x' is constrained between [0, 1]."""
    return ParabolaModel(
        x=Parameter(0.5, constraint=Interval(0.0, 1.0), name="x"),
        y=Parameter(0.0, name="y")
    )

def parabola_loss(model: ParabolaModel, args) -> jax.Array:
    """Minimum is at x=3.0, y=-2.0"""
    # We optimize with respect to the physical value
    loss = (model.x - 3.0)**2 + (model.y + 2.0)**2
    return loss

def parabola_loss_with_aux(model: ParabolaModel, args):
    loss = parabola_loss(model, args)
    aux_data = {"distance_from_origin": jnp.sqrt(model.x**2 + model.y**2)}
    return loss, aux_data

# =====================================================================
# Tests: ScipyMinimize Wrapper
# =====================================================================

def test_scipy_unbounded_minimize(initial_model):
    """Test that SciPy successfully finds the unconstrained minimum."""
    solver = ScipyMinimize(method="L-BFGS-B")
    
    results = minimize(
        fn=parabola_loss,
        solver=solver,
        y0=initial_model
    )
    
    final_model = results.model
    
    # Check that it found the minimum (x=3, y=-2)
    assert jnp.allclose(final_model.x, 3.0, atol=1e-4)
    assert jnp.allclose(final_model.y, -2.0, atol=1e-4)
    assert jnp.allclose(results.final_value, 0.0, atol=1e-4)
    assert results.metrics["success"]

def test_scipy_bounded_minimize(bounded_model):
    """Test that SciPy respects physical constraints."""
    solver = ScipyMinimize(method="L-BFGS-B")
    
    results = minimize(
        fn=parabola_loss,
        solver=solver,
        y0=bounded_model,
    )
    
    final_model = results.model
    
    # The true minimum is x=3, but the constraint bounds it to x <= 1.0
    # Therefore, the optimizer should hit the boundary exactly at x=1.0.
    assert jnp.allclose(final_model.x, 1.0, atol=1e-4)
    assert jnp.allclose(final_model.y, -2.0, atol=1e-4)

# =====================================================================
# Tests: Optimistix Wrapper
# =====================================================================

def test_optimistix_unbounded_minimize(initial_model):
    """Test that Optimistix successfully finds the unconstrained minimum."""
    # Note: BFGS requires rtol/atol for its stopping criteria
    optx_solver = optx.BFGS(rtol=1e-5, atol=1e-5)
    solver = OptimistixMinimise(solver=optx_solver)
    
    results = minimize(
        fn=parabola_loss,
        solver=solver,
        y0=initial_model
    )
    
    final_model = results.model
    
    assert jnp.allclose(final_model.x, 3.0, atol=1e-4)
    assert jnp.allclose(final_model.y, -2.0, atol=1e-4)

def test_optimistix_rejects_bounds(bounded_model):
    """Test that Optimistix raises an error if handed bounds."""
    optx_solver = optx.BFGS(rtol=1e-5, atol=1e-5)
    solver = OptimistixMinimise(solver=optx_solver)
    
    with pytest.raises(ValueError, match="Optimistix minimise does not support physical bounds"):
        # We manually call the wrapper to inject bounds
        solver.minimize(
            fn=lambda y, args: 0.0, 
            y0=jnp.array([0.0]), 
            lower=jnp.array([-1.0]), 
            upper=jnp.array([1.0])
        )

# =====================================================================
# Tests: Auxiliary Data (has_aux)
# =====================================================================

def test_scipy_has_aux(initial_model):
    solver = ScipyMinimize(method="L-BFGS-B")
    
    results = minimize(
        fn=parabola_loss_with_aux,
        solver=solver,
        y0=initial_model,
        has_aux=True
    )
    
    # At optimum x=3, y=-2, the distance from origin is sqrt(9 + 4) = sqrt(13) ~ 3.605
    expected_distance = jnp.sqrt(13.0)
    
    assert results.aux is not None
    assert "distance_from_origin" in results.aux
    assert jnp.allclose(results.aux["distance_from_origin"], expected_distance, atol=1e-3)

def test_optimistix_has_aux(initial_model):
    optx_solver = optx.BFGS(rtol=1e-5, atol=1e-5)
    solver = OptimistixMinimise(solver=optx_solver)
    
    results = minimize(
        fn=parabola_loss_with_aux,
        solver=solver,
        y0=initial_model,
        has_aux=True
    )
    
    expected_distance = jnp.sqrt(13.0)
    assert results.aux is not None
    assert jnp.allclose(results.aux["distance_from_origin"], expected_distance, atol=1e-3)