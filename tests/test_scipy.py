import pytest
import jax.numpy as jnp
import equinox as eqx
import parax as prx
from parax.optimize import minimize_scipy

# ==========================================
# Dummy Models for Testing
# ==========================================

class SimpleModel(eqx.Module):
    x: jnp.ndarray
    y: jnp.ndarray

class ConstrainedModel(eqx.Module):
    w: prx.ParamLike

class PhysicalModel(eqx.Module):
    p: prx.ParamLike

class MixedModel(eqx.Module):
    active: prx.ParamLike
    fixed: prx.ParamLike
    metadata: str

# ==========================================
# Test Suite
# ==========================================

def test_unconstrained_minimization():
    """Test standard minimization across the real line with raw JAX arrays."""
    model = SimpleModel(x=jnp.array(0.0), y=jnp.array(0.0))
    
    def loss_fn(m, args):
        # Global minimum is at x=3, y=-2
        return (m.x - 3.0)**2 + (m.y + 2.0)**2

    opt_model, result = minimize_scipy(loss_fn, model)
    
    assert result.success
    assert jnp.allclose(opt_model.x, 3.0, atol=1e-4)
    assert jnp.allclose(opt_model.y, -2.0, atol=1e-4)


def test_constrained_minimization():
    """Test that SciPy strictly respects physical bounds using L-BFGS-B."""
    # Initialize at 1.0, bound strictly between 0.0 and 2.0
    model = ConstrainedModel(
        w=prx.Constrained(1.0, prx.Interval(0.0, 2.0))
    )
    
    def loss_fn(m, args):
        # Unconstrained global minimum is at 5.0
        return (m.w - 5.0)**2

    opt_model, result = minimize_scipy(loss_fn, model)
    
    # assert result.success
    # The optimizer should get as close to 5.0 as the bounds allow (which is 2.0)
    assert jnp.allclose(prx.unwrap(opt_model).w, 2.0, atol=1e-4)
    # Ensure the PyTree reconstructed the wrapper perfectly
    assert isinstance(opt_model.w, prx.Constrained)


def test_physical_scale_minimization():
    """Test that Physical parameters correctly map optimizer base space to scaled space."""
    # Scale = 10.0, must be > 0.
    model = PhysicalModel(
        p=prx.Physical(base_value=1.0, scale=10.0, constraint=prx.Positive())
    )
    
    def loss_fn(m, args):
        # m.p is unwrapped here, so it represents the scaled physical value.
        # We want the final physical value to hit 40.0.
        return (m.p - 40.0)**2
        
    opt_model, result = minimize_scipy(loss_fn, model)
    
    assert result.success
    # The fully evaluated value should be 40.0
    assert jnp.allclose(opt_model.p.value, 40.0, atol=1e-4)
    # To get a value of 40.0 with a scale of 10.0, the base space must be 4.0!
    assert jnp.allclose(opt_model.p.base, 4.0, atol=1e-4)


def test_static_partitioning():
    """Test that strings and fixed parameters are ignored but faithfully reconstructed."""
    model = MixedModel(
        active=prx.Param(0.0),
        fixed=prx.Fixed(10.0),
        metadata="test_string"
    )
    
    def loss_fn(m, args):
        return (m.active - 5.0)**2 + (m.fixed - 5.0)**2
        
    opt_model, result = minimize_scipy(loss_fn, model)
    
    assert jnp.allclose(opt_model.active.value, 5.0, atol=1e-4)
    assert jnp.allclose(opt_model.fixed.value, 10.0, atol=1e-4)
    assert opt_model.metadata == "test_string"


# def test_gradient_free_minimization():
#     """Test the use_grad=False fallback path (SciPy computes numerical jacobians)."""
#     model = SimpleModel(x=jnp.array(10.0), y=jnp.array(-10.0))
    
#     def loss_fn(m, args):
#         return (m.x - 3.0)**2 + (m.y + 2.0)**2

#     # Disable exact JAX gradients
#     opt_model, result = minimize_scipy(loss_fn, model, use_grad=False)
    
#     assert result.success
#     assert jnp.allclose(opt_model.x, 3.0, atol=1e-3)
#     assert jnp.allclose(opt_model.y, -2.0, atol=1e-3)