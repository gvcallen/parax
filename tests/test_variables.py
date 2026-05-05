import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

# Assuming your package is structured so these imports work
from parax.variables import (
    AbstractVariable,
    Tagged,
    Derived,
    Constrained,
    Fixed,
    tagged,
    derived,
    constrained,
)
from parax.constraints import Positive


# ==========================================
# Core Variable & Dunder Tests
# ==========================================

def test_param_and_dunders():
    """
    Test basic parameter instantiation and prove that mathematical 
    dunder methods instantly strip wrappers and return standard arrays.
    """
    p = Tagged(raw_value=2.0, metadata={"name": "test_param"})
    
    assert p.shape == ()
    assert p.metadata["name"] == "test_param"
    
    # Forward math
    result_add = p + 3.0
    assert not isinstance(result_add, AbstractVariable)
    assert jnp.allclose(result_add, 5.0)
    
    # Reverse math
    result_rmul = 4.0 * p
    assert not isinstance(result_rmul, AbstractVariable)
    assert jnp.allclose(result_rmul, 8.0)
    
    # Array indexing/iterating
    p_arr = Tagged(jnp.array([1.0, 2.0, 3.0]))
    assert p_arr[1] == 2.0
    assert len(p_arr) == 3


def test_derived():
    """Test that Derived transforms its raw value dynamically."""
    d = Derived(raw_value=2.0, fn=lambda x: x ** 3)
    
    # Unwrapping or accessing .value should trigger the lambda
    assert jnp.allclose(d.value, 8.0)
    assert jnp.allclose(d.unwrap(), 8.0)


# ==========================================
# Constrained & Physical Tests
# ==========================================

def test_constrained_initialization_modes():
    """
    Test the mutually exclusive value/raw_value initialization to ensure 
    the inverse bijector properly calculates raw_value from a target value.
    """
    # Mode 1: Initialized via raw optimizer space
    c_raw = Constrained(raw_value=0.0, constraint=Positive())
    # softplus(0.0) is approx 0.693, not 0.0
    assert c_raw.value > 0.0 
    
    # Mode 2: Initialized via target physical value
    target = jnp.array(5.0)
    c_val = Constrained(value=target, constraint=Positive())
    # The constraint should have worked backwards to find the right raw_value
    assert jnp.allclose(c_val.value, target)

    # Mode 3: Invalid initializations
    with pytest.raises(ValueError, match="Must provide either"):
        Constrained(constraint=Positive())
        
    with pytest.raises(ValueError, match="Cannot provide both"):
        Constrained(value=5.0, raw_value=0.0, constraint=Positive())


# ==========================================
# Fixed Wrapper Tests
# ==========================================

def test_fixed_stops_gradients():
    """Mathematically prove that Fixed disconnects the gradient graph."""
    def loss_fn(raw_val):
        # Create a variable, then fix it
        p = Tagged(raw_value=raw_val)
        f = Fixed(p)
        return f.value ** 2

    # Normally d(x^2)/dx = 2x. But because of Fixed, it should be 0.
    grad_val = jax.grad(loss_fn)(3.0)
    assert jnp.allclose(grad_val, 0.0)


# ==========================================
# Dataclass & PyTree Helpers
# ==========================================

def test_dataclass_helpers():
    """
    Test that the eqx.field helpers successfully cast raw values into 
    variables, but gracefully passthrough existing variables to prevent 
    double-wrapping.
    """
    class TestModel(eqx.Module):
        # Python dataclass rules: fields without defaults MUST come before fields with defaults!
        c_val: AbstractVariable = constrained(constraint=Positive())
        d_val: AbstractVariable = derived(fn=jnp.exp)
        p_val: AbstractVariable = tagged(default=1.0)

    # 1. Provide raw floats (Converters should wrap them)
    model1 = TestModel(c_val=5.0, d_val=2.0)
    assert isinstance(model1.p_val, Tagged)
    assert isinstance(model1.c_val, Constrained)
    assert isinstance(model1.d_val, Derived)
    
    assert jnp.allclose(model1.p_val.value, 1.0) # default used
    assert jnp.allclose(model1.c_val.value, 5.0) # init via `value` due to kwarg pass
    assert jnp.allclose(model1.d_val.raw_value, 2.0)

    # 2. Provide already instantiated variables (Converters should passthrough)
    existing_param = Tagged(10.0)
    model2 = TestModel(c_val=1.0, d_val=1.0, p_val=existing_param)
    
    # It should be the exact instance, not Param(Param(10.0))
    assert model2.p_val is existing_param