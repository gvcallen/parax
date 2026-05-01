import sys
from unittest.mock import MagicMock
import dataclasses

import pytest
import jax.numpy as jnp
import equinox as eqx

# Adjust this import to match your actual module structure
from parax.parameter import Parameter, asparam, param
from parax.constraints import AbstractConstraint


# =========================================================================
# Mocks for testing constraints
# =========================================================================
class MockBijector:
    def forward(self, x):
        return x * 2.0

    def inverse(self, y):
        return y / 2.0


class MockConstraint(AbstractConstraint):
    bijector = MockBijector()
    bounds = (-jnp.inf, jnp.inf)


# =========================================================================
# Tests for Parameter Initialization & Validations
# =========================================================================
def test_parameter_init_missing_args():
    with pytest.raises(ValueError, match="Must provide either `base_value` or `raw_value`"):
        Parameter()


def test_parameter_init_mutually_exclusive_args():
    with pytest.raises(ValueError, match="Cannot provide both `base_value` and `raw_value`"):
        Parameter(base_value=1.0, raw_value=1.0)


def test_parameter_init_with_raw_value():
    p = Parameter(raw_value=5.0)
    assert jnp.allclose(p.raw_value, 5.0)
    assert jnp.allclose(p.base_value, 5.0)  # Default Unconstrained bijector is identity
    assert jnp.allclose(p.physical_value, 5.0)  # Default scale is 1.0


def test_parameter_scale_float_conversion():
    p = Parameter(base_value=2.0, scale=3.0)
    assert jnp.allclose(p.physical_value, 6.0)
    assert isinstance(p.scale, jnp.ndarray)


def test_parameter_string_scale_imports_unxt():
    # Mocking unxt to avoid needing the actual library installed for this test
    mock_unxt = MagicMock()
    mock_unxt.unit.return_value = "mocked_km_unit"
    sys.modules["unxt"] = mock_unxt

    try:
        p = Parameter(base_value=1.0, scale="km")
        mock_unxt.unit.assert_called_once_with("km")
        assert p.scale == "mocked_km_unit"
    finally:
        del sys.modules["unxt"]


def test_parameter_with_constraint_bijector():
    constraint = MockConstraint()
    p = Parameter(base_value=10.0, constraint=constraint)
    
    # Init logic: raw_value = bijector.inverse(base_value) -> 10.0 / 2.0 = 5.0
    assert jnp.allclose(p.raw_value, 5.0)
    
    # Property logic: base_value = bijector.forward(raw_value) -> 5.0 * 2.0 = 10.0
    assert jnp.allclose(p.base_value, 10.0)


# =========================================================================
# Tests for Properties
# =========================================================================
def test_parameter_properties():
    p = Parameter(base_value=jnp.ones((2, 3)))
    
    assert p.shape == (2, 3)
    assert p.size == 6
    assert p.unit is None


# =========================================================================
# Tests for Interactive Math Dunders
# =========================================================================
def test_parameter_math_operations():
    p = Parameter(base_value=2.0, scale=3.0)  # physical_value = 6.0

    # __jax_array__
    assert jnp.allclose(jnp.asarray(p), 6.0)
    
    # Add / Radd
    assert jnp.allclose(p + 4.0, 10.0)
    assert jnp.allclose(4.0 + p, 10.0)
    
    # Sub / Rsub
    assert jnp.allclose(p - 2.0, 4.0)
    assert jnp.allclose(2.0 - p, -4.0)
    
    # Mul / Rmul
    assert jnp.allclose(p * 2.0, 12.0)
    assert jnp.allclose(2.0 * p, 12.0)
    
    # Div / Rdiv
    assert jnp.allclose(p / 2.0, 3.0)
    assert jnp.allclose(12.0 / p, 2.0)
    
    # Pow
    assert jnp.allclose(p ** 2, 36.0)


# =========================================================================
# Tests for Utility Functions (asparam, field)
# =========================================================================
def test_asparam():
    # Test wrapping a primitive
    p1 = asparam(5.0)
    assert isinstance(p1, Parameter)
    assert jnp.allclose(p1.base_value, 5.0)

    # Test identity pass-through
    p2 = asparam(p1)
    assert p1 is p2


def test_parameter_field():
    # Define a dummy equinox module using the custom field
    class DummyModel(eqx.Module):
        p_required: Parameter = param(scale=2.0)
        p_default: Parameter = param(default=2.0, scale=4.0)

    # Instantiate triggering the converter
    model = DummyModel(p_required=3.0)
    
    assert isinstance(model.p_default, Parameter)
    assert isinstance(model.p_required, Parameter)
    
    # Check physical values (base * scale)
    assert jnp.allclose(model.p_default.physical_value, 8.0)
    assert jnp.allclose(model.p_required.physical_value, 6.0)


def test_parameter_field_with_explicit_parameter():
    class DummyModel(eqx.Module):
        p: Parameter = param(scale=2.0)

    # If the user passes an explicit Parameter, it should bypass the default field metadata
    explicit_p = Parameter(base_value=1.0, scale=10.0)
    model = DummyModel(p=explicit_p)
    
    assert model.p is explicit_p
    assert jnp.allclose(model.p.physical_value, 10.0)