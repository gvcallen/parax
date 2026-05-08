import pytest
import jax
import jax.numpy as jnp

# Assuming your package is structured so these imports work
from parax.constraints import (
    RealLine,
    GreaterThan,
    LessThan,
    Interval,
    Positive,
    Negative,
    Transformed,
    Leafwise,
    Custom,
)
from distreqx.bijectors import ScalarAffine

def test_real_line():
    """Test RealLine bounds and identity mapping."""
    constraint = RealLine(shape=(2,))
    lower, upper = constraint.bounds
    
    assert jnp.all(lower == -jnp.inf)
    assert jnp.all(upper == jnp.inf)
    
    # Identity mapping shouldn't change the input
    x = jnp.array([-5.0, 5.0])
    assert jnp.allclose(constraint.bijector.forward(x), x)


def test_greater_than_and_positive():
    """Test lower bounding logic for GreaterThan and Positive."""
    gt = GreaterThan(5.0)
    pos = Positive()
    
    assert gt.bounds[0] == 5.0
    assert gt.bounds[1] == jnp.inf
    assert pos.bounds[0] == 0.0
    
    # Extreme numbers will hit the float32 precision ceiling and equal the bound exactly
    x_extreme = jnp.array(-100.0)
    assert gt.bijector.forward(x_extreme) >= 5.0
    assert pos.bijector.forward(x_extreme) >= 0.0

    # Moderate numbers should strictly respect the > inequality
    x_moderate = jnp.array(-2.0)
    assert gt.bijector.forward(x_moderate) > 5.0
    assert pos.bijector.forward(x_moderate) > 0.0


def test_less_than_and_negative():
    """Test the double-affine flip corner case used in LessThan."""
    lt = LessThan(10.0)
    neg = Negative()
    
    assert lt.bounds[0] == -jnp.inf
    assert lt.bounds[1] == 10.0
    assert neg.bounds[1] == 0.0
    
    # Extreme numbers hit precision limits
    x_extreme = jnp.array(100.0)
    assert lt.bijector.forward(x_extreme) <= 10.0
    assert neg.bijector.forward(x_extreme) <= 0.0

    # Moderate numbers maintain strict inequality
    x_moderate = jnp.array(2.0)
    assert lt.bijector.forward(x_moderate) < 10.0
    assert neg.bijector.forward(x_moderate) < 0.0


def test_interval():
    """Test that Interval strictly bounds values between lower and upper."""
    interval = Interval(-5.0, 5.0)
    assert interval.bounds[0] == -5.0
    assert interval.bounds[1] == 5.0
    
    # Test a wide range of real inputs to ensure they are squashed into [-5, 5]
    # (Using >= and <= to account for exact float32 boundary rounding at the extremes)
    x = jnp.linspace(-100.0, 100.0, 500)
    mapped = interval.bijector.forward(x)
    
    assert jnp.all(mapped >= -5.0)
    assert jnp.all(mapped <= 5.0)


def test_transformed_constraint_monotonic_decrease():
    """
    Test the corner case where a monotonically decreasing bijector 
    inverts the lower and upper bounds of the base constraint.
    """
    base = Interval(1.0, 5.0)
    # A bijector that multiplies by -2
    inverting_bijector = ScalarAffine(shift=jnp.array(0.0), scale=jnp.array(-2.0))
    
    transformed = Transformed(base, inverting_bijector)
    lower, upper = transformed.bounds
    
    # Original bounds (1, 5) mapped by -2 become (-2, -10).
    # The constraint should gracefully flip them to (-10, -2).
    assert jnp.allclose(lower, -10.0)
    assert jnp.allclose(upper, -2.0)


def test_tree_constraint_valid_pytree():
    """
    Test that TreeConstraint extracts bounds and bijectors for a correctly 
    contracted PyTree (where all leaves are constraints).
    """
    tree_of_constraints = {
        "a": Positive(),
        "nested": {
            "b": Interval(0.0, 1.0),
            "c": GreaterThan(5.0)
        }
    }
    
    tree_constraint = Leafwise(tree_of_constraints)
    lower_bounds, upper_bounds = tree_constraint.bounds
    
    # Constraint bounds should be extracted maintaining PyTree structure
    assert lower_bounds["a"] == 0.0
    assert upper_bounds["a"] == jnp.inf
    
    assert lower_bounds["nested"]["b"] == 0.0
    assert upper_bounds["nested"]["b"] == 1.0
    
    assert lower_bounds["nested"]["c"] == 5.0
    assert upper_bounds["nested"]["c"] == jnp.inf


def test_tree_constraint_empty_rejection():
    """Test that TreeConstraint rejects a structurally empty PyTree."""
    # An actually empty PyTree will yield 0 leaves, triggering the ValueError
    empty_tree = {}
    
    with pytest.raises(ValueError, match="The pytree of `tree` cannot be empty."):
        Leafwise(empty_tree)


def test_custom_constraint():
    """Test that CustomConstraint correctly stores custom logic."""
    custom_bijector = ScalarAffine(shift=jnp.array(10.0), scale=jnp.array(1.0))
    custom_bounds = (jnp.array(0.0), jnp.array(20.0))
    
    constraint = Custom(bijector=custom_bijector, bounds=custom_bounds)
    
    assert constraint.bounds == custom_bounds
    assert constraint.bijector is custom_bijector