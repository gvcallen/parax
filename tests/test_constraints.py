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

try:
    from distreqx.bijectors import Identity
except ImportError:
    from parax._bijectors import Identity

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

# ==========================================
# Whitened base space
# ==========================================
#
# A bounded solver works directly in the base space, so that space carries the
# units the optimizer sees. Where a constraint has a finite extent the base is
# normalised to the unit box; where it does not, the base stays physical.


def test_interval_base_space_is_the_unit_box():
    """Interval exposes a normalised base, not the physical box."""
    interval = Interval(70.0, 80.0)

    assert jnp.allclose(interval.base_bounds[0], 0.0)
    assert jnp.allclose(interval.base_bounds[1], 1.0)

    # Midpoint of the interval sits at the midpoint of the base.
    assert jnp.allclose(interval.base_bijector.inverse(jnp.array(75.0)), 0.5)
    assert jnp.allclose(interval.base_bijector.forward(jnp.array(0.5)), 75.0)

    x = jnp.linspace(70.0, 80.0, 50)
    assert jnp.allclose(interval.base_bijector.forward(interval.base_bijector.inverse(x)), x)


def test_interval_base_space_is_unit_regardless_of_physical_scale():
    """
    The base is invariant to the units a parameter happens to be expressed in.

    This is the property that makes an isotropic trust region meaningful: a length
    in millimetres and the same length in metres must present the optimizer with
    the same geometry, rather than boxes differing by a factor of a thousand.
    """
    millimetres = Interval(70.0, 80.0)
    metres = Interval(0.07, 0.08)

    for fraction in (0.0, 0.25, 0.5, 1.0):
        assert jnp.allclose(
            millimetres.base_bijector.inverse(millimetres.base_bijector.forward(jnp.array(fraction))),
            metres.base_bijector.inverse(metres.base_bijector.forward(jnp.array(fraction))),
        )

    widths = [c.base_bounds[1] - c.base_bounds[0] for c in (millimetres, metres)]
    assert all(jnp.allclose(w, 1.0) for w in widths)


def test_interval_bijector_is_built_from_its_base():
    """
    The unconstrained bijector reuses the base, so the two spaces cannot drift apart.

    They differ only by the squashing step, which keeps step sizes comparable through
    the bulk of the interval while still mapping the whole real line inside the bounds.
    """
    interval = Interval(-5.0, 5.0)

    x = jnp.linspace(-100.0, 100.0, 500)
    mapped = interval.bijector.forward(x)
    assert jnp.all(mapped >= -5.0)
    assert jnp.all(mapped <= 5.0)

    # Going through the base explicitly must agree with the composed bijector.
    from distreqx.bijectors import Sigmoid
    assert jnp.allclose(interval.base_bijector.forward(Sigmoid().forward(x)), mapped)

    # The centre of the unconstrained space lands at the centre of the interval.
    assert jnp.allclose(interval.bijector.forward(jnp.array(0.0)), 0.0, atol=1e-6)


@pytest.mark.parametrize("constraint_factory", [
    lambda: RealLine(shape=()),
    lambda: GreaterThan(5.0),
    lambda: LessThan(5.0),
    lambda: Positive(),
    lambda: Negative(),
])
def test_infinite_domains_keep_a_physical_base(constraint_factory):
    """
    Constraints with a genuinely infinite domain are left alone.

    There is no finite extent to normalise against, so the base coincides with the
    physical space. A user wanting something else should build a Custom constraint.
    """
    constraint = constraint_factory()

    assert isinstance(constraint.base_bijector, Identity)
    for base, physical in zip(constraint.base_bounds, constraint.bounds):
        assert jnp.allclose(base, physical, equal_nan=True)


def test_transformed_constraint_composes_the_whitened_base():
    """A Transformed constraint inherits its inner constraint's whitened base."""
    base = Interval(1.0, 5.0)
    transformed = Transformed(base, ScalarAffine(shift=jnp.array(0.0), scale=jnp.array(-2.0)))

    assert jnp.allclose(transformed.base_bounds[0], 0.0)
    assert jnp.allclose(transformed.base_bounds[1], 1.0)

    # Base -> physical still lands inside the transformed physical bounds.
    lower, upper = transformed.bounds
    mapped = transformed.base_bijector.forward(jnp.linspace(0.0, 1.0, 25))
    assert jnp.all(mapped >= lower - 1e-6)
    assert jnp.all(mapped <= upper + 1e-6)
