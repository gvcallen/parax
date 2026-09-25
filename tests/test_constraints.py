import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Assuming your package is structured so these imports work
from parax.constraints import (
    RealLine,
    GreaterThan,
    LessThan,
    Interval,
    Positive,
    Negative,
    NonNegative,
    NonPositive,
    intersect,
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


def test_interval_round_trips_values_near_a_bound():
    """Values deep in the saturated tail of the sigmoid keep their precision."""
    bijector = Interval(0.0, 1.0).bijector
    y = jnp.array([1e-3, 1e-5, 1e-7, 1e-12])
    assert jnp.allclose(bijector.forward(bijector.inverse(y)), y, rtol=1e-6, atol=0.0)


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


# ==========================================
# Open and closed bounds
# ==========================================


def test_interval_is_closed_by_default():
    interval = Interval(0.0, 1.0)

    assert interval.closed == (True, True)
    assert not interval.is_outside(jnp.array(0.0))
    assert not interval.is_outside(jnp.array(1.0))
    assert interval.is_outside(jnp.array(1.5))


def test_interval_closed_per_bound():
    interval = Interval(0.0, 1.0, closed=(False, True))

    assert interval.closed == (False, True)
    assert interval.is_outside(jnp.array(0.0))
    assert not interval.is_outside(jnp.array(1.0))


def test_interval_closed_as_a_single_bool():
    interval = Interval(0.0, 1.0, closed=False)

    assert interval.closed == (False, False)
    assert interval.is_outside(jnp.array(0.0))
    assert interval.is_outside(jnp.array(1.0))
    assert not interval.is_outside(jnp.array(0.5))


def test_half_bounded_closedness():
    assert GreaterThan(5.0).closed == (True, True)
    assert GreaterThan(5.0, closed=False).closed == (False, False)
    assert GreaterThan(5.0, closed=(True, False)).closed == (True, False)
    assert LessThan(5.0).closed == (True, True)
    assert LessThan(5.0, closed=False).closed == (False, False)
    assert LessThan(5.0, closed=(False, True)).closed == (False, True)
    assert RealLine().closed == (True, True)

    assert not GreaterThan(5.0).is_outside(jnp.array(5.0))
    assert GreaterThan(5.0, closed=False).is_outside(jnp.array(5.0))
    assert not LessThan(5.0).is_outside(jnp.array(5.0))
    assert LessThan(5.0, closed=False).is_outside(jnp.array(5.0))


def test_positive_and_negative_are_open_at_zero():
    assert Positive().closed == (False, True)
    assert Negative().closed == (True, False)
    assert Positive().is_outside(jnp.array(0.0))
    assert Negative().is_outside(jnp.array(0.0))
    assert not Positive().is_outside(jnp.array(1e-30))


def test_non_negative_and_non_positive_are_closed():
    assert NonNegative().closed == (True, True)
    assert NonPositive().closed == (True, True)
    assert not NonNegative().is_outside(jnp.array(0.0))
    assert not NonPositive().is_outside(jnp.array(0.0))
    assert NonNegative().is_outside(jnp.array(-1e-30))
    assert NonPositive().is_outside(jnp.array(1e-30))


def test_non_negative_and_non_positive_keep_their_shape():
    assert NonNegative(shape=(3,)).bounds[0].shape == (3,)
    assert NonPositive(shape=(3,)).bounds[1].shape == (3,)


def test_intersection_open_wins_on_a_shared_bound():
    result = intersect(Interval(0.0, 10.0), Positive())

    assert isinstance(result, Interval)
    assert result.closed == (False, True)
    assert jnp.allclose(result.bounds[0], 0.0)
    assert jnp.allclose(result.bounds[1], 10.0)


def test_intersection_keeps_the_tighter_bounds_closedness():
    result = intersect(Interval(1.0, 10.0, closed=True), Positive())
    assert result.closed == (True, True)

    result = intersect(Interval(-1.0, 10.0, closed=True), Positive())
    assert result.closed == (False, True)


def test_intersection_resolves_to_the_named_half_bounded_constraints():
    assert isinstance(intersect(NonNegative(), GreaterThan(0.0)), NonNegative)
    assert isinstance(intersect(NonNegative(), Positive()), Positive)
    assert isinstance(intersect(NonPositive(), LessThan(0.0)), NonPositive)
    assert isinstance(intersect(NonPositive(), Negative()), Negative)


def test_transformed_reports_closedness():
    base = Interval(1.0, 5.0, closed=(True, False))

    increasing = Transformed(base, ScalarAffine(shift=jnp.array(0.0), scale=jnp.array(2.0)))
    assert increasing.closed == (True, False)

    # A decreasing bijector swaps the bounds, and their closedness with them.
    decreasing = Transformed(base, ScalarAffine(shift=jnp.array(0.0), scale=jnp.array(-2.0)))
    assert bool(decreasing.closed[0]) is False
    assert bool(decreasing.closed[1]) is True
    assert decreasing.is_outside(jnp.array(-10.0))
    assert not decreasing.is_outside(jnp.array(-2.0))


def test_leafwise_reports_closedness_per_leaf():
    tree = Leafwise({"a": Positive(), "b": Interval(0.0, 1.0, closed=(True, False))})
    lower_closed, upper_closed = tree.closed

    assert lower_closed == {"a": False, "b": True}
    assert upper_closed == {"a": True, "b": False}

    outside = tree.is_outside({"a": jnp.array(0.0), "b": jnp.array(0.0)})
    assert bool(outside["a"]) and not bool(outside["b"])


def test_custom_reports_closedness():
    bijector = ScalarAffine(shift=jnp.array(0.0), scale=jnp.array(1.0))
    bounds = (jnp.array(0.0), jnp.array(1.0))

    assert Custom(bijector, bounds).closed == (True, True)
    assert Custom(bijector, bounds, closed=(False, True)).closed == (False, True)

    tree_bounds = ({"a": jnp.array(0.0)}, {"a": jnp.array(1.0)})
    assert Custom(bijector, tree_bounds, closed=False).closed == ({"a": False}, {"a": False})


def test_custom_is_closed_by_default():
    bijector = ScalarAffine(shift=jnp.array(0.0), scale=jnp.array(1.0))

    lower_closed, upper_closed = Custom(bijector).closed
    assert bool(lower_closed) and bool(upper_closed)
    assert not Custom(bijector).is_outside(jnp.array(jnp.inf))
    assert Custom(bijector, closed=False).is_outside(jnp.array(jnp.inf))


# ==========================================
# Infinite bounds (ADR 0001)
# ==========================================


@pytest.mark.parametrize("constraint", [
    RealLine(),
    Positive(),
    NonNegative(),
    Negative(),
    NonPositive(),
    GreaterThan(1.0),
    LessThan(1.0),
    Interval(0.0, 1.0),
    Interval(-jnp.inf, jnp.inf),
    Custom(Identity()),
    Transformed(Interval(0.0, 1.0), Identity()),
], ids=lambda c: type(c).__name__)
def test_nan_is_outside_every_constraint(constraint):
    assert bool(constraint.is_outside(jnp.array(jnp.nan)))


@pytest.mark.parametrize("constraint, at", [
    (RealLine(), -jnp.inf),
    (RealLine(), jnp.inf),
    (Positive(), jnp.inf),
    (NonNegative(), jnp.inf),
    (Negative(), -jnp.inf),
    (NonPositive(), -jnp.inf),
    (GreaterThan(1.0), jnp.inf),
    (LessThan(1.0), -jnp.inf),
    (Custom(Identity()), jnp.inf),
    (Custom(Identity()), -jnp.inf),
], ids=lambda x: type(x).__name__ if not isinstance(x, float) else str(x))
def test_infinite_ends_are_closed_by_default(constraint, at):
    assert not bool(constraint.is_outside(jnp.array(at)))


def test_an_infinite_end_can_be_open():
    assert GreaterThan(1.0, closed=False).is_outside(jnp.array(1.0))
    assert GreaterThan(1.0, closed=False).is_outside(jnp.array(jnp.inf))

    half_open = GreaterThan(1.0, closed=(True, False))
    assert not half_open.is_outside(jnp.array(1.0))
    assert half_open.is_outside(jnp.array(jnp.inf))

    assert LessThan(1.0, closed=(False, True)).is_outside(jnp.array(-jnp.inf))


@pytest.mark.parametrize("lower, upper, y", [
    (0.0, jnp.inf, jnp.array([1e-6, 0.5, 3.0, 1e6])),
    (-jnp.inf, 0.0, jnp.array([-1e6, -3.0, -0.5, -1e-6])),
    (-jnp.inf, jnp.inf, jnp.array([-1e6, -3.0, 0.0, 2.5, 1e6])),
])
def test_interval_with_infinite_ends(lower, upper, y):
    interval = Interval(lower, upper)

    bijector = interval.bijector
    np.testing.assert_allclose(bijector.forward(bijector.inverse(y)), y, rtol=1e-6)
    assert jnp.all(jnp.isfinite(bijector.inverse(y)))

    # With no finite extent to normalise against, the base is the physical space.
    for base, physical in zip(interval.base_bounds, interval.bounds):
        np.testing.assert_array_equal(base, jnp.broadcast_to(physical, base.shape))
    np.testing.assert_allclose(interval.base_bijector.forward(y), y)
    np.testing.assert_allclose(interval.base_bijector.inverse(y), y)

    # An infinite end is closed by default, like the matching half-line or real line.
    assert not bool(interval.is_outside(jnp.array(upper)))
    assert not bool(interval.is_outside(jnp.array(lower)))
    assert bool(Interval(lower, upper, closed=False).is_outside(jnp.array(upper)))


def test_interval_with_mixed_ends_works_per_element():
    lower = jnp.array([2.0, -jnp.inf, -jnp.inf, 1.0])
    upper = jnp.array([4.0, 0.0, jnp.inf, jnp.inf])
    interval = Interval(lower, upper)
    y = jnp.array([3.5, -2.0, 5.0, 3.0])

    x = interval.bijector.inverse(y)
    assert jnp.all(jnp.isfinite(x))
    np.testing.assert_allclose(interval.bijector.forward(x), y, rtol=1e-6)

    # Every element stays inside its own bounds across the whole real line.
    for raw in (-50.0, 0.0, 50.0):
        mapped = interval.bijector.forward(jnp.full(4, raw))
        assert not jnp.any(Interval(lower, upper).is_outside(mapped))

    # The unit box where both ends are finite, the physical space elsewhere.
    np.testing.assert_array_equal(interval.base_bounds[0], jnp.array([0.0, -jnp.inf, -jnp.inf, 1.0]))
    np.testing.assert_array_equal(interval.base_bounds[1], jnp.array([1.0, 0.0, jnp.inf, jnp.inf]))
    np.testing.assert_allclose(
        interval.base_bijector.forward(jnp.array([0.75, -2.0, 5.0, 3.0])), y
    )


def test_interval_with_mixed_ends_has_finite_gradients():
    interval = Interval(jnp.array([2.0, -jnp.inf, -jnp.inf, 1.0]), jnp.array([4.0, 0.0, jnp.inf, jnp.inf]))
    y = jnp.array([3.5, -2.0, 5.0, 3.0])

    forward_grad = jax.grad(lambda x: jnp.sum(interval.bijector.forward(x)))(jnp.zeros(4))
    inverse_grad = jax.grad(lambda v: jnp.sum(interval.bijector.inverse(v)))(y)
    assert jnp.all(jnp.isfinite(forward_grad))
    assert jnp.all(jnp.isfinite(inverse_grad))

    _, log_det = interval.bijector.forward_and_log_det(jnp.zeros(4))
    assert jnp.all(jnp.isfinite(log_det))


def test_interval_with_infinite_ends_under_jit():
    @jax.jit
    def round_trip(lower, upper, y):
        bijector = Interval(lower, upper).bijector
        return bijector.forward(bijector.inverse(y))

    y = jnp.array([3.5, -2.0])
    np.testing.assert_allclose(
        round_trip(jnp.array([2.0, -jnp.inf]), jnp.array([4.0, jnp.inf]), y), y, rtol=1e-6
    )


def test_intersection_keeps_closedness_at_infinity():
    # Open at infinity wins over closed at infinity.
    result = intersect(GreaterThan(1.0, closed=False), NonNegative())
    assert result.closed == (False, False)
    assert result.is_outside(jnp.array(jnp.inf))

    result = intersect(Interval(-jnp.inf, jnp.inf, closed=False), RealLine())
    assert result.closed == (False, False)
    assert not isinstance(result, RealLine)
    assert result.is_outside(jnp.array(-jnp.inf))

    # Both closed at infinity stay closed there.
    result = intersect(RealLine(), Positive())
    assert isinstance(result, Positive)
    assert not result.is_outside(jnp.array(jnp.inf))
    assert isinstance(intersect(RealLine(), RealLine()), RealLine)


def test_intersection_names_a_class_only_when_its_closedness_matches():
    result = intersect(Positive(), GreaterThan(0.0, closed=(True, False)))
    assert not isinstance(result, Positive)
    assert result.closed == (False, False)
    assert jnp.allclose(result.bounds[0], 0.0)

    result = intersect(NonPositive(), LessThan(0.0, closed=(False, True)))
    assert not isinstance(result, NonPositive)
    assert result.closed == (False, True)


def test_transformed_real_line_through_sigmoid_is_closed():
    from parax.bijectors import Sigmoid

    unit = Transformed(RealLine(), Sigmoid())
    assert bool(unit.closed[0]) and bool(unit.closed[1])
    assert not unit.is_outside(jnp.array(0.0))
    assert not unit.is_outside(jnp.array(1.0))

    open_unit = Transformed(Interval(-jnp.inf, jnp.inf, closed=False), Sigmoid())
    assert open_unit.is_outside(jnp.array(0.0))
    assert open_unit.is_outside(jnp.array(1.0))


def test_transformed_through_a_mixing_bijector_accepts_its_image():
    """Bounds pushed through a mixing bijector are unknown, not NaN, so its image is inside."""
    from parax.bijectors import Chain, Shift, TriangularLinear

    bijector = Chain([
        Shift(jnp.array([1.0, -1.0])),
        TriangularLinear(matrix=jnp.array([[1.0, 0.0], [0.5, 2.0]])),
    ])
    constraint = Transformed(RealLine(shape=(2,)), bijector)

    lower, upper = constraint.bounds
    np.testing.assert_array_equal(lower, -jnp.inf)
    np.testing.assert_array_equal(upper, jnp.inf)
    y = bijector.forward(jnp.array([-5.0, -5.0]))
    assert not jnp.any(constraint.is_outside(y))


def test_transformed_leaves_a_mixed_element_unbounded():
    """An element the bijector mixes has no bounds of its own; an unmixed one keeps its image."""
    from parax.bijectors import TriangularLinear

    bijector = TriangularLinear(matrix=jnp.array([[1.0, 0.0], [-1.0, 1.0]]))
    constraint = Transformed(Interval(jnp.zeros(2), jnp.ones(2)), bijector)

    lower, upper = constraint.bounds
    np.testing.assert_array_equal(lower, jnp.array([0.0, -jnp.inf]))
    np.testing.assert_array_equal(upper, jnp.array([1.0, jnp.inf]))
    # (1, 0) maps to (1, -1), inside the image though below the second element's pushed bound.
    assert not jnp.any(constraint.is_outside(bijector.forward(jnp.array([1.0, 0.0]))))


def test_transformed_elementwise_bijector_keeps_pushed_bounds():
    """A vector-valued elementwise bijector over a scalar base still pushes bounds through."""
    constraint = Transformed(
        Interval(0.0, 1.0),
        ScalarAffine(shift=jnp.array([0.0, 1.0]), scale=jnp.array([2.0, -1.0])),
    )
    lower, upper = constraint.bounds
    np.testing.assert_allclose(lower, jnp.array([0.0, 0.0]))
    np.testing.assert_allclose(upper, jnp.array([2.0, 1.0]))
