import pytest
import numpy as np
import jax.numpy as jnp
import parax.distributions as dists
import parax.bijectors as bij

from parax.constraints import (
    infer_distribution_constraint,
    RealLine,
    Custom,
    Positive,
    Interval,
    Leafwise,
)
from parax._bijectors import NormalCDF, Quantile

# ==========================================
# Mocks & Dummies
# ==========================================

class DummyICDFDist:
    """A minimal mock distribution to test the Copula/ICDF intercept logic."""
    def __init__(self, lower, upper, shape=()):
        self._lower = jnp.asarray(lower)
        self._upper = jnp.asarray(upper)
        self.event_shape = shape

    def icdf(self, value: float):
        if value == 0.0:
            return self._lower
        elif value == 1.0:
            return self._upper
        raise ValueError("Only 0.0 and 1.0 are supported in this dummy.")


class BrokenDist:
    """A mock distribution that fails ICDF evaluation to test the last-resort fallback."""
    def __init__(self, shape=()):
        self.event_shape = shape

    def icdf(self, value: float):
        raise NotImplementedError("icdf not implemented")

# ==========================================
# Test Suites
# ==========================================

@pytest.mark.parametrize("dist_factory", [
    (lambda: dists.Normal(loc=0., scale=1.)),
    (lambda: dists.Logistic(loc=0., scale=1.)),
    (lambda: dists.MultivariateNormalDiag(loc=jnp.zeros(2), scale_diag=jnp.ones(2))),
])
def test_explicit_whitening_unconstrained(dist_factory):
    """
    Natively loc-scale distributions whiten raw space explicitly, while their base
    space is their support: the real line, left as it is.
    """
    dist = dist_factory()
    constraint = infer_distribution_constraint(dist)

    assert isinstance(constraint.bijector, bij.Chain)
    assert isinstance(constraint.base_bijector, bij.Identity)
    for bound, sign in zip(constraint.base_bounds, (-1, 1)):
        assert bound.shape == dist.event_shape
        assert jnp.all(bound == sign * jnp.inf)
    assert constraint.closed == (False, False)


def test_meta_distribution_joint():
    """Test that Joint distributions map correctly to a Leafwise constraint tree."""
    dist = dists.Joint({
        "a": dists.Normal(loc=0., scale=1.),
        "b": BrokenDist() # Will fall back to RealLine
    })
    
    constraint = infer_distribution_constraint(dist)
    
    assert isinstance(constraint, Leafwise)
    # "a" is whitened in raw space, with the real line as its base
    assert isinstance(constraint.tree["a"].base_bijector, bij.Identity)
    
    # "b" fell back to a raw RealLine because it failed ICDF extraction
    assert isinstance(constraint.tree["b"], RealLine)


def test_meta_distribution_transformed():
    """Test that Transformed distributions wrap the base constraint."""
    base_dist = dists.Normal(loc=0., scale=1.)
    bijector = bij.Exp()
    dist = dists.Transformed(distribution=base_dist, bijector=bijector)
    
    constraint = infer_distribution_constraint(dist)

    # The flow's support is (0, inf): half-bounded, so the base is the support itself.
    x = jnp.array([0.0, 1.0, 1e6])
    np.testing.assert_array_equal(constraint.base_bijector.forward(x), x)
    np.testing.assert_allclose(constraint.base_bounds[0], 0.0)
    np.testing.assert_array_equal(constraint.base_bounds[1], jnp.inf)
    assert jnp.allclose(constraint.bijector.forward(jnp.array(0.0)), 1.0)


def test_icdf_generates_copula_constraint():
    """
    Test that ANY distribution with a valid ICDF (like our Dummy) 
    is automatically whitened using the Copula transformation instead of raw bounds.
    """
    dist = DummyICDFDist(lower=5.0, upper=jnp.inf)
    constraint = infer_distribution_constraint(dist)
    
    # It should no longer be GreaterThan, but a perfectly whitened Custom constraint
    assert isinstance(constraint, Custom)
    
    # The bounds should still perfectly match the physical ICDF edges
    np.testing.assert_array_equal(constraint.bounds[0], jnp.array(5.0))
    np.testing.assert_array_equal(constraint.bounds[1], jnp.inf)
    
    # Verify the Right-to-Left TFP-style composition: NormalCDF -> Quantile
    assert isinstance(constraint.bijector, bij.Chain)
    assert len(constraint.bijector.bijectors) == 2
    assert isinstance(constraint.bijector.bijectors[0], Quantile)
    assert isinstance(constraint.bijector.bijectors[1], NormalCDF)
    
    # Ensure the Quantile bijector is targeting the exact original distribution
    assert constraint.bijector.bijectors[0].distribution == dist


def test_last_resort_fallback():
    """Test that exceptions during ICDF evaluation default safely to RealLine."""
    dist = BrokenDist(shape=(2, 2))
    constraint = infer_distribution_constraint(dist)
    
    # Since BrokenDist fails the ICDF try/except, it falls through to the end
    assert isinstance(constraint, RealLine)
    assert constraint.shape == (2, 2)

def test_icdf_constraint_base_is_the_unit_box_over_a_finite_support():
    """
    A support with two finite bounds gets the unit box as its base, mapped affinely
    onto the support: the base comes from the support, not the prior.
    """
    dist = DummyICDFDist(lower=1e-9, upper=10.0)
    constraint = infer_distribution_constraint(dist)

    np.testing.assert_allclose(constraint.base_bounds[0], 0.0)
    np.testing.assert_allclose(constraint.base_bounds[1], 1.0)

    base = jnp.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose(
        constraint.base_bijector.forward(base), 1e-9 + base * (10.0 - 1e-9), rtol=1e-6)


def test_icdf_constraint_base_is_the_support_when_half_bounded():
    dist = DummyICDFDist(lower=5.0, upper=jnp.inf)
    constraint = infer_distribution_constraint(dist)

    np.testing.assert_array_equal(constraint.base_bounds[0], 5.0)
    np.testing.assert_array_equal(constraint.base_bounds[1], jnp.inf)
    x = jnp.array([5.0, 6.0, 1e6])
    np.testing.assert_array_equal(constraint.base_bijector.forward(x), x)


def test_uniform_support_and_base_are_closed():
    constraint = infer_distribution_constraint(
        dists.Uniform(low=jnp.array(0.0), high=jnp.array(10.0)))

    assert bool(constraint.closed[0]) and bool(constraint.closed[1])
    assert not constraint.is_outside(jnp.array(0.0))
    assert not constraint.is_outside(jnp.array(10.0))

    np.testing.assert_array_equal(constraint.base_bounds[0], 0.0)
    np.testing.assert_array_equal(constraint.base_bounds[1], 1.0)
    # The base edge lands exactly on the support's edge.
    assert constraint.base_bijector.forward(jnp.array(0.0)) == 0.0
    np.testing.assert_allclose(constraint.base_bijector.forward(jnp.array(1.0)), 10.0)


def test_support_is_open_where_the_density_vanishes():
    """LogNormal's support reaches 0 but excludes it, and never reaches infinity."""
    constraint = infer_distribution_constraint(dists.LogNormal(0.0, 1.0))

    assert not bool(constraint.closed[0])
    assert not bool(constraint.closed[1])
    assert constraint.is_outside(jnp.array(0.0))


def test_random_on_a_bound_has_an_infinite_raw_value():
    """No silent clip on the way to raw: an on-bound value maps to -inf, like `Bounded`."""
    from parax.variables import Random

    variable = Random(dists.Uniform(low=jnp.array(0.0), high=jnp.array(10.0)), value=0.0)
    assert jnp.isneginf(variable.raw_value)

    variable = Random(dists.Uniform(low=jnp.array(0.0), high=jnp.array(10.0)), value=10.0)
    assert jnp.isposinf(variable.raw_value)


def test_icdf_base_is_invariant_to_physical_scale():
    """
    Two distributions differing only in units present the optimizer with one geometry.

    This is what stops a parameter measured in millimetres from dominating one
    measured in metres purely through its numerical extent.
    """
    wide = infer_distribution_constraint(dists.Uniform(low=jnp.array(70.0), high=jnp.array(80.0)))
    narrow = infer_distribution_constraint(dists.Uniform(low=jnp.array(0.07), high=jnp.array(0.08)))

    for constraint in (wide, narrow):
        np.testing.assert_allclose(np.asarray(constraint.base_bounds[0]).ravel()[0], 0.0)
        np.testing.assert_allclose(np.asarray(constraint.base_bounds[1]).ravel()[0], 1.0)

    # The same base coordinate maps to the same fraction of each physical range.
    for fraction in (0.25, 0.5, 0.75):
        for constraint in (wide, narrow):
            lower = np.asarray(constraint.bounds[0]).ravel()[0]
            upper = np.asarray(constraint.bounds[1]).ravel()[0]
            physical = np.asarray(
                constraint.base_bijector.forward(jnp.array(fraction))).ravel()[0]
            np.testing.assert_allclose(
                (physical - lower) / (upper - lower), fraction, rtol=1e-5)


def test_wide_prior_does_not_dominate_the_base_geometry():
    """
    A regression guard for the failure this whitening exists to prevent.

    An unjustifiably wide prior used to cost nothing, because the optimizer never saw
    the prior's extent -- it worked in physical units and a wide box simply went
    unnoticed. Every base box must now be unit width, so no single parameter can set
    the scale of the search for all the others.
    """
    constraints = [
        infer_distribution_constraint(dists.Uniform(low=jnp.array(lo), high=jnp.array(hi)))
        for lo, hi in [(70.0, 80.0), (0.0, 0.05), (2.5e-9, 2.5e-1), (1e-9, 10.0)]
    ]
    widths = [
        np.asarray(c.base_bounds[1]).ravel()[0] - np.asarray(c.base_bounds[0]).ravel()[0]
        for c in constraints
    ]
    np.testing.assert_allclose(widths, np.ones(len(widths)))


@pytest.mark.parametrize("concentration, closed", [(1.0, True), (2.0, False), (0.5, False)])
def test_beta_support_is_closed_only_where_the_density_is_positive_and_finite(concentration, closed):
    constraint = infer_distribution_constraint(dists.Beta(concentration, concentration))

    assert bool(constraint.closed[0]) is closed
    assert bool(constraint.closed[1]) is closed
    np.testing.assert_array_equal(constraint.base_bounds[0], 0.0)
    np.testing.assert_array_equal(constraint.base_bounds[1], 1.0)
