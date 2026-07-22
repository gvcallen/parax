import pytest
import numpy as np
import jax.numpy as jnp
import distreqx.distributions as dists
import distreqx.bijectors as bij

from parax.constraints import (
    infer_distribution_constraint,
    RealLine,
    Custom,
    Positive,
    Interval,
    Leafwise,
    Transformed,
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
    """Test that natively isotropic/multivariate distributions are explicitly whitened."""
    dist = dist_factory()
    constraint = infer_distribution_constraint(dist)
    
    # These are now Transformed constraints because we explicitly project 
    # them from a standard N(0, I) space using Shift/Scale/TriangularLinear.
    assert isinstance(constraint, Transformed)
    assert isinstance(constraint.base_constraint, RealLine)
    assert constraint.base_constraint.shape == dist.event_shape
    assert isinstance(constraint.transform_bijector, bij.Chain)


def test_meta_distribution_joint():
    """Test that Joint distributions map correctly to a Leafwise constraint tree."""
    dist = dists.Joint({
        "a": dists.Normal(loc=0., scale=1.),
        "b": BrokenDist() # Will fall back to RealLine
    })
    
    constraint = infer_distribution_constraint(dist)
    
    assert isinstance(constraint, Leafwise)
    # "a" is now perfectly whitened via a Transformed mapping
    assert isinstance(constraint.tree["a"], Transformed)
    assert isinstance(constraint.tree["a"].base_constraint, RealLine)
    
    # "b" fell back to a raw RealLine because it failed ICDF extraction
    assert isinstance(constraint.tree["b"], RealLine)


def test_meta_distribution_transformed():
    """Test that Transformed distributions wrap the base constraint."""
    base_dist = dists.Normal(loc=0., scale=1.)
    bijector = bij.Exp()
    dist = dists.Transformed(distribution=base_dist, bijector=bijector)
    
    constraint = infer_distribution_constraint(dist)
    assert isinstance(constraint, Transformed)


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

def test_icdf_constraint_exposes_a_whitened_base():
    """
    An inferred constraint carries a base that is bounded *and* already whitened.

    The quantile alone maps the probability-integral-transform space onto the physical
    one, so `[0, 1]` serves as the base: under that transform the prior is uniform
    there by construction, whatever the physical extent happens to be. Without it a
    bounded solver falls back to the physical box and gets no whitening at all.
    """
    dist = DummyICDFDist(lower=1e-9, upper=10.0)
    constraint = infer_distribution_constraint(dist)

    np.testing.assert_allclose(constraint.base_bounds[0], 0.0)
    np.testing.assert_allclose(constraint.base_bounds[1], 1.0)

    # The base map is the quantile on its own -- the whitening step without the
    # normal-space squashing that the full bijector adds on top.
    assert isinstance(constraint.base_bijector, Quantile)
    assert constraint.base_bijector.distribution == dist
    assert constraint.bijector.bijectors[0] is constraint.base_bijector


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
