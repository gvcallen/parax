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