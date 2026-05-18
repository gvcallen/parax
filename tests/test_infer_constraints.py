import pytest
import numpy as np
import jax.numpy as jnp
import distreqx.distributions as dists
import distreqx.bijectors as bij

from parax.constraints import (
    infer_distribution_constraint,
    RealLine,
    Positive,
    Negative,
    Interval,
    GreaterThan,
    LessThan,
    Leafwise,
    Transformed,
)

# ==========================================
# Mocks & Dummies
# ==========================================

class DummyICDFDist:
    """A minimal mock distribution to test the ICDF fallback logic."""
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

@pytest.mark.parametrize("dist_factory, expected_type", [
    (lambda: dists.Normal(loc=0., scale=1.), RealLine),
    (lambda: dists.Logistic(loc=0., scale=1.), RealLine),
    (lambda: dists.MultivariateNormalDiag(loc=jnp.zeros(2), scale_diag=jnp.ones(2)), RealLine),
    (lambda: dists.Gamma(concentration=1., rate=1.), Positive),
])
def test_exact_matches_simple(dist_factory, expected_type):
    """Test that common distributions map exactly to their expected constraint type."""
    dist = dist_factory()
    constraint = infer_distribution_constraint(dist)
    assert isinstance(constraint, expected_type)
    assert constraint.bounds[0].shape == dist.event_shape


def test_exact_match_interval_beta():
    """Test Beta distribution resolves to an Interval between 0 and 1."""
    dist = dists.Beta(alpha=1.0, beta=1.0)
    constraint = infer_distribution_constraint(dist)
    
    assert isinstance(constraint, Interval)
    np.testing.assert_array_equal(constraint.lower, jnp.zeros(dist.event_shape))
    np.testing.assert_array_equal(constraint.upper, jnp.ones(dist.event_shape))


def test_exact_match_interval_uniform():
    """Test Uniform distribution resolves to an Interval matching its high/low."""
    low = jnp.array([1.0, 2.0])
    high = jnp.array([3.0, 4.0])
    dist = dists.Uniform(low=low, high=high)
    constraint = infer_distribution_constraint(dist)
    
    assert isinstance(constraint, Interval)
    np.testing.assert_array_equal(constraint.lower, low)
    np.testing.assert_array_equal(constraint.upper, high)


def test_meta_distribution_joint():
    """Test that Joint distributions map correctly to a Leafwise constraint tree."""
    dist = dists.Joint({
        "a": dists.Normal(loc=0., scale=1.),
        "b": dists.Gamma(concentration=1., rate=1.)
    })
    
    constraint = infer_distribution_constraint(dist)
    
    assert isinstance(constraint, Leafwise)
    assert isinstance(constraint.tree["a"], RealLine)
    assert isinstance(constraint.tree["b"], Positive)


def test_meta_distribution_transformed():
    """Test that Transformed distributions wrap the base constraint."""
    base_dist = dists.Normal(loc=0., scale=1.)
    bijector = bij.Exp()
    dist = dists.Transformed(distribution=base_dist, bijector=bijector)
    
    constraint = infer_distribution_constraint(dist)
    assert isinstance(constraint, Transformed)


@pytest.mark.parametrize("lower, upper, expected_type", [
    (0.0, jnp.inf, Positive),                 # Lower bound is exactly 0
    (5.0, jnp.inf, GreaterThan),              # Lower bound > 0
    (-jnp.inf, 0.0, Negative),                # Upper bound is exactly 0
    (-jnp.inf, 5.0, LessThan),                # Upper bound > 0
    (-5.0, 5.0, Interval),                    # Both bounded
])
def test_icdf_fallbacks(lower, upper, expected_type):
    """Test that distributions lacking exact matches fallback correctly using ICDF."""
    dist = DummyICDFDist(lower=lower, upper=upper)
    constraint = infer_distribution_constraint(dist)
    assert isinstance(constraint, expected_type)


def test_icdf_fallback_bounds_values():
    """Specifically test that the bounds parsed from ICDF are correctly assigned."""
    dist_greater = DummyICDFDist(lower=5.0, upper=jnp.inf)
    constraint_greater = infer_distribution_constraint(dist_greater)
    assert isinstance(constraint_greater, GreaterThan)
    np.testing.assert_array_equal(constraint_greater.lower, jnp.array(5.0))

    dist_interval = DummyICDFDist(lower=-2.0, upper=2.0)
    constraint_interval = infer_distribution_constraint(dist_interval)
    assert isinstance(constraint_interval, Interval)
    np.testing.assert_array_equal(constraint_interval.lower, jnp.array(-2.0))
    np.testing.assert_array_equal(constraint_interval.upper, jnp.array(2.0))


def test_last_resort_fallback():
    """Test that exceptions during ICDF evaluation default safely to RealLine."""
    dist = BrokenDist(shape=(2, 2))
    constraint = infer_distribution_constraint(dist)
    
    assert isinstance(constraint, RealLine)
    assert constraint.shape == (2, 2)