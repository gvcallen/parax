"""Construction behavior across eager execution and JAX transformations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import parax.bijectors as bij
import parax.distributions as dists
from parax.constraints import Interval, Positive
from parax.variables import Constrained, Random
from parax.wrappers import as_unwrapped


@pytest.mark.parametrize("distribution", [
    dists.TruncatedNormal(1.0, 0.2, 0.0, 2.0),
    dists.Uniform(0.0, 10.0),
    dists.Beta(1.0, 2.0),
    dists.Gamma(2.0, 1.0),
    dists.LogNormal(0.0, 1.0),
    dists.Normal(1.0, 0.2),
    dists.Logistic(1.0, 0.2),
    dists.MultivariateNormalDiag(jnp.array([0.0, 1.0]), jnp.ones(2)),
    dists.MultivariateNormalTri(jnp.zeros(2), jnp.eye(2)),
    dists.MultivariateNormalFullCovariance(jnp.zeros(2), jnp.eye(2)),
    dists.Transformed(dists.Normal(0.0, 1.0), bij.Exp()),
])
def test_random_compiled_matches_eager_support(distribution):
    value = 1.0 if isinstance(distribution, dists.Transformed) else None
    with jax.disable_jit():
        eager = Random(distribution, value=value)
    compiled = Random(distribution, value=value)

    np.testing.assert_allclose(compiled.raw_value, eager.raw_value, equal_nan=True)
    for name in ("bounds", "base_bounds", "closed"):
        actual = getattr(as_unwrapped(compiled.constraint), name)
        expected = getattr(as_unwrapped(eager.constraint), name)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, equal_nan=True, atol=1e-6)


def test_invalid_value_raises_eager_and_under_jit():
    with pytest.raises(Exception, match="Constraint violated"):
        Constrained(Interval(0.0, 10.0), value=11.0)
    with pytest.raises(Exception, match="Constraint violated"):
        jax.jit(lambda x: Constrained(Interval(0.0, 10.0), value=x).raw_value)(11.0)
    with pytest.raises(Exception, match="Constraint violated"):
        Random(dists.Uniform(0.0, 10.0), value=11.0)
    with pytest.raises(Exception, match="Constraint violated"):
        jax.jit(lambda x: Random(dists.Uniform(0.0, 10.0), value=x).raw_value)(11.0)


def test_construction_under_jax_transforms_and_wrap():
    make = lambda x: Constrained(Positive(), value=x).raw_value
    expected = np.asarray(make(2.0))
    np.testing.assert_allclose(jax.jit(make)(2.0), expected)
    np.testing.assert_allclose(jax.vmap(make)(jnp.array([2.0, 2.0])), [expected, expected])
    np.testing.assert_allclose(jax.grad(make)(2.0), jax.grad(jax.jit(make))(2.0))

    make_random = lambda x: Random(dists.Normal(x, 0.2)).raw_value
    np.testing.assert_allclose(jax.jit(make_random)(1.0), make_random(1.0))
    np.testing.assert_allclose(jax.vmap(make_random)(jnp.array([1.0, 2.0])), [0.0, 0.0])
    with jax.disable_jit():
        eager_grad = jax.grad(lambda x: Random(dists.Normal(x, 0.2)).value)(1.0)
    np.testing.assert_allclose(jax.grad(lambda x: Random(dists.Normal(x, 0.2)).value)(1.0), eager_grad)

    for variable in (Constrained(Positive(), value=2.0), Random(dists.Normal(1.0, 0.2))):
        np.testing.assert_allclose(variable.wrap(variable.value).raw_value, variable.raw_value)


def test_random_constrain_still_truncates():
    variable = Random(dists.Normal(1.0, 0.2))
    constrained = variable.constrain(Interval(0.0, 2.0))
    np.testing.assert_allclose(constrained.value, variable.value)
    np.testing.assert_allclose(as_unwrapped(constrained.distribution).low, 0.0)
    np.testing.assert_allclose(as_unwrapped(constrained.distribution).high, 2.0)
