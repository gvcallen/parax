import jax
import jax.numpy as jnp
import numpy as np
import pytest

import parax.distributions as dists


def test_independent_takes_reinterpreted_batch_ndims_whatever_distreqx_is_installed():
    """Upstream distreqx's Independent takes no `reinterpreted_batch_ndims`; parax
    always provides one that does, with 0 behaving as upstream's."""
    uniform = dists.Independent(dists.Uniform(jnp.zeros(2), jnp.full(2, 2.0)), 1)
    assert uniform.event_shape == (2,)
    assert np.allclose(uniform.log_prob(jnp.array([0.5, 1.5])), -2 * np.log(2.0))
    normal = dists.Independent(dists.Normal(jnp.zeros(2), jnp.ones(2)))
    assert np.allclose(normal.log_prob(jnp.zeros(2)), -np.log(2 * np.pi))


def test_truncated_normal_is_owned_by_parax():
    assert dists.TruncatedNormal.__module__ == "parax._distributions"


@pytest.mark.parametrize("low, high", [(1.0, jnp.inf), (-jnp.inf, 3.2)])
def test_distant_one_sided_truncation_has_normal_variance(low, high):
    with jax.enable_x64():
        loc = 2.1
        dist = dists.TruncatedNormal(loc, 0.105, low, high)
        np.testing.assert_allclose(dist.variance(), 0.105**2, rtol=1e-12)
        np.testing.assert_allclose(dist.stddev(), 0.105, rtol=1e-12)


@pytest.mark.parametrize("low, high", [(1.0, jnp.inf), (-jnp.inf, 3.0), (-jnp.inf, jnp.inf)])
def test_infinite_bounds_have_finite_moments_and_gradients(low, high):
    with jax.enable_x64():
        dist = dists.TruncatedNormal(2.1, 0.105, low, high)
        for method in ("mean", "variance", "entropy"):
            assert np.isfinite(getattr(dist, method)())
        for method in ("mean", "variance"):
            for index in (0, 1):
                gradient = jax.grad(
                    lambda loc, scale: getattr(
                        dists.TruncatedNormal(loc, scale, low, high), method
                    )(),
                    argnums=index,
                )(2.1, 0.105)
                assert np.isfinite(gradient)


def test_unbounded_truncated_normal_matches_normal():
    with jax.enable_x64():
        truncated = dists.TruncatedNormal(2.1, 0.105, -jnp.inf, jnp.inf)
        normal = dists.Normal(2.1, 0.105)
        for method in ("mean", "variance", "entropy"):
            np.testing.assert_allclose(
                getattr(truncated, method)(), getattr(normal, method)(), rtol=1e-12
            )


def test_two_sided_truncated_normal_moments_match_monte_carlo():
    with jax.enable_x64():
        dist = dists.TruncatedNormal(0.3, 1.0, -0.5, 1.2)
        # 100,000 draws give mean and variance estimates within 0.01 here.
        samples = jax.vmap(dist.sample)(jax.random.split(jax.random.PRNGKey(0), 100_000))
        np.testing.assert_allclose(dist.mean(), jnp.mean(samples), atol=0.01)
        np.testing.assert_allclose(dist.variance(), jnp.var(samples), atol=0.01)
