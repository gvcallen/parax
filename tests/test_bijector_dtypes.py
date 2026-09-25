import jax
import jax.numpy as jnp
from jax import enable_x64

from parax._bijectors import NormalCDF, Quantile
import parax.distributions as dists


def test_normal_cdf_inverse_preserves_float64():
    with enable_x64():
        y = jnp.array([1e-10, 0.5, 1.0 - 1e-10], dtype=jnp.float64)
        x, _ = NormalCDF().inverse_and_log_det(y)

        assert x.dtype == jnp.float64
        assert abs(float(x[0]) + 6.361) < 1e-2
        assert abs(float(x[2]) - 6.361) < 1e-2


def test_normal_cdf_inverse_does_not_clip():
    """The edges of the unit interval map to -inf and inf, not to a clipped eps."""
    y = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)
    x = NormalCDF().inverse(y)

    assert x.dtype == jnp.float32
    assert jnp.isneginf(x[0])
    assert x[1] == 0.0
    assert jnp.isposinf(x[2])


def test_quantile_forward_still_guards_the_tails():
    """Raw values far beyond where `ndtr` underflows stay inside the support."""
    quantile = Quantile(dists.Uniform(low=jnp.array(0.0), high=jnp.array(10.0)))
    y = quantile.forward(NormalCDF().forward(jnp.array([-jnp.inf, jnp.inf])))

    assert bool(jnp.all(jnp.isfinite(y)))
    assert y[0] > 0.0
    assert y[1] < 10.0
