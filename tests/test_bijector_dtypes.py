import jax
import jax.numpy as jnp
from jax import enable_x64

from parax._bijectors import NormalCDF


def test_normal_cdf_inverse_preserves_float64():
    """The safety clip must use the input's dtype, not a hardcoded float32."""
    with enable_x64():
        y = jnp.array([1e-10, 0.5, 1.0 - 1e-10], dtype=jnp.float64)
        x, _ = NormalCDF().inverse_and_log_det(y)

        assert x.dtype == jnp.float64
        # With a float32 eps (~1.19e-7) the tails would clip to +-5.2 instead.
        assert abs(float(x[0]) + 6.361) < 1e-2
        assert abs(float(x[2]) - 6.361) < 1e-2


def test_normal_cdf_inverse_still_safe_in_float32():
    y = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)
    x, log_det = NormalCDF().inverse_and_log_det(y)

    assert x.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(x)))
    assert bool(jnp.all(jnp.isfinite(log_det)))
