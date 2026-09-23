import jax.numpy as jnp
import numpy as np

import parax.distributions as dists


def test_independent_takes_reinterpreted_batch_ndims_whatever_distreqx_is_installed():
    """Upstream distreqx's Independent takes no `reinterpreted_batch_ndims`; parax
    always provides one that does, with 0 behaving as upstream's."""
    uniform = dists.Independent(dists.Uniform(jnp.zeros(2), jnp.full(2, 2.0)), 1)
    assert uniform.event_shape == (2,)
    assert np.allclose(uniform.log_prob(jnp.array([0.5, 1.5])), -2 * np.log(2.0))
    normal = dists.Independent(dists.Normal(jnp.zeros(2), jnp.ones(2)))
    assert np.allclose(normal.log_prob(jnp.zeros(2)), -np.log(2 * np.pi))
