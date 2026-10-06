import jax.numpy as jnp
import pytest

from parax._bijectors import Softplus
from parax.bijectors import Chain, Shift


def test_fallback_softplus_reports_nonconstant_jacobian_and_log_det():
    bijector = Softplus()

    assert not bijector.is_constant_jacobian
    assert not bijector.is_constant_log_det


def test_fallback_softplus_log_det_varies_with_input():
    bijector = Softplus()
    x = jnp.array([-2.0, 2.0])

    y, log_det = bijector.forward_and_log_det(x)

    assert jnp.allclose(log_det, jnp.array([-2.126928, -0.126928]))
    assert log_det[0] != log_det[1]

    recovered_x, inverse_log_det = bijector.inverse_and_log_det(y)

    assert jnp.allclose(recovered_x, x)
    assert jnp.allclose(inverse_log_det, jnp.array([2.126928, 0.126928]))


@pytest.mark.parametrize("softplus_first", [True, False])
def test_chain_with_fallback_softplus_remains_nonconstant(softplus_first):
    bijectors = [Softplus(), Shift(jnp.array(1.0))]
    if not softplus_first:
        bijectors.reverse()
    bijector = Chain(bijectors)

    assert not bijector.is_constant_jacobian
    assert not bijector.is_constant_log_det

    _, log_det = bijector.forward_and_log_det(jnp.array([-2.0, 2.0]))

    assert log_det[0] != log_det[1]
