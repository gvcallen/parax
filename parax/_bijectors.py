"""
Temporary bijector implementations until upstream PR in gvcallen/distreqx.
"""
import jax
from jaxtyping import PyTree, Array
import jax.numpy as jnp
import jax.nn as jnn

import equinox as eqx

from distreqx.bijectors import AbstractForwardInverseBijector, AbstractInvLogDetJacBijector, AbstractFwdLogDetJacBijector, AbstractBijector

class Identity(
    AbstractForwardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """Identity bijector: y = x."""

    _is_constant_jacobian: bool = True
    _is_constant_log_det: bool = True

    def forward_and_log_det(self, x: PyTree) -> tuple[PyTree, PyTree]:
        """Computes y = x and log|det J(f)(x)| = 0."""
        log_det = jax.tree_util.tree_map(jnp.zeros_like, x)
        return x, log_det

    def inverse_and_log_det(self, y: PyTree) -> tuple[PyTree, PyTree]:
        """Computes x = y and log|det J(f^{-1})(y)| = 0."""
        log_det = jax.tree_util.tree_map(jnp.zeros_like, y)
        return y, log_det

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return type(other) is Identity
    

class Inverse(AbstractFwdLogDetJacBijector, AbstractInvLogDetJacBijector, strict=True):
    """Inverted version of a given bijector."""

    bijector: AbstractBijector

    _is_constant_jacobian: bool = eqx.field(init=False)
    _is_constant_log_det: bool = eqx.field(init=False)

    def __post_init__(self):
        is_constant_jacobian = self.bijector.is_constant_jacobian
        is_constant_log_det = self.bijector.is_constant_log_det

        if is_constant_jacobian and not is_constant_log_det:
            raise ValueError(
                "The Jacobian is said to be constant, but its "
                "determinant is said not to be, which is impossible."
            )

        object.__setattr__(self, "_is_constant_jacobian", is_constant_jacobian)
        object.__setattr__(self, "_is_constant_log_det", is_constant_log_det)

    def forward(self, x: PyTree) -> PyTree:
        """Computes y = f(x)."""
        return self.bijector.inverse(x)

    def inverse(self, y: PyTree) -> PyTree:
        """Computes x = f^{-1}(y)."""
        return self.bijector.forward(y)

    def forward_and_log_det(self, x: PyTree) -> tuple[PyTree, PyTree]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.bijector.inverse_and_log_det(x)

    def inverse_and_log_det(self, y: PyTree) -> tuple[PyTree, PyTree]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.bijector.forward_and_log_det(y)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is Inverse:
            return self.bijector.same_as(other.bijector)
        return False



class Softplus(
    AbstractForwardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """
    Transforms the real line to the positive domain using
    softplus y = log(1 + exp(x)).
    """

    _is_constant_jacobian: bool = True
    _is_constant_log_det: bool = True

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = softplus(x) and log|det J(f)(x)|."""
        y = jnn.softplus(x)
        logdet = -jnn.softplus(-x)
        return y, logdet

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = softplus^{-1}(y) and log|det J(f^{-1})(y)|."""
        x = jnp.log(-jnp.expm1(-y)) + y
        logdet = jnn.softplus(-x)
        return x, logdet

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return type(other) is Softplus
