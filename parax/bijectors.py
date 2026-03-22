"""
Additional bijectors not present in distreqx.
"""
import jax.numpy as jnp
from jaxtyping import PyTree, Array

import jax.numpy as jnp
import jax.scipy as jsp
import distreqx.bijectors as bij
from typing import Tuple

class Identity(
    bij.AbstractFowardInverseBijector,
    bij.AbstractInvLogDetJacBijector,
    bij.AbstractFwdLogDetJacBijector,
    strict=True,
):
    """Identity bijector: y = x."""

    _is_constant_jacobian: bool = True
    _is_constant_log_det: bool = True

    def forward_and_log_det(self, x: PyTree):
        log_det = jnp.zeros_like(x)
        return x, log_det

    def inverse_and_log_det(self, y: PyTree):
        log_det = jnp.zeros_like(y)
        return y, log_det

    def same_as(self, other) -> bool:
        return isinstance(other, Identity)

class Inverse(bij.AbstractFwdLogDetJacBijector, bij.AbstractInvLogDetJacBijector, strict=True):
    """Inverted version of a given bijector.
    """

    bijector: bij.AbstractBijector
    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def __init__(self, bijectors: bij.AbstractBijector):
        """Initializes an Inverted bijector.

        **Arguments:**

        - `bijector`: The bijector to invert.
        """
        self.bijector = bijectors

        is_constant_jacobian = self.bijector.is_constant_jacobian
        is_constant_log_det = self.bijector.is_constant_log_det
        if is_constant_log_det is None:
            is_constant_log_det = is_constant_jacobian
        if is_constant_jacobian and not is_constant_log_det:
            raise ValueError(
                "The Jacobian is said to be constant, but its "
                "determinant is said not to be, which is impossible."
            )
        self._is_constant_jacobian = is_constant_jacobian
        self._is_constant_log_det = is_constant_log_det

    def forward(self, x: Array) -> Array:
        """Computes y = f(x)."""
        return self.bijector.inverse(x)

    def inverse(self, y: Array) -> Array:
        """Computes x = f^{-1}(y)."""
        return self.bijector.forward(y)

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.bijector.inverse_and_log_det(x)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.bijector.forward_and_log_det(y)

    def same_as(self, other: bij.AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is Inverse:
            return self.bijector.same_as(other.bijector)
        else:
            return self.bijector.same_as(other)


class Exponential(
    bij.AbstractFowardInverseBijector,
    bij.AbstractInvLogDetJacBijector,
    bij.AbstractFwdLogDetJacBijector,
    strict=True,
):
    """Exponential bijector: y = exp(x)."""
    
    _is_constant_jacobian: bool = False
    _is_constant_log_det: bool = False

    def forward_and_log_det(self, x: PyTree):
        y = jnp.exp(x)
        return y, x

    def inverse_and_log_det(self, y: PyTree):
        x = jnp.log(y)
        return x, -jnp.log(y)

    def same_as(self, other) -> bool:
        return isinstance(other, Exponential)

class NormalToUniform(bij.AbstractBijector):
    """
    A bijection mapping a Normal(loc, scale) distribution to a Uniform(0, 1) distribution 
    using the Probability Integral Transform.
    """
    loc: jnp.ndarray
    scale: jnp.ndarray

    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        self.loc = jnp.asarray(loc)
        self.scale = jnp.asarray(scale)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Maps x ~ Normal to y ~ Uniform(0, 1)."""
        z = (x - self.loc) / self.scale
        return jsp.special.ndtr(z)

    def inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        """Maps y ~ Uniform(0, 1) to x ~ Normal."""
        # Clip to prevent evaluating ndtri at exact 0 or 1, which causes NaNs/Infs
        eps = jnp.finfo(y.dtype).eps
        y_clipped = jnp.clip(y, a_min=eps, a_max=1.0 - eps)
        z = jsp.special.ndtri(y_clipped)
        return z * self.scale + self.loc

    def forward_and_log_det(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns the forward transformation and its log-determinant Jacobian."""
        y = self.forward(x)
        # The derivative of the CDF is exactly the PDF. 
        # log(det(Jacobian)) = log(PDF(x))
        log_det = jsp.stats.norm.logpdf(x, loc=self.loc, scale=self.scale)
        return y, log_det

    def inverse_and_log_det(self, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns the inverse transformation and its log-determinant Jacobian."""
        x = self.inverse(y)
        # By the inverse function theorem, the log det of the inverse is the 
        # negative log det of the forward function evaluated at x.
        log_det = -jsp.stats.norm.logpdf(x, loc=self.loc, scale=self.scale)
        return x, log_det