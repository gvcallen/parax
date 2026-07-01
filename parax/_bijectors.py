"""
Temporary bijector implementations until upstream PR in gvcallen/distreqx.
"""
import functools

import jax
from jaxtyping import PyTree, Array
import jax.numpy as jnp
import jax.nn as jnn
import jax.scipy.special as jss
import jax.scipy.stats as jstats

from distreqx.distributions import AbstractDistribution
from distreqx.bijectors import (
    AbstractForwardInverseBijector, 
    AbstractInvLogDetJacBijector, 
    AbstractFwdLogDetJacBijector, 
    AbstractBijector
)

def _is_bijector(node: PyTree) -> bool:
    return isinstance(node, AbstractBijector)

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


class NormalCDF(
    AbstractForwardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """
    Transforms the unconstrained real line to the (0, 1) interval 
    using the Standard Normal Cumulative Distribution Function.
    """
    _is_constant_jacobian: bool = False
    _is_constant_log_det: bool = False

    def forward_and_log_det(self, x: PyTree) -> tuple[PyTree, PyTree]:
        """Computes y = CDF(x) and element-wise log|det J(f)(x)|."""
        y = jss.ndtr(x)
        # The derivative of the CDF is the PDF.
        log_det = jstats.norm.logpdf(x)
        return y, log_det

    def inverse_and_log_det(self, y: PyTree) -> tuple[PyTree, PyTree]:
        """Computes x = ICDF(y) and element-wise log|det J(f^{-1})(y)|."""
        eps = jnp.finfo(jnp.float32).eps
        y_safe = jnp.clip(y, eps, 1.0 - eps)
        
        x = jss.ndtri(y_safe)
        # The derivative of the ICDF is 1 / PDF(ICDF(y)).
        log_det = -jstats.norm.logpdf(x)
        return x, log_det

    def same_as(self, other: AbstractBijector) -> bool:
        return type(other) is NormalCDF
    

class Quantile(
    AbstractForwardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """
    Transforms the (0, 1) interval to a target distribution's physical 
    domain using its Inverse Cumulative Distribution Function.
    """
    distribution: AbstractDistribution

    _is_constant_jacobian: bool = False
    _is_constant_log_det: bool = False

    def forward_and_log_det(self, u: PyTree) -> tuple[PyTree, PyTree]:
        """Computes y = ICDF(u) and log|det J(f)(u)|."""
        eps = jnp.finfo(jnp.float32).eps
        u_safe = jnp.clip(u, eps, 1.0 - eps)
        
        y = self.distribution.icdf(u_safe)
        # Derivative of ICDF w.r.t u is 1 / PDF(y)
        log_det = -self.distribution.log_prob(y)
        return y, log_det

    def inverse_and_log_det(self, y: PyTree) -> tuple[PyTree, PyTree]:
        """Computes u = CDF(y) and log|det J(f^{-1})(y)|."""
        u = self.distribution.cdf(y)
        # Derivative of CDF w.r.t y is PDF(y)
        log_det = self.distribution.log_prob(y)
        return u, log_det

    def same_as(self, other: AbstractBijector) -> bool:
        return type(other) is Quantile and self.distribution == other.distribution



class Leafwise(AbstractFwdLogDetJacBijector, AbstractInvLogDetJacBijector, strict=True):
    """Applies a pytree of bijectors to a pytree of inputs.

    This behaves analogously to TensorFlow Probability's `JointMap`. It allows
    applying independent bijectors to a structured input (e.g., a tuple or dict
    of arrays) and aggregates the log-determinants across the structure.
    """

    bijectors: PyTree[AbstractBijector]
    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def __init__(self, bijectors: PyTree[AbstractBijector]):
        """Initializes a TreeMap bijector."""
        leaves = jax.tree_util.tree_leaves(bijectors, is_leaf=_is_bijector)
        if not leaves:
            raise ValueError("The pytree of bijectors cannot be empty.")

        self.bijectors = bijectors

        is_constant_jacobian = all(b.is_constant_jacobian for b in leaves)
        is_constant_log_det = all(b.is_constant_log_det for b in leaves)

        if is_constant_log_det is None:
            is_constant_log_det = is_constant_jacobian
        if is_constant_jacobian and not is_constant_log_det:
            raise ValueError(
                "The Jacobian is said to be constant, but its "
                "determinant is said not to be, which is impossible."
            )
        self._is_constant_jacobian = is_constant_jacobian
        self._is_constant_log_det = is_constant_log_det

    def forward(self, x: PyTree) -> PyTree:
        """Computes y = f(x)."""
        return jax.tree_util.tree_map(
            lambda b, v: b.forward(v), self.bijectors, x, is_leaf=_is_bijector
        )

    def inverse(self, y: PyTree) -> PyTree:
        """Computes x = f^{-1}(y)."""
        return jax.tree_util.tree_map(
            lambda b, v: b.inverse(v), self.bijectors, y, is_leaf=_is_bijector
        )

    def forward_and_log_det(self, x: PyTree) -> tuple[PyTree, PyTree]:
        """Computes y = f(x) and sum of log|det J(f)(x)|."""
        ys_and_log_dets = jax.tree_util.tree_map(
            lambda b, v: b.forward_and_log_det(v),
            self.bijectors,
            x,
            is_leaf=_is_bijector,
        )

        y = jax.tree_util.tree_map(
            lambda b, res: res[0], self.bijectors, ys_and_log_dets, is_leaf=_is_bijector
        )
        log_dets = jax.tree_util.tree_map(
            lambda b, res: res[1], self.bijectors, ys_and_log_dets, is_leaf=_is_bijector
        )

        log_det_leaves = jax.tree_util.tree_leaves(log_dets)
        total_log_det = functools.reduce(jnp.add, log_det_leaves)
        return y, total_log_det

    def inverse_and_log_det(self, y: PyTree) -> tuple[PyTree, PyTree]:
        """Computes x = f^{-1}(y) and sum of log|det J(f^{-1})(y)|."""
        xs_and_log_dets = jax.tree_util.tree_map(
            lambda b, v: b.inverse_and_log_det(v),
            self.bijectors,
            y,
            is_leaf=_is_bijector,
        )

        x = jax.tree_util.tree_map(
            lambda b, res: res[0], self.bijectors, xs_and_log_dets, is_leaf=_is_bijector
        )
        log_dets = jax.tree_util.tree_map(
            lambda b, res: res[1], self.bijectors, xs_and_log_dets, is_leaf=_is_bijector
        )

        log_det_leaves = jax.tree_util.tree_leaves(log_dets)
        total_log_det = functools.reduce(jnp.add, log_det_leaves)
        return x, total_log_det

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is Leafwise:
            if jax.tree_util.tree_structure(
                self.bijectors, is_leaf=_is_bijector
            ) != jax.tree_util.tree_structure(other.bijectors, is_leaf=_is_bijector):
                return False

            match_tree = jax.tree_util.tree_map(
                lambda b1, b2: b1.same_as(b2),
                self.bijectors,
                other.bijectors,
                is_leaf=_is_bijector,
            )
            return all(jax.tree_util.tree_leaves(match_tree))
        return False
