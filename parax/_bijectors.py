"""
Bijectors that are parax's own, plus fallbacks for distreqx names that not every
distreqx version provides. Import them from `parax.bijectors`, which prefers the
installed distreqx's implementation where there is one.
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

    _is_constant_jacobian: bool = False
    _is_constant_log_det: bool = False

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

    The inverse does not clip: 0 and 1 map to -inf and inf.
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
        x = jss.ndtri(y)
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
        # Keeps the ICDF off the support's edge. At float64 this clips the raw
        # space at about +-8.1 sigma on both sides: past that, every raw value gives
        # the same constrained value, and a constrained value maps to an infinite raw.
        eps = jnp.finfo(jnp.result_type(u, float)).eps
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


def _log_sub_exp(a: Array, b: Array) -> Array:
    """`log(exp(a) - exp(b))` for `a >= b`, without forming either exponential."""
    return a + jnp.log(-jnp.expm1(b - a))


def _lower_ndtri(log_p: Array, steps: int = 3) -> Array:
    """
    The standard normal quantile of `exp(log_p)`, for `log_p <= log(1/2)`.

    `ndtri` gives the start where the probability is a normal float, and an
    asymptotic expansion below that. Newton steps on `log_ndtr` then refine it,
    which also gives the result an exact gradient. A `log_p` of -inf gives -inf.
    """
    dtype = jnp.result_type(log_p, float)
    log_p = jnp.asarray(log_p, dtype)
    log_half = jnp.log(jnp.asarray(0.5, dtype))
    log_tiny = jnp.log(jnp.asarray(jnp.finfo(dtype).tiny, dtype))

    finite = jnp.isfinite(log_p)
    target = jnp.where(finite, log_p, log_half)
    deep = target < log_tiny
    # Each start sees a stand-in where it is not chosen, so neither makes a NaN.
    x_ndtri = jss.ndtri(jnp.exp(jnp.where(deep, log_half, target)))
    # log Phi(x) ~ -x^2/2 - log(-x) - log(2 pi)/2 as x -> -inf.
    t = -2.0 * jnp.where(deep, target, log_tiny)
    x_deep = -jnp.sqrt(t - jnp.log(t) - jnp.log(2.0 * jnp.pi))
    x = jnp.where(deep, x_deep, x_ndtri)

    for _ in range(steps):
        log_cdf = jss.log_ndtr(x)
        x = x - (log_cdf - target) * jnp.exp(log_cdf - jstats.norm.logpdf(x))
    return jnp.where(finite, x, -jnp.inf)


def _log_ndtri(log_p: Array) -> Array:
    """
    The standard normal quantile of `exp(log_p)`, accurate in both tails: above
    one half it is taken from the complement. A `log_p` of 0 gives inf.
    """
    upper = log_p > jnp.log(0.5)
    log_q = jnp.log(-jnp.expm1(jnp.where(upper, log_p, jnp.log(0.5))))
    x = _lower_ndtri(jnp.where(upper, log_q, log_p))
    return jnp.where(upper, -x, x)


def _log_ndtr(x: Array) -> Array:
    """`log_ndtr`, with a clean gradient at an infinite `x`."""
    finite = jnp.isfinite(x)
    value = jss.log_ndtr(jnp.where(finite, x, 0.0))
    return jnp.where(finite, value, jnp.where(x > 0, 0.0, -jnp.inf))


def _from_tails(lower: Array, log_p: Array, log_q: Array, ndtri) -> Array:
    """
    `ndtri(log_p)` where `lower`, and `-ndtri(log_q)` elsewhere: a standard
    normal value from whichever tail probability is its own. Each side sees a
    stand-in where it is not chosen, so that neither makes a NaN.
    """
    stand_in = jnp.log(0.25)
    return jnp.where(
        lower,
        ndtri(jnp.where(lower, log_p, stand_in)),
        -ndtri(jnp.where(lower, stand_in, log_q)),
    )


def _keep_nan(value: Array, out: Array, log_det: Array) -> tuple[Array, Array]:
    """
    `out` and `log_det`, but NaN wherever `value` is. The tail helpers send a
    NaN to an infinity, as they do any non-finite probability, so a NaN input
    is passed through here instead.
    """
    nan = jnp.isnan(value)
    return jnp.where(nan, value, out), jnp.where(nan, value, log_det)


class TruncatedNormalWhitening(
    AbstractForwardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """
    Maps a standard normal raw value onto a truncated normal's support, so that the
    raw value stays standard normal under the distribution.

    It is the copula map `icdf(ndtr(z))`, computed so that neither tail saturates.
    Each half of the support comes from its own small tail probability, in log
    space: with standardised bounds `a`, `b` and `Z = Phi(b) - Phi(a)`, a value
    below the median of the untruncated normal is `Phi^-1(Phi(a) + Z Phi(z))`, and
    one above it is `-Phi^-1(Phi(-b) + Z Phi(-z))`. With both bounds infinite it is
    exactly `loc + scale * z`.

    Attributes:
        loc: The mean of the untruncated normal.
        scale: The standard deviation of the untruncated normal.
        low: The lower bound, possibly `-inf`.
        high: The upper bound, possibly `inf`.
    """
    loc: Array
    scale: Array
    low: Array
    high: Array

    _is_constant_jacobian: bool = False
    _is_constant_log_det: bool = False

    def _standardize(self, value: Array) -> Array:
        """`(value - loc) / scale`, keeping an infinite value, and its gradient, clean."""
        finite = jnp.isfinite(value)
        standard = (jnp.where(finite, value, 0.0) - self.loc) / self.scale
        return jnp.where(finite, standard, value)

    def _standard_bounds(self) -> tuple[Array, Array, Array]:
        """The standardised bounds `a`, `b` and `log Z`."""
        a = self._standardize(self.low)
        b = self._standardize(self.high)
        # Take Z from whichever tail it is small in.
        log_z = jnp.where(
            a + b > 0,
            _log_sub_exp(_log_ndtr(-a), _log_ndtr(-b)),
            _log_sub_exp(_log_ndtr(b), _log_ndtr(a)),
        )
        return a, b, log_z

    def _log_det(self, z: Array, x: Array, log_z: Array) -> Array:
        """log|dy/dz| at raw `z` and standardised value `x`."""
        return (
            log_z + jstats.norm.logpdf(z) - jstats.norm.logpdf(x) + jnp.log(self.scale)
        )

    def forward_and_log_det(self, z: Array) -> tuple[Array, Array]:
        """Computes the constrained value at raw `z`, and log|det J(f)(z)|."""
        a, b, log_z = self._standard_bounds()
        log_p = jnp.logaddexp(_log_ndtr(a), log_z + jss.log_ndtr(z))
        log_q = jnp.logaddexp(_log_ndtr(-b), log_z + jss.log_ndtr(-z))
        # Both tail probabilities are at most one half here, as they sum to one.
        x = _from_tails(log_p < jnp.log(0.5), log_p, log_q, _lower_ndtri)
        x = jnp.where(jnp.isinf(a) & jnp.isinf(b), z, x)

        y = self.loc + self.scale * x
        # Rounding must not carry a value past a bound. Unlike `jnp.clip`, this
        # keeps the full gradient at a value that lands exactly on one.
        y = jnp.where(y < self.low, self.low, jnp.where(y > self.high, self.high, y))
        return _keep_nan(z, y, self._log_det(z, x, log_z))

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes the raw value at constrained `y`, and log|det J(f^{-1})(y)|."""
        a, b, log_z = self._standard_bounds()
        x = self._standardize(y)
        lower = x < 0
        # Each branch sees an interior stand-in where it is not chosen.
        inside = jnp.where(
            jnp.isfinite(a) & jnp.isfinite(b), (a + b) / 2,
            jnp.where(jnp.isfinite(a), a + 1.0, jnp.where(jnp.isfinite(b), b - 1.0, 0.0)),
        )
        x_lower = jnp.where(lower, x, inside)
        x_upper = jnp.where(lower, inside, x)
        log_p = _log_sub_exp(jss.log_ndtr(x_lower), _log_ndtr(a)) - log_z
        log_q = _log_sub_exp(jss.log_ndtr(-x_upper), _log_ndtr(-b)) - log_z
        # Here a tail probability may pass one half, as the raw value is not
        # split at its median, so take the quantile that handles both halves.
        z = _from_tails(lower, log_p, log_q, _log_ndtri)
        z = jnp.where(jnp.isinf(a) & jnp.isinf(b), x, z)
        return _keep_nan(y, z, -self._log_det(z, x, log_z))

    def same_as(self, other: AbstractBijector) -> bool:
        return other is self



class Elementwise(
    AbstractForwardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """
    Applies one of several elementwise bijectors to each element, chosen by a mask
    per bijector. The masks cover every element exactly once.

    `Leafwise` gives each leaf of a pytree its own bijector; this gives each element
    of a single array its own, so one array can mix, say, half-line and interval
    elements. Its log-determinant stays elementwise.

    Every bijector is evaluated everywhere. Where its mask is off, its inverse is fed
    a value from its own image instead of the real one, so that no NaN from an
    unused branch reaches a gradient.

    Attributes:
        masks: One boolean array per bijector, `True` where it applies.
        bijectors: The elementwise bijectors, one per mask.
        safe_values: One value per bijector inside its image, used in place of
            the input to its inverse where its mask is off.
    """
    masks: tuple[Array, ...]
    bijectors: tuple[AbstractBijector, ...]
    safe_values: tuple[Array, ...]

    _is_constant_jacobian: bool = False
    _is_constant_log_det: bool = False

    def _select(self, pairs):
        (out, log_det), *rest = pairs
        for mask, (branch_out, branch_log_det) in zip(self.masks[1:], rest):
            out = jnp.where(mask, branch_out, out)
            log_det = jnp.where(mask, branch_log_det, log_det)
        return out, log_det

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        return self._select([b.forward_and_log_det(x) for b in self.bijectors])

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        return self._select([
            b.inverse_and_log_det(jnp.where(mask, y, safe))
            for mask, b, safe in zip(self.masks, self.bijectors, self.safe_values)
        ])

    def same_as(self, other: AbstractBijector) -> bool:
        return other is self



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
