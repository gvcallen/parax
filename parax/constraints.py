"""
Constraints with bijector mapping and associated interfaces.

This module provides the tools to map unconstrained optimizer spaces 
(spanning the real line) into bounded physical spaces, as well as
to mark a PyTree as constrainable.
"""

from functools import singledispatch
from abc import abstractmethod
from typing import TypeVar, Union, Any, TypeGuard, Self

import jax
from jax.flatten_util import ravel_pytree
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

import parax.distributions as dists
import distreqx.distributions as distreqx_dists
from parax.bijectors import (
    AbstractBijector,
    Sigmoid,
    Chain,
    Shift,
    ScalarAffine,
    TriangularLinear,
    Identity,
    Leafwise as LeafwiseBijector,
    Elementwise,
    Softplus,
)

from parax.bounds import AbstractBounded
from parax._bijectors import NormalCDF, Quantile

T = TypeVar("Value")

class AbstractConstraint(eqx.Module):
    """
    The base class for all physical constraints in Parax.
    
    Constraints are a higher-level concept that provide bounds and bijectors over constrained domains.
    This is useful for use with unconstrained solvers (which require a bijector from the
    unconstrained real line to the constrained domain) and bounded solvers (which accept
    lower and upper bounds directly).

    Attributes:
        bounds: A tuple containing the physical lower and upper bounds of the constrained space.
        closed: A tuple saying whether each of `bounds` is included in the constrained space.
            Each side matches the structure of its bound, with a bool (or bool array) for each
            leaf: `True` for an edge a value may sit on exactly, `False` for one it may only
            approach. An infinite bound is closed unless asked otherwise, so ±∞ is inside.
        bijector: A `distreqx.bijectors.AbstractBijector` mapping from the unconstrained real line to the physical space.
            It maps onto the open interior, so a value exactly on a closed bound has an
            infinite raw value.
        base_bounds: A tuple containing the foundational, un-skewed orthogonal bounds, which a
            bounded optimizer works in directly. Where the physical space has two finite bounds
            this is the unit box, so that a step of a given size carries the same meaning along
            every axis whatever the physical units. For transformed constraints, this isolates
            the safe topological box before any dense correlations or skews are applied. It falls
            back to `bounds` where there is no finite extent to normalise against, such as an
            unbounded or half-bounded domain.
        base_bijector: A `distreqx.bijectors.AbstractBijector` mapping from the orthogonal `base_bounds`
            space into the physical `bounds` space, edge onto edge. Defaults to `Identity` unless
            geometric skews are present.
    """
    bounds: eqx.AbstractVar[tuple[PyTree, PyTree]]
    closed: eqx.AbstractVar[tuple[PyTree, PyTree]]
    bijector: eqx.AbstractVar[AbstractBijector]

    base_bounds: eqx.AbstractVar[tuple[PyTree, PyTree]]
    base_bijector: eqx.AbstractVar[AbstractBijector]
    
    def clip(self, value: PyTree) -> PyTree:
        """
        Clip a value to lie within this constraint.
        """
        return jax.tree.map(jnp.clip, value, self.bounds[0], self.bounds[1])
    
    def is_outside(self, value: PyTree) -> PyTree:
        """
        Returns if another value is outside the constraint.

        A value on a closed bound is inside; a value on an open bound is outside.
        NaN is outside every constraint.
        """
        lower, upper = self.bounds
        lower_closed, upper_closed = self.closed

        def _is_outside(x, l, u, lc, uc):
            above_lower = jnp.where(lc, x >= l, x > l)
            below_upper = jnp.where(uc, x <= u, x < u)
            return jnp.logical_not(jnp.logical_and(above_lower, below_upper))

        return jax.tree.map(_is_outside, value, lower, upper, lower_closed, upper_closed)
    
    def midpoint(self) -> PyTree:
        """
        Returns the midpoint of the constraint.
        
        Note that non-finite constraints may return infinity.
        """
        return jax.tree.map(lambda a, b: (a + b) / 2.0, self.bounds[0], self.bounds[1])
    

def is_constraint(x: Any) -> TypeGuard[AbstractConstraint]:
    """
    Returns True if `x` is an instance of `parax.AbstractConstraint`.
    """
    return isinstance(x, AbstractConstraint)    
    

def _as_pair(closed: bool | tuple[Any, Any]) -> tuple[Any, Any]:
    """Expands a `closed` argument given as one bool into a `(lower, upper)` pair."""
    if isinstance(closed, tuple):
        return closed
    return (closed, closed)


class AbstractUncorrelatedConstraint(AbstractConstraint):
    """
    A mixin for base constraints (like Interval, RealLine, Positive)
    carrying no dense correlation between axes.

    The default is for the base space to coincide with the physical one, which suits an
    unbounded or half-bounded domain where there is no finite extent to normalise
    against. A subclass with a bounded domain should override `base_bounds` and
    `base_bijector` to expose a whitened base instead, since a bounded optimizer works
    in that space directly and would otherwise inherit the physical units.
    """
    bounds: eqx.AbstractVar[tuple[PyTree, PyTree]]
    bijector: eqx.AbstractVar[AbstractBijector]

    @property
    def base_bounds(self) -> tuple[PyTree, PyTree]:
        return self.bounds

    @property
    def base_bijector(self) -> AbstractBijector:
        return Identity()        


class RealLine(AbstractUncorrelatedConstraint):
    """
    Represents a value that can span the entire real number line.
    
    Effectively a structural no-op constraint using an Identity bijector, 
    useful for maintaining consistent types in mixed parameter sets. Both infinite
    ends are closed, so ±∞ is inside; for an open end, use `Interval(-inf, inf, closed=...)`.

    Attributes:
        shape: The expected shape of the unconstrained parameter.
    """
    shape: Any = eqx.field(static=True)

    def __init__(self, shape: Any = ()):
        """
        Args:
            shape: The expected shape of the unconstrained parameter.
        """
        self.shape = shape
    
    @property
    def bounds(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        return (
            jnp.full(self.shape, -jnp.inf, dtype=float),
            jnp.full(self.shape, jnp.inf, dtype=float),
        )

    @property
    def closed(self) -> tuple[bool, bool]:
        return (True, True)

    @property
    def bijector(self) -> AbstractBijector:
        return Identity()


class GreaterThan(AbstractUncorrelatedConstraint):
    """
    Represents a value greater than, or equal to, a lower bound.
    
    Attributes:
        lower: The lower bound array or scalar.
        closed: Whether each bound is included, as a `(lower, upper)` pair. The upper
            bound is +∞.
    """
    lower: jnp.ndarray
    closed: tuple[bool, bool] = eqx.field(static=True)
    
    def __init__(
        self,
        lower: Union[float, Array],
        closed: bool | tuple[bool, bool] = True,
    ):
        """
        Args:
            lower: The lower bound.
            closed: Whether the bounds themselves are included: one bool for both
                `lower` and +∞, or a `(lower, upper)` pair.
        """
        self.lower = jnp.asarray(lower, dtype=float)
        self.closed = _as_pair(closed)
        
    @property
    def bounds(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        return (self.lower, jnp.full_like(self.lower, jnp.inf))

    @property
    def bijector(self) -> AbstractBijector:
        return Chain([Shift(self.lower), Softplus()])


class LessThan(AbstractUncorrelatedConstraint):
    """
    Represents a value less than, or equal to, an upper bound.
    
    Attributes:
        upper: The upper bound array or scalar.
        closed: Whether each bound is included, as a `(lower, upper)` pair. The lower
            bound is -∞.
    """
    upper: jnp.ndarray
    closed: tuple[bool, bool] = eqx.field(static=True)

    def __init__(
        self,
        upper: Union[float, Array],
        closed: bool | tuple[bool, bool] = True,
    ):
        """
        Args:
            upper: The upper bound.
            closed: Whether the bounds themselves are included: one bool for both
                -∞ and `upper`, or a `(lower, upper)` pair.
        """
        self.upper = jnp.asarray(upper, dtype=float)
        self.closed = _as_pair(closed)

    @property
    def bounds(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        return (jnp.full_like(self.upper, -jnp.inf), self.upper)

    @property
    def bijector(self) -> AbstractBijector:
        # Corner Case Note: To implement a LessThan constraint using Softplus 
        # (which inherently bounds > 0), we apply a double affine flip:
        # invert -> softplus -> invert -> shift.
        return Chain([
            Shift(self.upper),
            ScalarAffine(shift=jnp.array(0.0), scale=jnp.array(-1.0)),
            Softplus(),
            ScalarAffine(shift=jnp.array(0.0), scale=jnp.array(-1.0)),
        ])


def _base_space(lower: Array, upper: Array) -> tuple[Array, Array, AbstractBijector]:
    """
    The base space over one array of bounds, as `(base_lower, base_upper, base_bijector)`.

    Elementwise: the unit box mapped affinely onto the bounds where both are finite,
    and the physical space itself, unchanged, where either is infinite.
    """
    finite = jnp.isfinite(lower) & jnp.isfinite(upper)
    base_lower = jnp.where(finite, 0.0, lower)
    base_upper = jnp.where(finite, 1.0, upper)
    base_bijector = Chain([
        Shift(jnp.where(finite, lower, 0.0)),
        ScalarAffine(shift=jnp.array(0.0), scale=jnp.where(finite, upper - lower, 1.0)),
    ])
    return base_lower, base_upper, base_bijector


class Interval(AbstractUncorrelatedConstraint):
    """
    Represents a value bounded between a lower and upper value.

    Either end may be infinite, per element. Where it is, the element behaves like the
    matching half-line or real line (bijector and base space), keeping its own `closed`.
    
    Attributes:
        lower: The lower bound.
        upper: The upper bound.
        closed: Whether `lower` and `upper` themselves are included, as a `(lower, upper)` pair.
    """
    lower: jnp.ndarray
    upper: jnp.ndarray
    closed: tuple[bool, bool] = eqx.field(static=True)

    def __init__(
        self,
        lower: Union[float, Array],
        upper: Union[float, Array],
        closed: bool | tuple[bool, bool] = True,
    ):
        """
        Args:
            lower: The lower bound.
            upper: The upper bound.
            closed: Whether the bounds themselves are included: one bool for both,
                or a `(lower, upper)` pair.
        """
        self.lower = jnp.asarray(lower, dtype=float)
        self.upper = jnp.asarray(upper, dtype=float)
        self.closed = _as_pair(closed)

    @property
    def bounds(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        return (self.lower, self.upper)

    @property
    def base_bounds(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        """
        The unit box where both ends are finite, and the physical space elsewhere.

        Overrides the `Identity` default from `AbstractUncorrelatedConstraint`, which
        would hand a bounded optimizer the physical box. Normalising to `[0, 1]`
        is generally numerically better during optimization.
        """
        base_lower, base_upper, _ = _base_space(self.lower, self.upper)
        return (base_lower, base_upper)

    @property
    def base_bijector(self) -> AbstractBijector:
        """
        Maps the base space onto the physical interval: affinely from the unit box
        where both ends are finite, and unchanged elsewhere.
        """
        return _base_space(self.lower, self.upper)[2]

    @property
    def bijector(self) -> AbstractBijector:
        # Where both ends are finite, squashes the real line into the unit box, then
        # reuses `base_bijector` to reach physical units, so the bounded and
        # unconstrained spaces cannot drift apart. The two then differ only by the
        # sigmoid: comparable step sizes through the bulk of the interval, with the
        # unconstrained side damping towards the bounds rather than meeting them.
        # Where an end is infinite, each element maps as the matching half-line or
        # real line does.
        lower_finite = jnp.isfinite(self.lower)
        upper_finite = jnp.isfinite(self.upper)
        # Stand-in bounds where an element takes another branch, so none is infinite.
        lower = jnp.where(lower_finite, self.lower, 0.0)
        upper = jnp.where(upper_finite, self.upper, 0.0)
        return Elementwise(
            masks=(
                ~lower_finite & ~upper_finite,
                lower_finite & upper_finite,
                lower_finite & ~upper_finite,
                ~lower_finite & upper_finite,
            ),
            bijectors=(
                Identity(),
                Chain([self.base_bijector, Sigmoid()]),
                GreaterThan(lower).bijector,
                LessThan(upper).bijector,
            ),
            safe_values=(
                jnp.array(0.0),
                jnp.array(0.5),
                lower + 1.0,
                upper - 1.0,
            ),
        )


class Positive(GreaterThan):
    """Convenience constraint for values that must be strictly positive: (0, ∞]."""
    def __init__(self, shape: Any = (), dtype: Any = None):
        """
        Args:
            shape: The shape of the parameter array.
            dtype: The JAX data type of the parameter array.
        """
        super().__init__(lower=jnp.zeros(shape, dtype=dtype), closed=(False, True))


class NonNegative(GreaterThan):
    """Convenience constraint for values that must be non-negative: [0, ∞]."""
    def __init__(self, shape: Any = (), dtype: Any = None):
        """
        Args:
            shape: The shape of the parameter array.
            dtype: The JAX data type of the parameter array.
        """
        super().__init__(lower=jnp.zeros(shape, dtype=dtype), closed=True)


class Negative(LessThan):
    """Convenience constraint for values that must be strictly negative: [-∞, 0)."""
    def __init__(self, shape: Any = (), dtype: Any = None):
        """
        Args:
            shape: The shape of the parameter array.
            dtype: The JAX data type of the parameter array.
        """
        super().__init__(upper=jnp.zeros(shape, dtype=dtype), closed=(True, False))


class NonPositive(LessThan):
    """Convenience constraint for values that must be non-positive: [-∞, 0]."""
    def __init__(self, shape: Any = (), dtype: Any = None):
        """
        Args:
            shape: The shape of the parameter array.
            dtype: The JAX data type of the parameter array.
        """
        super().__init__(upper=jnp.zeros(shape, dtype=dtype), closed=True)


def _mixed_elements(
    constraint: AbstractConstraint, bijector: AbstractBijector, like: PyTree
) -> PyTree:
    """
    Which elements of `bijector`'s output depend on more than the matching input element.

    Read off the Jacobian at the point raw zero maps to, an interior point of
    `constraint`. Returned in the structure of `like`, the bijector's output. Every
    element counts as mixed when the input and output sizes differ.
    """
    x = constraint.bijector.forward(jax.tree.map(jnp.zeros_like, constraint.bounds[0]))
    if jax.tree.structure(x) == jax.tree.structure(like):
        # A scalar base under an array-valued bijector acts on each element alike.
        x = jax.tree.map(lambda a, b: jnp.broadcast_to(a, jnp.broadcast_shapes(a.shape, b.shape)), x, like)
    flat_x, unravel_x = ravel_pytree(x)
    flat_like, unravel_like = ravel_pytree(like)

    if flat_x.size != flat_like.size:
        mixed = jnp.ones(flat_like.shape, dtype=bool)
    else:
        jacobian = jax.jacfwd(lambda f: ravel_pytree(bijector.forward(unravel_x(f)))[0])(flat_x)
        off_diagonal = jnp.where(jnp.eye(flat_x.size, dtype=bool), 0.0, jacobian)
        # NaN counts as nonzero, so an unreadable row counts as mixed.
        mixed = jnp.any(off_diagonal != 0, axis=1)

    return jax.tree.map(lambda m: m > 0, unravel_like(mixed.astype(flat_like.dtype)))


class Transformed(AbstractConstraint):
    """
    A constraint modified by an arbitrary distreqx bijector.
    
    The custom bijector is applied *after* the base constraint. This allows 
    for complex normalizations or transformations on top of physical boundaries.

    The bounds, and their closedness, are the base bounds pushed through the bijector
    element by element. Where the bijector mixes an element with others (a
    `TriangularLinear`, say), that element's bounds are ±∞, closed only where the base
    space is closed at both ends.

    Attributes:
        base_constraint: The underlying physical constraint applied first.
        bijector: The bijector applied on top of the base constraint.
    """
    base_constraint: AbstractConstraint
    transform_bijector: AbstractBijector
    bijector: AbstractBijector
    bounds: tuple[PyTree, PyTree]
    closed: tuple[PyTree, PyTree]

    def __init__(
        self, 
        constraint: AbstractConstraint, 
        bijector: AbstractBijector
    ):
        self.base_constraint = constraint
        self.transform_bijector = bijector
        self.bijector = Chain([bijector, constraint.bijector])
        
        # Bounds follow the base bounds through the bijector, element by element. Where
        # that says nothing, the bound is unknown and falls back to ±∞: an element
        # the bijector mixes with others, whose image depends on more than its own
        # bounds, or one whose pushed bound is NaN (∞ - ∞ or 0·∞ inside the bijector).
        lower, upper = constraint.bounds
        l_transformed = bijector.forward(lower)
        u_transformed = bijector.forward(upper)
        mixed = _mixed_elements(constraint, bijector, l_transformed)
        unknown = jax.tree.map(
            lambda m, l, u: m | jnp.isnan(l) | jnp.isnan(u), mixed, l_transformed, u_transformed
        )
        self.bounds = (
            jax.tree.map(
                lambda n, l, u: jnp.where(n, -jnp.inf, jnp.minimum(l, u)),
                unknown, l_transformed, u_transformed,
            ),
            jax.tree.map(
                lambda n, l, u: jnp.where(n, jnp.inf, jnp.maximum(l, u)),
                unknown, l_transformed, u_transformed,
            ),
        )

        # A decreasing bijector swaps the bounds, and their closedness with them. An
        # unknown bound is closed only where the base space is closed at both ends.
        lower_closed, upper_closed = constraint.closed
        swapped = jax.tree.map(jnp.greater, l_transformed, u_transformed)
        both_closed = jax.tree.map(jnp.logical_and, lower_closed, upper_closed)
        self.closed = (
            jax.tree.map(
                lambda n, s, b, lc, uc: jnp.where(n, b, jnp.where(s, uc, lc)),
                unknown, swapped, both_closed, lower_closed, upper_closed,
            ),
            jax.tree.map(
                lambda n, s, b, lc, uc: jnp.where(n, b, jnp.where(s, lc, uc)),
                unknown, swapped, both_closed, lower_closed, upper_closed,
            ),
        )

    @property
    def base_bounds(self) -> tuple[PyTree, PyTree]:
        return self.base_constraint.base_bounds

    @property
    def base_bijector(self) -> AbstractBijector:
        return Chain([self.transform_bijector, self.base_constraint.base_bijector])


class Leafwise(AbstractConstraint):
    """
    Represents a PyTree of constraints mapping over a PyTree of inputs.
    
    Useful for applying heterogeneous constraints to complex nested structures 
    (like `equinox.Module` instances) simultaneously.

    Attributes:
        tree: The PyTree containing `AbstractConstraint` leaves.
    """
    tree: PyTree[AbstractConstraint]

    def __init__(
        self, 
        tree: PyTree[AbstractConstraint],
    ):
        leaves = jax.tree.leaves(tree, is_leaf=is_constraint)
        if not leaves:
            raise ValueError("The pytree of `tree` cannot be empty.")
        self.tree = tree

    @property
    def bounds(self) -> tuple[PyTree[Array], PyTree[Array]]:
        def get_lower(node: Any) -> Any: return node.bounds[0]
        def get_upper(node: Any) -> Any: return node.bounds[1]
        lower = jax.tree.map(get_lower, self.tree, is_leaf=is_constraint)
        upper = jax.tree.map(get_upper, self.tree, is_leaf=is_constraint)
        return lower, upper

    @property
    def closed(self) -> tuple[PyTree, PyTree]:
        def get_lower(node: Any) -> Any: return node.closed[0]
        def get_upper(node: Any) -> Any: return node.closed[1]
        lower = jax.tree.map(get_lower, self.tree, is_leaf=is_constraint)
        upper = jax.tree.map(get_upper, self.tree, is_leaf=is_constraint)
        return lower, upper

    @property
    def bijector(self) -> AbstractBijector:
        def get_bijector(node: Any) -> Any: return node.bijector
        bijectors = jax.tree.map(get_bijector, self.tree, is_leaf=is_constraint)
        return LeafwiseBijector(bijectors)

    @property
    def base_bounds(self) -> tuple[PyTree[Array], PyTree[Array]]:
        def get_lower(node: Any) -> Any: return node.base_bounds[0]
        def get_upper(node: Any) -> Any: return node.base_bounds[1]
        
        lower = jax.tree.map(get_lower, self.tree, is_leaf=is_constraint)
        upper = jax.tree.map(get_upper, self.tree, is_leaf=is_constraint)
        return lower, upper

    @property
    def base_bijector(self) -> AbstractBijector:
        def get_bijector(node: Any) -> Any: return node.base_bijector
        
        bijectors = jax.tree.map(get_bijector, self.tree, is_leaf=is_constraint)
        return LeafwiseBijector(bijectors)
    

class Custom(AbstractConstraint):
    """
    An escape hatch for power users who need a specific distreqx bijector 
    mapping with predefined physical bounds.

    Attributes:
        bijector: The internal, user-defined distreqx bijector mapping from the
            unconstrained real line to the physical space.
        bounds: The manually defined physical boundaries `(lower, upper)`,
            each a PyTree matching the constrained value's structure.
        closed: Whether each of `bounds` is included, as a `(lower, upper)` pair
            of PyTrees matching `bounds`, with a bool (or bool array) for each leaf.
        base_bounds: The orthogonal base boundaries. Defaults to `bounds` if omitted.
        base_bijector: The bijector mapping from `base_bounds` to `bounds`.
            Defaults to `Identity` if omitted.
    """
    bijector: AbstractBijector
    bounds: tuple[PyTree, PyTree]
    closed: tuple[PyTree, PyTree]
    base_bounds: tuple[PyTree, PyTree]
    base_bijector: AbstractBijector

    def __init__(
        self,
        bijector: AbstractBijector,
        bounds: tuple[PyTree, PyTree] = (jnp.array(-jnp.inf), jnp.array(jnp.inf)),
        closed: bool | tuple[PyTree, PyTree] = True,
        base_bounds: tuple[PyTree, PyTree] | None = None,
        base_bijector: AbstractBijector | None = None
    ):
        """
        Args:
            bijector: The custom `distreqx` bijector.
            bounds: A tuple of `(lower, upper)` defining the physical
                boundaries of the constrained space, each a PyTree matching
                the constrained value's structure. Defaults to `(-inf, inf)`.
            closed: Whether the bounds themselves are included: one bool for every
                bound, or a `(lower, upper)` pair, each side a bool for all of that
                side's leaves or a PyTree of them matching the bound. Defaults to
                closed, infinite bounds included.
            base_bounds: Optional. A tuple of `(lower, upper)` defining the orthogonal
                base boundaries. If None, defaults to `bounds`.
            base_bijector: Optional. The bijector handling spatial skew/correlation.
                If None, defaults to `distreqx.bijectors.Identity`.
        """
        self.bijector = bijector
        self.bounds = tuple(jax.tree.map(jnp.asarray, b) for b in bounds)
        self.closed = tuple(
            jax.tree.map(lambda _: side, bound) if isinstance(side, bool) else side
            for side, bound in zip(_as_pair(closed), self.bounds)
        )

        # Default base_bounds to physical bounds if not provided
        if base_bounds is None:
            self.base_bounds = self.bounds
        else:
            self.base_bounds = tuple(jax.tree.map(jnp.asarray, b) for b in base_bounds)
            
        # Default base_bijector to Identity if not provided
        if base_bijector is None:
            self.base_bijector = Identity()
        else:
            self.base_bijector = base_bijector


@singledispatch
def infer_distribution_constraint(dist: dists.AbstractDistribution) -> AbstractConstraint:
    """
    Infers the physical support AND the optimal whitening bijector of a
    distreqx distribution, returning the corresponding constraint mapping.

    This is the *default* (leaf) handler: it assumes `dist` is a terminal
    distribution and whitens it via the copula transform, falling back to
    hard-coded physical bounds when no ICDF is available.

    The whitening only shapes the raw space. The bounds are the distribution's
    support, closed at a finite endpoint where the density there is positive and
    finite, and open at ±∞, where a prior puts no mass. The base space comes from
    the support too, not the prior: the unit box mapped affinely onto a support
    with two finite bounds, and the support itself otherwise.

    Container / structural distributions (Joint, Transformed, Combined, ...)
    are handled by the registered overloads below. To support a new
    distribution, register a handler with
    `@infer_distribution_constraint.register(...)` rather than editing this body.
    """
    # The TFP-Standard Copula Whitening (NormalCDF -> Quantile).
    # Automatically intercepts bounded/skewed distributions (Gamma, Beta,
    # Uniform, LogNormal, ...) and custom distributions, mapping them to an
    # isotropic Standard Normal raw space. The base space comes from the support.
    try:
        lower_bound = jnp.asarray(dist.icdf(0.0))
        upper_bound = jnp.asarray(dist.icdf(1.0))
    except (NotImplementedError, AttributeError, ValueError, TypeError):
        pass
    else:
        whitening_bijector = Chain([Quantile(dist), NormalCDF()])
        return _from_support(
            whitening_bijector,
            bounds=(lower_bound, upper_bound),
            closed=(_support_includes(dist, lower_bound), _support_includes(dist, upper_bound)),
        )

    # Fallback: hard-coded physical bounds for distributions lacking an ICDF.
    if isinstance(dist, (dists.LogNormal, dists.Gamma)):
        support = Positive(shape=dist.event_shape)
    elif isinstance(dist, dists.Beta):
        support = Interval(lower=jnp.zeros(dist.event_shape), upper=jnp.ones(dist.event_shape))
    elif isinstance(dist, dists.Uniform):
        support = Interval(lower=dist.low, upper=dist.high)
    else:
        return _on_real_line(dist, Identity())

    lower_bound, upper_bound = support.bounds
    return _from_support(
        support.bijector,
        bounds=(lower_bound, upper_bound),
        closed=(_support_includes(dist, lower_bound), _support_includes(dist, upper_bound)),
    )


def _support_includes(dist: dists.AbstractDistribution, endpoint: Array) -> Array:
    """
    Whether a support includes one of its endpoints: it must be finite, with a
    positive, finite density there. Without a density, a finite endpoint counts.
    """
    finite = jnp.isfinite(endpoint)
    try:
        return finite & jnp.isfinite(dist.log_prob(endpoint))
    except (NotImplementedError, AttributeError, ValueError, TypeError):
        return finite


def _from_support(
    bijector: AbstractBijector,
    bounds: tuple[PyTree, PyTree],
    closed: tuple[PyTree, PyTree],
) -> Custom:
    """A constraint over a distribution's support, with the base space the support gives."""
    lower, upper = bounds
    lower_leaves, treedef = jax.tree.flatten(lower)
    upper_leaves = treedef.flatten_up_to(upper)
    bases = [_base_space(l, u) for l, u in zip(lower_leaves, upper_leaves)]

    base_lower = treedef.unflatten([b[0] for b in bases])
    base_upper = treedef.unflatten([b[1] for b in bases])
    base_bijectors = treedef.unflatten([b[2] for b in bases])
    if jax.tree_util.treedef_is_leaf(treedef):
        base_bijector = base_bijectors
    else:
        base_bijector = LeafwiseBijector(base_bijectors)

    return Custom(
        bijector=bijector,
        bounds=bounds,
        closed=closed,
        base_bounds=(base_lower, base_upper),
        base_bijector=base_bijector,
    )


# --- Native flow whitening -----------------------------------------------


@infer_distribution_constraint.register(dists.Transformed)
def _infer_transformed(dist) -> AbstractConstraint:
    """Whiten the base distribution, then apply the flow bijector on top."""
    base_constraint = infer_distribution_constraint(dist.distribution)
    flowed = Transformed(constraint=base_constraint, bijector=dist.bijector)
    return _from_support(flowed.bijector, flowed.bounds, flowed.closed)


# --- Explicit Gaussian whitening -----------------------------------------
# Maps an isotropic N(0, I) latent space to the physical (correlated) space.
# The support is the real line, which is also the base space.


def _on_real_line(dist, bijector: AbstractBijector) -> Custom:
    """A constraint over the real line, open at ±∞, whitened in raw space by `bijector`."""
    return Custom(bijector=bijector, bounds=RealLine(shape=dist.event_shape).bounds, closed=False)


@infer_distribution_constraint.register(dists.Normal)
@infer_distribution_constraint.register(dists.Logistic)
def _infer_loc_scale(dist) -> AbstractConstraint:
    bijector = Chain(
        [Shift(dist.loc), ScalarAffine(shift=jnp.array(0.0), scale=dist.scale)]
    )
    return _on_real_line(dist, bijector)


@infer_distribution_constraint.register(dists.MultivariateNormalDiag)
def _infer_mvn_diag(dist) -> AbstractConstraint:
    bijector = Chain(
        [Shift(dist.loc), ScalarAffine(shift=jnp.array(0.0), scale=dist.scale_diag)]
    )
    return _on_real_line(dist, bijector)


@infer_distribution_constraint.register(dists.MultivariateNormalTri)
def _infer_mvn_tri(dist) -> AbstractConstraint:
    bijector = Chain([Shift(dist.loc), TriangularLinear(matrix=dist.scale_tri)])
    return _on_real_line(dist, bijector)


@infer_distribution_constraint.register(dists.MultivariateNormalFullCovariance)
def _infer_mvn_full(dist) -> AbstractConstraint:
    L = jnp.linalg.cholesky(dist.covariance_matrix)
    bijector = Chain([Shift(dist.loc), TriangularLinear(matrix=L)])
    return _on_real_line(dist, bijector)


# --- Recursive container handlers (version-guarded) -----------------------


def _infer_joint(dist) -> AbstractConstraint:
    """Recursively unwrap a Joint into a Leafwise tree of constraints."""
    constraints_tree = jax.tree.map(
        infer_distribution_constraint,
        dist.distributions,
        is_leaf=lambda x: isinstance(x, dists.AbstractDistribution),
    )
    return Leafwise(tree=constraints_tree)


def _infer_combined(dist) -> AbstractConstraint:
    """
    Each part of `Combined` owns a disjoint subset of leaves of the shared
    event pytree (`None` elsewhere), mirroring how `Combined` itself merges
    samples and values. We infer each part's own constraint the same way,
    then merge the resulting (`None`-holed) constraint trees together,
    exactly as `Combined` merges values.
    """
    def _pick(*leaves):
        for leaf in leaves:
            if leaf is not None:
                return leaf
        return None

    def _tree_of(part) -> PyTree:
        constraint = infer_distribution_constraint(part)
        return constraint.tree if isinstance(constraint, Leafwise) else constraint

    def _is_leaf(x: Any) -> bool:
        # `None` must also stop recursion here: `jax.tree.map`'s structural
        # matching across multiple trees is driven only by the first tree's
        # `is_leaf`, so a `None` hole in one part's constraint tree must be
        # recognized as a (terminal) leaf position too, or JAX tries to
        # match it against the *internal* structure of another part's real
        # (non-`None`) constraint object at the same position.
        return is_constraint(x) or x is None

    trees = [_tree_of(part) for part in dist.distributions]
    merged = jax.tree.map(_pick, *trees, is_leaf=_is_leaf)
    return Leafwise(tree=merged)


# Register the container handlers only for classes that are actually available:
# `Joint` is always (parax fills it in), `Combined` only with gvcallen's fork.
for _name, _handler in (
    ("Joint", _infer_joint),
    ("Combined", _infer_combined),
):
    _cls = getattr(dists, _name, None) or getattr(distreqx_dists, _name, None)
    if _cls is not None:
        infer_distribution_constraint.register(_cls)(_handler)

del _name, _handler, _cls


class AbstractConstrained(AbstractBounded[T]):
    """
    The abstract interface for a constrained PyTree.

    Used as a type check for `parax.is_constrained`.
    
    Implies that the PyTree has associated constraints (and therefore bounds),
    but does not necessarily enforce that the PyTree follows those constraints.

    Attributes:
        constraint: Returns the active constraint of the PyTree.
        bounds: Returns the current PyTree bounds. Each must have a matching PyTree structure as `self`.
    """
    constraint: eqx.AbstractVar[AbstractConstraint]
    bounds: eqx.AbstractVar[tuple[T, T]]


def is_constrained(x: Any) -> TypeGuard[AbstractConstrained]:
    """
    Returns True if `x` is an instance of `parax.AbstractConstrained`.
    
    Args:
        x: The object to check.
        
    Returns:
        True if `x` implements `AbstractConstrained`, False otherwise.
    """
    return isinstance(x, AbstractConstrained)


def tree_constraints(tree: PyTree) -> PyTree:
    """
    Extracts the individual constraints of a PyTree.
    
    Standard arrays default to `parax.constraints.RealLine`.

    Note that this function does not allow non-array/constrainable leaf nodes.
    If you have leaves in your tree that are neither arrays nor derive
    from `parax.constraints.AbstractConstrainable`, be sure to mark
    them as static or filter them out using e.g. `eqx.filter` first.    

    Args:
        tree: The PyTree model to extract constraints from.

    Returns:
        A PyTree representing the active constraints.
    """
    from parax.wrappers import as_unwrapped
    
    def _get_constraint(x):
        if is_constrained(x):
            return as_unwrapped(x.constraint)
        if eqx.is_inexact_array(x):
            return RealLine(shape=x.shape)
        raise ValueError(
            f"Found a leaf node of type {type(x)} that is neither constrained "
            f"nor an array in `parax.constraints.tree_constraints`. Value: {x}"
        )

    return jax.tree_util.tree_map(_get_constraint, tree, is_leaf=is_constrained)


def tree_leafwise_constraint(tree: PyTree) -> Leafwise:
    """
    Extracts the single leafwise constraint of a PyTree.
    
    Wraps the output of `parax.constraints.tree_constraints`
    in a `parax.constraints.Leafwise` constraint to define
    a single constraint that matches the shape of `tree`.

    Args:
        tree: The PyTree model containing probabilistic nodes or standard arrays.

    Returns:
        A single constraint whose shape matches the structure of `tree`.
    """
    return Leafwise(tree_constraints(tree)) 


class AbstractConstrainable(AbstractConstrained[T]):
    """
    The abstract interface for a constrainable PyTree.

    Variables implementing this interface support the dynamic injection
    and updating of constraints.

    Used as a type check for `parax.is_constrainable`.
    """
    @abstractmethod
    def constrain(self, constraint: AbstractConstraint) -> Self:
        """
        Returns a new instance of the PyTree with the updated constraint,
        ensuring internal state (like unconstrained raw values) is 
        recalculated if necessary.

        Args:
            constraint: The new constraint to apply.

        Returns:
            A new instance of the constrainable PyTree.
        """
        raise NotImplementedError


def is_constrainable(x: Any) -> TypeGuard[AbstractConstrainable]:
    """
    Returns True if `x` is an instance of `parax.AbstractConstrainable`.
    
    Args:
        x: The object to check.
        
    Returns:
        True if `x` implements `AbstractConstrainable`, False otherwise.
    """
    return isinstance(x, AbstractConstrainable)


def tree_constrain(tree: PyTree, constraints: PyTree) -> PyTree:
    """
    Applies a PyTree of constraints to a PyTree of constrainable PyTrees.
    
    Standard arrays will be returned untouched if the matching constraint 
    is a `RealLine`. Attempting to apply a bounded constraint directly 
    to a standard array will raise an error.

    Args:
        tree: The PyTree model to update. Must have a matching PyTree structure 
            to `constraints`.
        constraints: A PyTree of `parax.AbstractConstraint` objects.

    Returns:
        A new PyTree with the constraints applied.
    """
    def _apply_constraint(x, c):
        if is_constrainable(x):
            return x.constrain(c)
        if eqx.is_inexact_array(x):
            if isinstance(c, RealLine):
                return x
            raise TypeError(
                "Cannot apply a bounded constraint to a raw JAX array directly. "
                "Ensure the array is wrapped in a `parax.Constrained` variable first."
            )
        raise ValueError(
            f"Found a leaf node of type {type(x)} that is neither constrainable "
            f"nor an array in `parax.constraints.tree_constrain`. Value: {x}"
        )

    return jax.tree_util.tree_map(
        _apply_constraint, tree, constraints, is_leaf=is_constrainable
    )
    
    
def intersect(a: AbstractConstraint, b: AbstractConstraint) -> AbstractConstraint:
    """
    Calculates the intersection of two constraints.
    Returns the most specific constraint class possible.

    Each bound is the tighter of the two, keeping its closedness. Where both
    constraints share a bound, finite or infinite, open wins. The result is a named
    class (`RealLine`, `Positive`, ...) only when that class's closedness matches.
    """
    a_lower, a_upper = a.bounds
    b_lower, b_upper = b.bounds
    a_lower_closed, a_upper_closed = a.closed
    b_lower_closed, b_upper_closed = b.closed

    lower = jnp.maximum(a_lower, b_lower)
    upper = jnp.minimum(a_upper, b_upper)

    def _closedness(a_bound, b_bound, a_closed, b_closed, a_tighter):
        closed = jnp.where(
            a_tighter,
            a_closed,
            jnp.where(a_bound == b_bound, jnp.logical_and(a_closed, b_closed), b_closed),
        )
        # One flag per bound: open wins wherever the elements disagree.
        return bool(jnp.all(closed))

    lower_closed = _closedness(a_lower, b_lower, a_lower_closed, b_lower_closed, a_lower > b_lower)
    upper_closed = _closedness(a_upper, b_upper, a_upper_closed, b_upper_closed, a_upper < b_upper)

    # Convert to concrete numpy arrays for boolean checks during init
    np_lower = jnp.asarray(lower)
    np_upper = jnp.asarray(upper)

    np_lower, np_upper = eqx.error_if(
        (np_lower, np_upper),
        jnp.any(jnp.greater_equal(np_lower, np_upper)),
        f"Constraint intersection is empty or invalid."
    )
    
    is_neginf_lower = jnp.all(jnp.isneginf(np_lower))
    is_posinf_upper = jnp.all(jnp.isposinf(np_upper))
    is_zero_lower = jnp.all(jnp.equal(np_lower, 0.0))
    is_zero_upper = jnp.all(jnp.equal(np_upper, 0.0))

    # Resolve to the most specific constraint class whose closedness matches
    closed = (lower_closed, upper_closed)
    if is_neginf_lower and is_posinf_upper and closed == (True, True):
        return RealLine()
    elif is_zero_lower and is_posinf_upper and closed == (True, True):
        return NonNegative()
    elif is_zero_lower and is_posinf_upper and closed == (False, True):
        return Positive()
    elif is_neginf_lower and is_zero_upper and closed == (True, True):
        return NonPositive()
    elif is_neginf_lower and is_zero_upper and closed == (True, False):
        return Negative()
    elif is_posinf_upper and not is_neginf_lower:
        return GreaterThan(lower, closed=closed)
    elif is_neginf_lower and not is_posinf_upper:
        return LessThan(upper, closed=closed)
    else:
        return Interval(lower, upper, closed=closed)
    

def _is_unwrappable_constrained(x):
    from parax.wrappers import is_unwrappable
    return is_constrained(x) and is_unwrappable(x) 

def is_leaf(x):
    """Defines the tree traversal boundaries for constrained partitioning."""
    from parax.constants import is_constant
    return _is_unwrappable_constrained(x) or is_constant(x)

def is_dynamic(x):
    """Identifies parameters that should be updated during constrained inference."""    
    from parax.constants import is_constant
    if is_constant(x): 
        return False
    if _is_unwrappable_constrained(x): 
        return True
    return eqx.is_inexact_array(x)