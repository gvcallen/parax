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
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

import distreqx.distributions as dists
from distreqx.bijectors import (
    AbstractBijector, 
    Sigmoid, 
    Chain, 
    Shift, 
    ScalarAffine,
    TriangularLinear,
)

from parax.bounds import AbstractBounded
from parax._bijectors import NormalCDF, Quantile

T = TypeVar("Value")

try:
    from distreqx.bijectors import Leafwise as LeafwiseBijector
except:
    from parax._bijectors import Leafwise as LeafwiseBijector

try:
    from distreqx.bijectors import Identity
except ImportError:
    from parax._bijectors import Identity    

try:
    from distreqx.bijectors import Softplus
except ImportError:
    from parax._bijectors import Softplus

class AbstractConstraint(eqx.Module):
    """
    The base class for all physical constraints in Parax.
    
    Constraints are a higher-level concept that provide bounds and bijectors over constrained domains.
    This is useful for use with unconstrained solvers (which require a bijector from the
    unconstrained real line to the constrained domain) and bounded solvers (which accept
    lower and upper bounds directly).

    Attributes:
        bounds: A tuple containing the physical lower and upper bounds of the constrained space.
        bijector: A `distreqx.bijectors.AbstractBijector` mapping from the unconstrained real line to the physical space.
        base_bounds: A tuple containing the foundational, un-skewed orthogonal bounds. Where a
            whitened base space exists this is the normalised box the optimizer works in, so that
            a step of a given size carries the same meaning along every axis: the unit box for an
            `Interval`, and the probability-integral-transform space for a constraint inferred
            from a distribution. For transformed constraints, this isolates the safe topological
            box before any dense correlations or skews are applied. It falls back to `bounds` only
            where no such normalisation exists, such as an unbounded or half-bounded domain.
        base_bijector: A `distreqx.bijectors.AbstractBijector` mapping from the orthogonal `base_bounds`
            space into the physical `bounds` space. Defaults to `Identity` unless geometric skews are present.
    """
    bounds: eqx.AbstractVar[tuple[PyTree, PyTree]]
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
        """
        lower, upper = self.bounds
        return jax.tree.map(lambda x, l, u: jnp.logical_or(x < l, x > u), value, lower, upper)
    
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
    useful for maintaining consistent types in mixed parameter sets.

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
    def bijector(self) -> AbstractBijector:
        return Identity()


class GreaterThan(AbstractUncorrelatedConstraint):
    """
    Represents a value strictly greater than a lower bound.
    
    Attributes:
        lower: The exclusive lower bound array or scalar.
    """
    lower: jnp.ndarray
    
    def __init__(self, lower: Union[float, Array]):
        """
        Args:
            lower: The exclusive lower bound.
        """
        self.lower = jnp.asarray(lower, dtype=float)
        
    @property
    def bounds(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        return (self.lower, jnp.full_like(self.lower, jnp.inf))

    @property
    def bijector(self) -> AbstractBijector:
        from distreqx.bijectors import Chain, Shift

        return Chain([Shift(self.lower), Softplus()])


class LessThan(AbstractUncorrelatedConstraint):
    """
    Represents a value strictly less than an upper bound.
    
    Attributes:
        upper: The exclusive upper bound array or scalar.
    """
    upper: jnp.ndarray

    def __init__(self, upper: Union[float, Array]):
        """
        Args:
            upper: The exclusive upper bound.
        """
        self.upper = jnp.asarray(upper, dtype=float)

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


class Interval(AbstractUncorrelatedConstraint):
    """
    Represents a value strictly bounded between a lower and upper value.
    
    Attributes:
        lower: The exclusive lower bound.
        upper: The exclusive upper bound.
    """
    lower: jnp.ndarray
    upper: jnp.ndarray

    def __init__(self, lower: Union[float, Array], upper: Union[float, Array]):
        """
        Args:
            lower: The exclusive lower bound.
            upper: The exclusive upper bound.
        """
        self.lower = jnp.asarray(lower, dtype=float)
        self.upper = jnp.asarray(upper, dtype=float)

    @property
    def bounds(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        return (self.lower, self.upper)

    @property
    def base_bounds(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        """
        The unit box.

        Overrides the `Identity` default from `AbstractUncorrelatedConstraint`, which
        would hand a bounded optimizer the physical box. Normalising to `[0, 1]`
        is generally numerically better during optimization.
        """
        return (jnp.zeros_like(self.lower), jnp.ones_like(self.upper))

    @property
    def base_bijector(self) -> AbstractBijector:
        """
        Maps the unit box onto the physical interval.
        """
        return Chain([
            Shift(self.lower),
            ScalarAffine(shift=jnp.array(0.0), scale=self.upper - self.lower),
        ])

    @property
    def bijector(self) -> AbstractBijector:
        # Squashes the real line into the unit box, then reuses `base_bijector` to
        # reach physical units, so the bounded and unconstrained spaces cannot drift
        # apart. The two then differ only by the sigmoid: comparable step sizes through
        # the bulk of the interval, with the unconstrained side damping towards the
        # bounds rather than meeting them.
        return Chain([self.base_bijector, Sigmoid()])


class Positive(GreaterThan):
    """Convenience constraint for values that must be strictly positive (> 0)."""
    def __init__(self, shape: Any = (), dtype: Any = None):
        """
        Args:
            shape: The shape of the parameter array.
            dtype: The JAX data type of the parameter array.
        """
        super().__init__(lower=jnp.zeros(shape, dtype=dtype))


class Negative(LessThan):
    """Convenience constraint for values that must be strictly negative (< 0)."""
    def __init__(self, shape: Any = (), dtype: Any = None):
        """
        Args:
            shape: The shape of the parameter array.
            dtype: The JAX data type of the parameter array.
        """
        super().__init__(upper=jnp.zeros(shape, dtype=dtype))


class Transformed(AbstractConstraint):
    """
    A constraint modified by an arbitrary distreqx bijector.
    
    The custom bijector is applied *after* the base constraint. This allows 
    for complex normalizations or transformations on top of physical boundaries.

    Attributes:
        base_constraint: The underlying physical constraint applied first.
        bijector: The bijector applied on top of the base constraint.
    """
    base_constraint: AbstractConstraint
    transform_bijector: AbstractBijector
    bijector: AbstractBijector
    bounds: tuple[PyTree, PyTree]

    def __init__(
        self, 
        constraint: AbstractConstraint, 
        bijector: AbstractBijector
    ):
        self.base_constraint = constraint
        self.transform_bijector = bijector
        self.bijector = Chain([bijector, constraint.bijector])
        
        # Bounds
        lower, upper = constraint.bounds
        l_transformed = bijector.forward(lower)
        u_transformed = bijector.forward(upper)
        self.bounds = (
            jax.tree.map(jnp.minimum, l_transformed, u_transformed), 
            jax.tree.map(jnp.maximum, l_transformed, u_transformed)
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
        base_bounds: The orthogonal base boundaries. Defaults to `bounds` if omitted.
        base_bijector: The bijector mapping from `base_bounds` to `bounds`.
            Defaults to `Identity` if omitted.
    """
    bijector: AbstractBijector
    bounds: tuple[PyTree, PyTree]
    base_bounds: tuple[PyTree, PyTree]
    base_bijector: AbstractBijector

    def __init__(
        self,
        bijector: AbstractBijector,
        bounds: tuple[PyTree, PyTree] = (jnp.array(-jnp.inf), jnp.array(jnp.inf)),
        base_bounds: tuple[PyTree, PyTree] | None = None,
        base_bijector: AbstractBijector | None = None
    ):
        """
        Args:
            bijector: The custom `distreqx` bijector.
            bounds: A tuple of `(lower, upper)` defining the physical
                boundaries of the constrained space, each a PyTree matching
                the constrained value's structure. Defaults to `(-inf, inf)`.
            base_bounds: Optional. A tuple of `(lower, upper)` defining the orthogonal
                base boundaries. If None, defaults to `bounds`.
            base_bijector: Optional. The bijector handling spatial skew/correlation.
                If None, defaults to `distreqx.bijectors.Identity`.
        """
        self.bijector = bijector
        self.bounds = tuple(jax.tree.map(jnp.asarray, b) for b in bounds)

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

    Container / structural distributions (Joint, Transformed, Combined, ...)
    are handled by the registered overloads below. To support a new
    distribution, register a handler with
    `@infer_distribution_constraint.register(...)` rather than editing this body.
    """
    # The TFP-Standard Copula Whitening (NormalCDF -> Quantile).
    # Automatically intercepts bounded/skewed distributions (Gamma, Beta,
    # Uniform, LogNormal, ...) and custom distributions, mapping them to an
    # isotropic Standard Normal space.
    try:
        lower_bound = dist.icdf(0.0)
        upper_bound = dist.icdf(1.0)

        # The quantile alone maps the probability-integral-transform space onto the
        # physical one, so `[0, 1]` serves as a base that is both bounded and already
        # whitened: under the transform the prior is uniform there by construction.
        # Without it a bounded solver falls back to `Identity` over the physical box
        # and gets no whitening at all, however wide that box happens to be.
        quantile = Quantile(dist)
        whitening_bijector = Chain([quantile, NormalCDF()])
        return Custom(
            bijector=whitening_bijector,
            bounds=(lower_bound, upper_bound),
            base_bounds=(jnp.zeros_like(lower_bound), jnp.ones_like(upper_bound)),
            base_bijector=quantile,
        )
    except (NotImplementedError, AttributeError, ValueError, TypeError):
        pass

    # Fallback: hard-coded physical bounds for distributions lacking an ICDF.
    if (hasattr(dists, "LogNormal") and isinstance(dist, dists.LogNormal)) or isinstance(
        dist, dists.Gamma
    ):
        return Positive(shape=dist.event_shape)
    elif isinstance(dist, dists.Beta):
        return Interval(
            lower=jnp.zeros(dist.event_shape),
            upper=jnp.ones(dist.event_shape),
        )
    elif isinstance(dist, dists.Uniform):
        return Interval(lower=dist.low, upper=dist.high)

    return RealLine(shape=dist.event_shape)


# --- Native flow whitening -----------------------------------------------


@infer_distribution_constraint.register(dists.Transformed)
def _infer_transformed(dist) -> AbstractConstraint:
    """Whiten the base distribution, then apply the flow bijector on top."""
    base_constraint = infer_distribution_constraint(dist.distribution)
    return Transformed(constraint=base_constraint, bijector=dist.bijector)


# --- Explicit Gaussian whitening -----------------------------------------
# Maps an isotropic N(0, I) latent space to the physical (correlated) space.


@infer_distribution_constraint.register(dists.Normal)
@infer_distribution_constraint.register(dists.Logistic)
def _infer_loc_scale(dist) -> AbstractConstraint:
    base_constraint = RealLine(shape=dist.event_shape)
    bijector = Chain(
        [Shift(dist.loc), ScalarAffine(shift=jnp.array(0.0), scale=dist.scale)]
    )
    return Transformed(constraint=base_constraint, bijector=bijector)


@infer_distribution_constraint.register(dists.MultivariateNormalDiag)
def _infer_mvn_diag(dist) -> AbstractConstraint:
    base_constraint = RealLine(shape=dist.event_shape)
    bijector = Chain(
        [Shift(dist.loc), ScalarAffine(shift=jnp.array(0.0), scale=dist.scale_diag)]
    )
    return Transformed(constraint=base_constraint, bijector=bijector)


@infer_distribution_constraint.register(dists.MultivariateNormalTri)
def _infer_mvn_tri(dist) -> AbstractConstraint:
    base_constraint = RealLine(shape=dist.event_shape)
    bijector = Chain([Shift(dist.loc), TriangularLinear(matrix=dist.scale_tri)])
    return Transformed(constraint=base_constraint, bijector=bijector)


@infer_distribution_constraint.register(dists.MultivariateNormalFullCovariance)
def _infer_mvn_full(dist) -> AbstractConstraint:
    base_constraint = RealLine(shape=dist.event_shape)
    L = jnp.linalg.cholesky(dist.covariance_matrix)
    bijector = Chain([Shift(dist.loc), TriangularLinear(matrix=L)])
    return Transformed(constraint=base_constraint, bijector=bijector)


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


# Register the container handlers only if the classes exist in the installed
# distreqx (mirroring the original `hasattr` guards).
for _name, _handler in (
    ("Joint", _infer_joint),
    ("Combined", _infer_combined),
):
    _cls = getattr(dists, _name, None)
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
    """
    a_lower, a_upper = a.bounds
    b_lower, b_upper = b.bounds

    lower = jnp.maximum(a_lower, b_lower)
    upper = jnp.minimum(a_upper, b_upper)

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

    # Resolve to the most specific constraint class
    if is_neginf_lower and is_posinf_upper:
        return RealLine()
    elif is_zero_lower and is_posinf_upper:
        return Positive()
    elif is_neginf_lower and is_zero_upper:
        return Negative()
    elif is_posinf_upper:
        return GreaterThan(lower)
    elif is_neginf_lower:
        return LessThan(upper)
    else:
        return Interval(lower, upper)
    

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