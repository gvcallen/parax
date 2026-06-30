"""
Constraints with bijector mapping and associated interfaces.

This module provides the tools to map unconstrained optimizer spaces 
(spanning the real line) into bounded physical spaces, as well as
to mark a PyTree as constrainable.
"""

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
        bounds: A tuple containing the lower and upper bounds of the constrained space.
        bijector: A `distreqx.bijectors.AbstractBijector` mapping from the unconstrained real line to the constrained space.
    """
    bounds: eqx.AbstractVar[tuple[PyTree, PyTree]]
    bijector: eqx.AbstractVar[AbstractBijector]
    
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


class RealLine(AbstractConstraint):
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
        try:
            from distreqx.bijectors import Identity
        except:
            from parax._bijectors import Identity
        return Identity()


class GreaterThan(AbstractConstraint):
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
        try:
            from distreqx.bijectors import Softplus
        except ImportError:
            from parax._bijectors import Softplus
        from distreqx.bijectors import Chain, Shift

        return Chain([Shift(self.lower), Softplus()])


class LessThan(AbstractConstraint):
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
        try:
            from distreqx.bijectors import Softplus
        except:
            from parax._bijectors import Softplus

        # Corner Case Note: To implement a LessThan constraint using Softplus 
        # (which inherently bounds > 0), we apply a double affine flip:
        # invert -> softplus -> invert -> shift.
        return Chain([
            Shift(self.upper),
            ScalarAffine(shift=jnp.array(0.0), scale=jnp.array(-1.0)),
            Softplus(),
            ScalarAffine(shift=jnp.array(0.0), scale=jnp.array(-1.0)),
        ])


class Interval(AbstractConstraint):
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
    def bijector(self) -> AbstractBijector:
        scale_val = self.upper - self.lower
        return Chain([
            Shift(self.lower), 
            ScalarAffine(shift=jnp.array(0.0), scale=scale_val),
            Sigmoid()
        ])


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
        custom_bijector: The bijector applied on top of the base constraint.
    """
    bijector: AbstractBijector
    bounds: tuple[PyTree, PyTree]

    def __init__(
        self, 
        constraint: AbstractConstraint, 
        bijector: AbstractBijector
    ):
        """
        Args:
            constraint: The base physical constraint (e.g., Positive, Interval).
            bijector: The bijector to apply on top of the base constraint.
        """
        
        # Bijector
        self.bijector = Chain([bijector, constraint.bijector])
        
        # Bounds
        lower, upper = constraint.bounds
        l_transformed = bijector.forward(lower)
        u_transformed = bijector.forward(upper)
        self.bounds = jax.tree.map(jnp.minimum, l_transformed, u_transformed), jax.tree.map(jnp.maximum, l_transformed, u_transformed)


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
        """
        Args:
            tree: A PyTree containing `AbstractConstraint` leaves.
                Non-constraint leaves are ignored.
        
        Raises:
            ValueError: If the provided PyTree contains no constraint leaves.
        """
        # Local import prevents circular dependency at initialization time
        leaves = jax.tree.leaves(tree, is_leaf=is_constraint)
        if not leaves:
            raise ValueError("The pytree of `tree` cannot be empty.")

        self.tree = tree

    @property
    def bounds(self) -> tuple[PyTree[Array], PyTree[Array]]:
        """
        Extracts a PyTree of lower bounds and a PyTree of upper bounds.
        Non-constraint nodes in the original PyTree are left unmodified.
        """
        def get_lower(node: Any) -> Any:
            if not is_constraint(node):
                raise ValueError(f"Found a leaf node of type {type(node)} that is not a constraint in `parax.constraints.Leafwise`. Value: {node}")
            return node.bounds[0]
        
        def get_upper(node: Any) -> Any:
            if not is_constraint(node):
                raise ValueError(f"Found a leaf node of type {type(node)} that is not a constraint in `parax.constraints.Leafwise`. Value: {node}")
            return node.bounds[1]
                
        lower = jax.tree.map(get_lower, self.tree, is_leaf=is_constraint)
        upper = jax.tree.map(get_upper, self.tree, is_leaf=is_constraint)
        
        return lower, upper

    @property
    def bijector(self) -> AbstractBijector:
        """
        Returns a `distreqx.bijectors.Leafwise` bijector that applies each respective 
        leaf constraint's bijector.
        """
        from distreqx.bijectors import Leafwise as LeafwiseBijector

        def get_bijector(node: Any) -> Any:
            if not is_constraint(node):
                raise ValueError(f"Found a leaf node of type {type(node)} that is not a constraint in `parax.constraints.Leafwise`. Value: {node}")
            return node.bijector
        
        bijector = jax.tree.map(get_bijector, self.tree, is_leaf=is_constraint)

        return LeafwiseBijector(bijector)
    

class Custom(AbstractConstraint):
    """
    An escape hatch for power users who need a specific distreqx bijector 
    mapping with predefined physical bounds.

    Attributes:
        _custom_bijector: The internal, user-defined distreqx bijector.
        _custom_bounds: The manually defined physical boundaries `(lower, upper)`.
    """
    bijector: AbstractBijector
    bounds: tuple[Array, Array]

    def __init__(
        self, 
        bijector: AbstractBijector, 
        bounds: tuple[Array, Array] = (jnp.array(-jnp.inf), jnp.array(jnp.inf))
    ):
        """
        Args:
            bijector: The custom `distreqx` bijector.
            bounds: A tuple of `(lower, upper)` defining the physical 
                boundaries of the constrained space. Defaults to `(-inf, inf)`.
        """
        self.bijector = bijector
        self.bounds = tuple(jnp.asarray(b) for b in bounds)


def infer_distribution_constraint(dist: dists.AbstractDistribution) -> AbstractConstraint:
    """
    Infers the physical support AND the optimal whitening bijector of a 
    distreqx distribution, returning the corresponding constraint mapping.
    """
    # Recursive unwrapping for Joint Distributions
    if hasattr(dists, 'Joint') and isinstance(dist, dists.Joint):
        sub_distributions = dist.distributions 
        constraints_tree = jax.tree.map(
            infer_distribution_constraint, 
            sub_distributions,
            is_leaf=lambda x: isinstance(x, dists.AbstractDistribution)
        )
        return Leafwise(tree=constraints_tree)
        
    # Native Flow Whitening for Transformed Distributions
    elif isinstance(dist, dists.Transformed):
        base_constraint = infer_distribution_constraint(dist.distribution)
        return Transformed(
            constraint=base_constraint, 
            bijector=dist.bijector
        )

    # 3. Fast-Path for natively unconstrained isotropic distributions.
    # We skip the Copula for these to avoid expensive erf/erfinv cycles.
    if isinstance(dist, (dists.Normal, dists.Logistic)):
        return RealLine(shape=dist.event_shape)
        
    elif isinstance(dist, (dists.MultivariateNormalDiag, 
                           dists.MultivariateNormalFullCovariance, 
                           dists.MultivariateNormalTri)):
        return RealLine(shape=dist.event_shape)

    # The TFP-Standard Copula Whitening (NormalCDF -> Quantile)
    # Automatically intercepts bounded/skewed distributions (Gamma, Beta, Uniform) 
    # and custom distributions to map them to an isotropic Standard Normal space.
    try:
        lower_bound = dist.icdf(0.0)
        upper_bound = dist.icdf(1.0)
        
        # Explicit right-to-left Copula transformation
        whitening_bijector = Chain([Quantile(dist), NormalCDF()])
        
        return Custom(
            bijector=whitening_bijector,
            bounds=(lower_bound, upper_bound)
        )
    except (NotImplementedError, AttributeError, ValueError, TypeError):
        pass

    # Fallback: Hard-coded physical bounds for distributions lacking an ICDF
    if (hasattr(dists, 'LogNormal') and isinstance(dist, (dists.LogNormal)) or isinstance(dist, dists.Gamma)):
        return Positive(shape=dist.event_shape)

    elif isinstance(dist, dists.Beta):
        return Interval(
            lower=jnp.zeros(dist.event_shape), 
            upper=jnp.ones(dist.event_shape)
        )

    elif isinstance(dist, dists.Uniform):
        return Interval(lower=dist.low, upper=dist.high)

    return RealLine(shape=dist.event_shape)


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
    from parax.wrappers import unwrap
    
    def _get_constraint(x):
        if is_constrained(x):
            return unwrap(x.constraint)
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


def tree_leafwise_bijector(tree: PyTree) -> AbstractBijector:
    """
    Extracts the constraint bijector of a PyTree.
    
    Returns the bijector of `parax.probability.tree_leafwise_constraint`,
    which transforms unconstrained values to constrained values.

    Args:
        tree: The PyTree model containing probabilistic nodes or standard arrays.

    Returns:
        A single bijector whose shape matches the structure of `tree`.
    """
    return tree_leafwise_constraint(tree).bijector    


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