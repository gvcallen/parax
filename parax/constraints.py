"""
Physical constraints and bijector mappings for parametric modeling.

This module provides the tools to map unconstrained optimizer spaces 
(spanning the real line) into bounded physical spaces.
"""

from typing import Union, Any, TypeGuard

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
    base_constraint: AbstractConstraint
    custom_bijector: AbstractBijector

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
        self.base_constraint = constraint
        self.custom_bijector = bijector

    @property
    def bounds(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        """
        Calculates the new topological bounds by passing the base extrema 
        through the custom bijector.

        **Corner Case Note:** Uses `jnp.minimum` and `jnp.maximum` to gracefully 
        handle monotonically decreasing bijectors that might invert the order 
        of the lower and upper bounds.
        """
        lower, upper = self.base_constraint.bounds
        
        # Pass boundaries through
        l_transformed = self.custom_bijector.forward(lower)
        u_transformed = self.custom_bijector.forward(upper)
        
        # Cater for monotonically decreasing bijectors
        return jnp.minimum(l_transformed, u_transformed), jnp.maximum(l_transformed, u_transformed)

    @property
    def bijector(self) -> AbstractBijector:
        """
        The composed bijector mapping from the unconstrained optimizer space 
        to the fully transformed physical space.
        """
        return Chain([self.custom_bijector, self.base_constraint.bijector])
  

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
    _custom_bijector: AbstractBijector
    _custom_bounds: tuple[Array, Array]

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
        self._custom_bijector = bijector
        self._custom_bounds = tuple(jnp.asarray(b) for b in bounds)

    @property
    def bounds(self) -> tuple[Array, Array]:
        return self._custom_bounds

    @property
    def bijector(self) -> AbstractBijector:
        return self._custom_bijector
    

def get_constraint_for_distribution(dist: dists.AbstractDistribution) -> AbstractConstraint:
    """
    Infers the physical support of a distreqx distribution and returns 
    the corresponding constraint mapping.

    Resolution Strategy:
    1. Exact type matching for common `distreqx` distributions.
    2. Fallback to `icdf(0.0)` and `icdf(1.0)` evaluation.
    3. Last resort: Unconstrained `RealLine`.
    """
    if isinstance(dist, (dists.Normal, dists.Logistic)):
        return RealLine(shape=dist.event_shape)
        
    elif isinstance(dist, (dists.MultivariateNormalDiag, 
                           dists.MultivariateNormalFullCovariance, 
                           dists.MultivariateNormalTri)):
        return RealLine(shape=dist.event_shape)

    elif isinstance(dist, (dists.LogNormal, dists.Gamma)):
        return Positive(shape=dist.event_shape)

    elif isinstance(dist, dists.Beta):
        return Interval(
            lower=jnp.zeros(dist.event_shape), 
            upper=jnp.ones(dist.event_shape)
        )

    elif isinstance(dist, dists.Uniform):
        return Interval(lower=dist.low, upper=dist.high)

    try:
        lower_bound = dist.icdf(0.0)
        upper_bound = dist.icdf(1.0)
        
        is_lower_bounded = not jnp.all(jnp.isneginf(lower_bound))
        is_upper_bounded = not jnp.all(jnp.isposinf(upper_bound))

        if is_lower_bounded and not is_upper_bounded:
            if jnp.all(lower_bound == 0.0):
                return Positive(shape=dist.event_shape, dtype=lower_bound.dtype)
            return GreaterThan(lower=lower_bound)

        elif not is_lower_bounded and is_upper_bounded:
            if jnp.all(upper_bound == 0.0):
                return Negative(shape=dist.event_shape, dtype=upper_bound.dtype)
            return LessThan(upper=upper_bound)

        elif is_lower_bounded and is_upper_bounded:
            return Interval(lower=lower_bound, upper=upper_bound)

    except (NotImplementedError, AttributeError, ValueError, TypeError):
        pass

    return RealLine(shape=dist.event_shape)