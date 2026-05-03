"""
Physical constraints and bijector mappings for parametric modeling.

This module provides the tools to map unconstrained optimizer spaces 
(spanning the real line) into bounded physical spaces.
"""

from typing import Union, Any

import jax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

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
    
    A constraint acts as a bridge between hard physical boundaries (used by 
    bounded optimizers or user inspection) and topological mappings (used by 
    unconstrained ML optimizers).

    Constraints may be used directly on arrays or mapped over PyTrees.
    """
    bounds: eqx.AbstractVar[tuple[PyTree, PyTree]]
    bijector: eqx.AbstractVar[AbstractBijector]


class RealLine(AbstractConstraint):
    """
    Represents a value that can span the entire real number line.
    
    Effectively a structural no-op constraint using an Identity bijector, 
    useful for maintaining consistent types in mixed parameter sets.
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
    """Represents a value strictly greater than a lower bound."""
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
    """Represents a value strictly less than an upper bound."""
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
    """Represents a value strictly bounded between a lower and upper value."""
    lower: jnp.ndarray
    upper: jnp.ndarray

    def __init__(self, lower: Union[float, Array], upper: Union[float, Array]):
        """
        Args:
            lower: The exclusive lower bound.
            upper: The exclusive upper bound.
        """
        self.lower = jnp.asarray(lower)
        self.upper = jnp.asarray(upper)

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


class TransformedConstraint(AbstractConstraint):
    """
    A constraint modified by an arbitrary distreqx bijector.
    
    The custom bijector is applied *after* the base constraint. This allows 
    for complex normalizations or transformations on top of physical boundaries.
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
  

class TreeConstraint(AbstractConstraint):
    """
    Represents a PyTree of constraints mapping over a PyTree of inputs.
    
    Useful for applying heterogeneous constraints to complex nested structures 
    (like `equinox.Module` instances) simultaneously.
    """
    tree: PyTree[AbstractConstraint]

    def __init__(
        self, 
        constraints: PyTree[AbstractConstraint],
    ):
        """
        Args:
            constraints: A PyTree containing `AbstractConstraint` leaves.
                Non-constraint leaves are ignored.
        
        Raises:
            ValueError: If the provided PyTree contains no constraint leaves.
        """
        # Local import prevents circular dependency at initialization time
        from parax.filters import is_constraint

        leaves = jax.tree.leaves(constraints, is_leaf=is_constraint)
        if not leaves:
            raise ValueError("The pytree of constraints cannot be empty.")

        self.tree = constraints

    @property
    def bounds(self) -> tuple[PyTree[Array], PyTree[Array]]:
        """
        Extracts a PyTree of lower bounds and a PyTree of upper bounds.
        Non-constraint nodes in the original PyTree are left unmodified.
        """
        from parax.filters import is_constraint

        def get_lower(node: Any) -> Any:
            if not is_constraint(node):
                return node
            return node.bounds[0]
        
        def get_upper(node: Any) -> Any:
            if not is_constraint(node):
                return node
            return node.bounds[1]
                
        lower = jax.tree_util.tree_map(get_lower, self.tree, is_leaf=is_constraint)
        upper = jax.tree_util.tree_map(get_upper, self.tree, is_leaf=is_constraint)
        
        return lower, upper

    @property
    def bijector(self) -> AbstractBijector:
        """
        Returns a `distreqx.TreeMap` bijector that applies each respective 
        leaf constraint's bijector.
        """
        from parax.filters import is_constraint
        from distreqx.bijectors import TreeMap

        def get_bijector(node: Any) -> Any:
            if not is_constraint(node):
                return node
            return node.bijector
        
        bijector = jax.tree_util.tree_map(get_bijector, self.tree, is_leaf=is_constraint)

        return TreeMap(bijector)
    

class CustomConstraint(AbstractConstraint):
    """
    An escape hatch for power users who need a specific distreqx bijector 
    mapping with predefined physical bounds.
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