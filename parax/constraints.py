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
    The base class for all physical constraints in parax.
    
    A constraint acts as a bridge between hard physical boundaries (used by 
    bounded optimizers) and topological mappings (used by unconstrained ML optimizers).

    Constraints may be used directly on arrays or on PyTrees,
    depending on the type of constraint.
    """
    bounds: eqx.AbstractVar[tuple[PyTree, PyTree]]
    bijector: eqx.AbstractVar[AbstractBijector]


class RealLine(AbstractConstraint):
    """Represents a value that can span the entire real number line."""
    shape: Any = eqx.field(static=True)

    def __init__(self, shape: Any = ()):
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
    """
    lower: jnp.ndarray
    
    def __init__(self, lower: Union[float, Array]):
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
    """
    upper: jnp.ndarray

    def __init__(self, upper: Union[float, Array]):
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

        return Chain([
            Shift(self.upper),
            ScalarAffine(shift=jnp.array(0.0), scale=jnp.array(-1.0)),
            Softplus(),
            ScalarAffine(shift=jnp.array(0.0), scale=jnp.array(-1.0)),
        ])


class Interval(AbstractConstraint):
    """
    Represents a value strictly bounded between a lower and upper value.
    """
    lower: jnp.ndarray
    upper: jnp.ndarray

    def __init__(self, lower: Union[float, Array], upper: Union[float, Array]):
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
    """
    Convenience constraint for values that must be strictly positive (> 0).
    """
    def __init__(self, shape: Any = (), dtype: Any = None):
        super().__init__(lower=jnp.zeros(shape, dtype=dtype))


class Negative(LessThan):
    """
    Convenience constraint for values that must be strictly negative (< 0).
    """
    def __init__(self, shape: Any = (), dtype: Any = None):
        super().__init__(upper=jnp.zeros(shape, dtype=dtype))


class TransformedConstraint(AbstractConstraint):
    """
    A constraint modified by an arbitrary bijector.
    
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
            bijector: The distreqx bijector to apply on top of the base constraint.
        """
        self.base_constraint = constraint
        self.custom_bijector = bijector

    @property
    def bounds(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        """
        Calculates the new topological bounds by passing the base extrema 
        through the custom bijector.
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
    Represents a PyTree of constraints over a PyTree of inputs.
    """
    constraints: PyTree[AbstractConstraint]

    def __init__(
        self, 
        constraints: PyTree[AbstractConstraint],
    ):
        from parax.filters import is_constraint

        leaves = jax.tree.leaves(constraints, is_leaf=is_constraint)
        if not leaves:
            raise ValueError("The pytree of constraints cannot be empty.")

        self.constraints = constraints

    @property
    def bounds(self) -> tuple[PyTree[Array], PyTree[Array]]:
        from parax.filters import is_constraint

        def get_lower(node: Any) -> Any:
            if not is_constraint(node):
                return node
            return node.bounds[0]
        
        def get_upper(node: Any) -> Any:
            if not is_constraint(node):
                return node
            return node.bounds[1]
                
        lower = jax.tree_util.tree_map(get_lower, self.constraints, is_leaf=is_constraint)
        upper = jax.tree_util.tree_map(get_upper, self.constraints, is_leaf=is_constraint)
        
        return lower, upper

    @property
    def bijector(self) -> AbstractBijector:
        from parax.filters import is_constraint
        from distreqx.bijectors import TreeMap

        def get_bijector(node: Any) -> Any:
            if not is_constraint(node):
                return node
            return node.bijector
        
        bijector = jax.tree_util.tree_map(get_bijector, self.constraints, is_leaf=is_constraint)

        return TreeMap(bijector)
    


class CustomConstraint(AbstractConstraint):
    """
    An escape hatch for power users who need a specific distreqx bijector.
    """
    _custom_bijector: AbstractBijector
    _custom_bounds: tuple[Array, Array]

    def __init__(
        self, 
        bijector: AbstractBijector, 
        bounds: tuple[Array, Array] = (jnp.array(-jnp.inf), jnp.array(jnp.inf))
    ):
        self._custom_bijector = bijector
        self._custom_bounds = tuple(jnp.asarray(b) for b in bounds)

    @property
    def bounds(self) -> tuple[Array, Array]:
        return self._custom_bounds

    @property
    def bijector(self) -> AbstractBijector:
        return self._custom_bijector