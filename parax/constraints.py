from typing import Union, TypeGuard, Any

import jax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from distreqx.bijectors import (
    AbstractBijector, 
    Softplus, 
    Sigmoid, 
    Chain, 
    Shift, 
    ScalarAffine,
    Identity,
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


class Unconstrained(AbstractConstraint):
    """Represents a value that can span the entire real number line."""
    shape: Any = eqx.field(static=True)
    dtype: Any = eqx.field(default=None, static=True)

    def __init__(self, shape: Any = (), dtype: Any = None):
        self.shape = shape
        self.dtype = dtype
    
    @property
    def bounds(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        return (
            jnp.full(self.shape, -jnp.inf, dtype=self.dtype),
            jnp.full(self.shape, jnp.inf, dtype=self.dtype),
        )

    @property
    def bijector(self) -> AbstractBijector:
        return Identity()


class GreaterThan(AbstractConstraint):
    """
    Represents a value strictly greater than a lower bound.
    """
    lower: Float[Array, "..."]

    def __init__(self, lower: Union[float, Array]):
        self.lower = jnp.asarray(lower)

    @property
    def bounds(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        return (self.lower, jnp.full_like(self.lower, jnp.inf))

    @property
    def bijector(self) -> AbstractBijector:
        if jnp.allclose(self.lower, 0.0):
            return Softplus()
        return Chain([Shift(self.lower), Softplus()])


class LessThan(AbstractConstraint):
    """
    Represents a value strictly less than an upper bound.
    """
    upper: Float[Array, "..."]

    def __init__(self, upper: Union[float, Array]):
        self.upper = jnp.asarray(upper)

    @property
    def bounds(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        return (jnp.full_like(self.upper, -jnp.inf), self.upper)

    @property
    def bijector(self) -> AbstractBijector:
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
    lower: Float[Array, "..."]
    upper: Float[Array, "..."]

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