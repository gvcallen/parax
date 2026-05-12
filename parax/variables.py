"""
Computable JAX arrays with metadata and constraints.

This module provides the core variable types for Parax. They act as 
array-like objects that can be directly injected into JAX computations 
or unwrapped prior to execution.
"""
from abc import abstractmethod
from typing import Any, Iterator, Callable, TypeGuard, Self
import dataclasses

import jax
import jax.numpy as jnp
from jaxtyping import Array, Inexact
import equinox as eqx

from parax.constraints import AbstractConstraint, RealLine, get_constraint_for_distribution
from parax.constant import AbstractConstant
from parax.unwrappable import AbstractUnwrappable, as_unwrapped
from parax.wrappable import AbstractWrappable
from parax.wrappers import as_frozen, as_frozen_or_static
from parax.annotated import AbstractAnnotated
from parax.bounded import AbstractBounded
from parax.constrainable import AbstractConstrainable
from parax.probabilistic import AbstractProbabilistic

from distreqx.distributions import AbstractDistribution

class AbstractVariable(AbstractUnwrappable[Array]):
    """
    The abstract interface for all model variables.

    Derive from this class and override `value` to implement
    custom variable unwrapping behaviour.

    All parameters in Parax, such as `parax.Param`,
    `parax.Constrained` etc., derive from this class.

    **Corner Case Note (Math & Dunders):** Because this class implements the 
    `__jax_array__` protocol and all standard math dunder methods, variables 
    can be used directly in JAX expressions *without* explicitly calling 
    `unwrap()`. However, applying any math operation (e.g., `var + 1`) instantly 
    evaluates the value and returns a standard `jax.Array`, stripping away 
    the metadata and constraint wrappers.
    """
    @property
    @abstractmethod
    def value(self) -> Array:
        """Returns the underlying, fully computed value of the variable."""
        pass    

    def unwrap(self) -> Array:
        return self.value
    
    def as_fixed(self) -> 'Fixed':
        return Fixed(self)

    @property
    def shape(self) -> tuple[int, ...]: return self.value.shape
    
    @property
    def size(self) -> int: return self.value.size
    
    @property
    def dtype(self) -> Any: return self.value.dtype

    # Experimental __jax_array__ protocol
    def __jax_array__(self) -> Array: return self.value

    # JAX Integration & Container Dunders
    def __getitem__(self, key: Any) -> Array: return self.value[key]
    def __len__(self) -> int: return len(self.value)
    def __iter__(self) -> Iterator[Any]: return iter(self.value)
    def __contains__(self, item: object) -> bool: return item in self.value

    # Forward Operators
    def __add__(self, other: Any) -> Array: return self.value + jnp.asarray(other)
    def __sub__(self, other: Any) -> Array: return self.value - jnp.asarray(other)
    def __mul__(self, other: Any) -> Array: return self.value * jnp.asarray(other)
    def __matmul__(self, other: Any) -> Array: return self.value @ jnp.asarray(other)
    def __truediv__(self, other: Any) -> Array: return self.value / jnp.asarray(other)
    def __floordiv__(self, other: Any) -> Array: return self.value // jnp.asarray(other)
    def __mod__(self, other: Any) -> Array: return self.value % jnp.asarray(other)
    def __divmod__(self, other: Any) -> tuple[Array, Array]: return divmod(self.value, jnp.asarray(other))
    def __pow__(self, other: Any) -> Array: return self.value ** jnp.asarray(other)

    # Reverse Operators
    def __radd__(self, other: Any) -> Array: return jnp.asarray(other) + self.value
    def __rsub__(self, other: Any) -> Array: return jnp.asarray(other) - self.value
    def __rmul__(self, other: Any) -> Array: return jnp.asarray(other) * self.value
    def __rmatmul__(self, other: Any) -> Array: return jnp.asarray(other) @ self.value
    def __rtruediv__(self, other: Any) -> Array: return jnp.asarray(other) / self.value
    def __rfloordiv__(self, other: Any) -> Array: return jnp.asarray(other) // self.value
    def __rmod__(self, other: Any) -> Array: return jnp.asarray(other) % self.value
    def __rdivmod__(self, other: Any) -> tuple[Array, Array]: return divmod(jnp.asarray(other), self.value)
    def __rpow__(self, other: Any) -> Array: return jnp.asarray(other) ** self.value

    # Unary Operators & Casting
    def __neg__(self) -> Array: return -self.value
    def __pos__(self) -> Array: return +self.value
    def __abs__(self) -> Array: return abs(self.value)
    def __invert__(self) -> Array: return ~self.value
    def __complex__(self) -> complex: return complex(self.value) # type: ignore
    def __int__(self) -> int: return int(self.value) # type: ignore
    def __float__(self) -> float: return float(self.value) # type: ignore
    def __index__(self) -> int: return self.value.__index__()
    def __round__(self, ndigits: int = 0) -> Array: return jnp.round(self.value, ndigits) 


Param = AbstractVariable | Inexact[Array, "..."]
"""
A type alias representing a JAX parameter.

This includes any Parax variables (like `Tagged`, `Constrained`, `Derived`) 
as well as standard JAX inexact arrays.
"""


def is_variable(x: Any) -> TypeGuard[AbstractVariable]:
    """
    Returns True if `x` is an instance of `parax.AbstractVariable`.
    """
    return isinstance(x, AbstractVariable)


def is_param(x: Any) -> bool:
    """
    Returns True if `x` is an instance of `parax.AbstractVariable`
    or returns True for `eqx.is_inexact_array`.
    """
    return isinstance(x, AbstractVariable) or eqx.is_inexact_array(x)


def as_param(value: Any) -> Any:
    """
    Returns `value` as a `parax.Param`, wrapping it if necessary.

    Args:
        value: An arbitrary value or array.

    Returns:
        The instantiated parameter.
    """    
    if is_param(value):
        return value
    return jnp.asarray(value, dtype=float)


class Real(AbstractVariable, AbstractWrappable[Array]):
    """
    A plane real variable.

    Useful as a placeholder, e.g. for frameworks that only want to allow
    `parax.AbstractVariable` instances to be trainable.

    Attributes:
        raw_value: The raw value used by optimizers and samplers.
    """
    raw_value: Param = eqx.field(converter=as_param)

    @property
    def value(self) -> Array:
        return jnp.asarray(self.raw_value)
    
    def wrap(self, value: Array) -> Self:
        return eqx.tree_at(lambda m: m.raw_value, self, value)
    
    
def as_variable(value: Any) -> Any:
    """
    Returns `value` as a `parax.AbstractVariable`, wrapping it if necessary.

    Args:
        value: An arbitrary value or array.

    Returns:
        The instantiated parameter.
    """    
    if is_variable(value):
        return value
    return Real(value)


class Tagged(AbstractVariable, AbstractAnnotated[dict], AbstractWrappable[Array]):
    """
    A variable with dictionary metadata.

    Represents a simple, trainable variable
    with a single underlying `raw_value` and metadata.

    Attributes:
        raw_value: The raw value used by optimizers and samplers.
        metadata: Additional arbitrary metadata.
    """
    raw_value: Param = eqx.field(converter=as_param)
    metadata: dict = eqx.field(default_factory=dict, static=True)

    @property
    def value(self) -> Array:
        return jnp.asarray(self.raw_value)
    
    def wrap(self, value: Array) -> Self:
        return eqx.tree_at(lambda m: m.raw_value, self, value)


class Fixed(AbstractVariable, AbstractConstant[Param]):
    """
    A fixed variable.
    
    Implements `AbstractConstant` for filtering during partitioning.
     
    Attributes:
        raw_value: The underlying variable that is being fixed.
    """
    raw_value: Param = eqx.field(converter=as_param)

    def __init__(self, raw_value: Param | None = None):
        """
        Args:
            raw_value: The underlying value to be fixed.
        """
        # Error checking
        if isinstance(raw_value, Fixed):
            raw_value = raw_value.raw_value
        self.raw_value = raw_value    

    def as_free(self) -> AbstractVariable:
        return self.raw_value

    @property
    def value(self) -> Array:
        value = self.raw_value
        if isinstance(self.raw_value, AbstractVariable):
            value = value.value
        return jax.lax.stop_gradient(value)
   

def as_fixed(value: Param) -> Fixed:
    """
    Returns `value` as a `parax.Fixed` variable, wrapping it if necessary.

    Args:
        value: An arbitrary variable or array-like object.

    Returns:
        A fixed version of the variable.
    """    
    if isinstance(value, Fixed):
        return value
    return Fixed(value)


class Derived(AbstractVariable):
    """
    A derived variable.
     
    The parameter's value is dynamically derived via an arbitrary callable.

    This is ideal for one-way transformations, projections, or normalizations 
    where a strict bijector (with an inverse) is not required or mathematically 
    possible (e.g., applying `jax.nn.softmax` to raw logits).

    Attributes:
        raw_value: The raw value used by optimizers and samplers.
        fn: The callable used to transform the raw value.
    """
    fn: Callable = eqx.field(converter=as_frozen_or_static)
    raw_value: Param = eqx.field(converter=as_param)

    @property
    def value(self) -> Array:
        """
        The derived value.
        
        Returns the raw state transformed by the derivation function.
        """
        return as_unwrapped(self.fn)(jnp.asarray(self.raw_value))
    

class Constrained(
    AbstractVariable,
    AbstractConstrainable[Array],
    AbstractWrappable[Array]
):
    """
    A constrained variable.
    
    The constraint is specified via a `parax.AbstractConstraint`.

    The constraint is automatically applied as a bijection mapping during 
    evaluation. Implements the `parax.bounded.AbstractBounded` interface
    for integration with bounded optimizers.

    Attributes:
        raw_value: The raw, unconstrained value on the real number line.
        constraint: The parameter constraint defining bounds and bijector mappings.
    """
    constraint: AbstractConstraint = eqx.field(converter=as_frozen)
    raw_value: Array = eqx.field(converter=jnp.asarray)

    def __init__(
        self,
        constraint: AbstractConstraint | None = None,
        value: Array | None = None,
        *,
        raw_value: Array | None = None,
    ):
        """
        Args:
            constraint: A Parax constraint. If None, defaults to `parax.RealLine` (unconstrained).
            value: The desired output (constrained) value. If provided, the internal 
                `raw_value` is computed dynamically via the constraint's inverse bijector. 
                Mutually exclusive with `raw_value`.
            raw_value: The unconstrained underlying value. Mutually exclusive with `value`.
        """
        # Error checking
        if value is None and raw_value is None:
            raise ValueError("Must provide either `value` or `raw_value`.")
        if value is not None and raw_value is not None:
            raise ValueError("Cannot provide both `value` and `raw_value`.")
        
        # Array standardization
        if raw_value is not None:
            raw_value = jnp.asarray(raw_value)
            shape = raw_value.shape
        else:
            value = jnp.asarray(value)
            shape = value.shape

        # Constraint and distribution standardization
        if constraint is None:
            constraint = RealLine(shape=shape)
        
        # Raw value standardization
        if value is not None:
            raw_value = constraint.bijector.inverse(value)
            if jnp.any(jnp.isnan(raw_value)):
                raise ValueError(f"Constraint {constraint} violated for variable with value `{value}` upon initialization")

        self.constraint = constraint
        self.raw_value = raw_value
       
    @property
    def value(self) -> Array:
        return as_unwrapped(self.constraint).bijector.forward(self.raw_value)
    
    @property
    def bounds(self) -> tuple[Array, Array]:
        constraint_bounds = as_unwrapped(self.constraint).bounds
        lower = jnp.broadcast_to(constraint_bounds[0], self.value.shape)
        upper = jnp.broadcast_to(constraint_bounds[1], self.value.shape)
        return lower, upper
    
    def constrain(self, constraint: AbstractConstraint) -> Self:
        return Constrained(constraint=constraint, value=self.value)    
    
    def wrap(self, value: Array) -> Self:
        new_raw = as_unwrapped(self.constraint).bijector.inverse(value)
        return eqx.tree_at(lambda x: x.raw_value, self, new_raw)
    

class Random(
    AbstractVariable,
    AbstractProbabilistic[Array],
    AbstractWrappable[Array]
):
    """
    A random variable with an optional constraint.
    
    The distribution is specified via a `distreqx.distributions.AbstractDistribution`.
    The constraint is specified via a `parax.constraint.AbstractConstraint`.

    The variable implements the `parax.probabilistic.AbstractProbabilistic` interface
    to integrate with stochastic samplers and other algorithms.

    Attributes:
        distribution: The probability distribution of `raw_value`.
        constraint: The constraint that defines the support of `distribution`.
                    Can be None, in which case this function will attempt
                    to automatically infer the constraint from the distribution's
                    using `parax.constraints.get_constraint_for_distribution`.
        raw_value: The raw un-probabilistic value on the real number line. 
                   Can be None, in which case the mean of the distribution is used.
                   If the mean is not supported, an exception is thrown.
    """
    distribution: AbstractDistribution = eqx.field(converter=as_frozen)
    constraint: AbstractConstraint = eqx.field(converter=as_frozen)
    raw_value: Array = eqx.field(converter=jnp.asarray)

    def __init__(
        self,
        distribution: AbstractDistribution,
        constraint: AbstractConstraint | None = None,
        value: Array | None = None,
        *,
        raw_value: Array | None = None,
    ):
        """
        Args:
            raw_value: The un-probabilistic raw value.
            distribution: The probability distribution.
            constraint: The distribution's constraint. If `None`, then a 
        """
        if value is not None and raw_value is not None:
            raise ValueError("Cannot provide both `value` and `raw_value`.")

        # Derive physical value if both are missing
        if value is None and raw_value is None:
            try:
                value = jnp.array(distribution.mean())
            except Exception:
                raise ValueError(
                    "`value` or `raw_value` must be provided if the "
                    "distribution does not support `mean()`."
                )
            
        # Constraint resolution
        if constraint is None:
            constraint = get_constraint_for_distribution(distribution)

        # Calculate unconstrained raw_value
        if value is not None:
            raw_value = constraint.bijector.inverse(jnp.asarray(value))
            if jnp.any(jnp.isnan(raw_value)):
                raise ValueError(f"Constraint {constraint} violated for variable with value `{value}` upon initialization")
        else:
            raw_value = jnp.asarray(raw_value)

        self.distribution = distribution
        self.constraint = constraint
        self.raw_value = raw_value

    @property
    def value(self) -> Array:
        return as_unwrapped(self.constraint).bijector.forward(self.raw_value)
    
    @property
    def bounds(self) -> tuple[Array, Array]:
        constraint_bounds = as_unwrapped(self.constraint).bounds
        lower = jnp.broadcast_to(constraint_bounds[0], self.value.shape)
        upper = jnp.broadcast_to(constraint_bounds[1], self.value.shape)
        return lower, upper
    
    def constrain(self, constraint: AbstractConstraint) -> Self:
        return Random(distribution=self.distribution, constraint=constraint, value=self.value)
    
    def wrap(self, value: Array) -> Self:
        new_raw = as_unwrapped(self.constraint).bijector.inverse(value)
        return eqx.tree_at(lambda x: x.raw_value, self, new_raw)

def tagged(
    raw_value: Param = dataclasses.MISSING,
    *,
    metadata: dict | None = None,
) -> Any:
    """
    Specifies a dataclass field for a Parax `Tagged` variable.

    Args:
        raw_value: The default raw value. If omitted, this field becomes required 
            by the user during instantiation.
        metadata: Additional static metadata to store.
        
    Returns:
        An `equinox.field` properly configured for the field type.
    """
    if metadata is None: metadata = {}

    def converter(x: Any) -> AbstractVariable:
        return Tagged(raw_value=x, metadata=metadata)

    field_kwargs = {"converter": converter}
    if raw_value is not dataclasses.MISSING:
        field_kwargs["default"] = raw_value

    return eqx.field(**field_kwargs)


def derived(
    fn: Callable = lambda x: x,
    raw_value: Param = dataclasses.MISSING,
) -> Any:
    """
    Specifies a dataclass field for a Parax `Derived` variable.

    Args:
        fn: The callable used to transform the raw value.
        raw_value: The default raw value. If omitted, this field becomes required.
        
    Returns:
        An `equinox.field` properly configured for the field type.
    """
    def converter(x: Any) -> AbstractVariable:
        return Derived(fn=fn, raw_value=x)

    field_kwargs = {"converter": converter}
    if raw_value is not dataclasses.MISSING:
        field_kwargs["default"] = raw_value
        
    return eqx.field(**field_kwargs)


def constrained(
    constraint: AbstractConstraint | None = None,
    value: Array = dataclasses.MISSING,
) -> Any:
    """
    Specifies a dataclass field for a Parax `parax.Constrained` variable.

    Args:
        constraint: The abstract constraint defining base bounds and mappings.
        value: The default constrained value. If omitted, this field becomes required.
        
    Returns:
        An `equinox.field` properly configured for the field type.
    """
    def converter(x: Any) -> AbstractVariable:
        return Constrained(constraint=constraint, value=x)

    field_kwargs = {"converter": converter}
    if value is not dataclasses.MISSING:
        field_kwargs["default"] = value
        
    return eqx.field(**field_kwargs)


def random(
    distribution: AbstractDistribution | None = None,
    constraint: AbstractConstraint | None = None,
    value: Array = dataclasses.MISSING,
) -> Any:
    """
    Specifies a dataclass field for a Parax `parax.Random` variable.

    Args:
        distribution: The distribution defining base bounds and mappings.
        value: The default value. If omitted, this field becomes required.
        
    Returns:
        An `equinox.field` properly configured for the field type.
    """
    def converter(x: Any) -> AbstractVariable:
        return Random(distribution=distribution, constraint=constraint, value=x)

    field_kwargs = {"converter": converter}
    if value is not dataclasses.MISSING:
        field_kwargs["default"] = value
        
    return eqx.field(**field_kwargs)