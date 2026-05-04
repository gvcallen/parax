"""
Computable JAX arrays with metadata and constraints.

This module provides the core variable types for Parax. They act as 
array-like objects that can be directly injected into JAX computations 
or unwrapped prior to execution.
"""
from abc import abstractmethod
from typing import Any, Iterator, Callable
import dataclasses

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Inexact, ArrayLike
import equinox as eqx

from parax.constraints import AbstractConstraint, RealLine
from parax.unwrappables import AbstractUnwrappable, Frozen
from parax.bounded import AbstractBounded
from parax.probabilistic import AbstractProbabilistic
from parax.constant import AbstractConstant
from parax.tagged import AbstractTagged

from distreqx.distributions import AbstractDistribution, ImproperUniform

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


ParamLike = AbstractVariable | Inexact[Array, "..."]
"""
A type alias representing any parameter-like object.

This includes any Parax variables (like `Param`, `Constrained`, `Physical`) 
as well as standard JAX inexact arrays.
"""

def _as_param_like(x):
    if isinstance(x, AbstractVariable):
        return x
    return jnp.asarray(x)


class Param(AbstractVariable, AbstractTagged):
    """
    A canonical parameter with metadata.

    Represents a simple, trainable variable
    with a single underlying `raw_value` and metadata.

    Attributes:
        raw_value: The raw value used by optimizers and samplers.
        metadata: Additional arbitrary metadata.
    """
    raw_value: ParamLike = eqx.field(converter=_as_param_like)
    metadata: dict = eqx.field(default_factory=dict, static=True, kw_only=True)

    @property
    def value(self) -> Array:
        return self.raw_value
    

class Fixed(AbstractVariable, AbstractConstant[AbstractVariable]):
    """
    A fixed variable.
    
    Implements `AbstractConstant` for structural filtering during partitioning.
     
    **Corner Case Note:** This class implements `__getattr__` to forward all 
    unrecognized attribute lookups to the underlying wrapped variable. This means 
    a `Fixed(Constrained(...))` object will still safely expose `.constraint`, 
    `.bounds`, and `.metadata` to the user as if it weren't wrapped at all.

    Attributes:
        raw_value: The underlying variable that is being fixed.
    """
    raw_value: ParamLike

    def __post_init__(self):
        if isinstance(self.raw_value, Fixed):
            self.raw_value = self.raw_value.raw_value

    def as_free(self) -> AbstractVariable:
        return self.raw_value

    @property
    def value(self) -> Array:
        value = self.raw_value
        if isinstance(self.raw_value, AbstractVariable):
            value = value.value
        return jax.lax.stop_gradient(value)
   

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
    raw_value: ParamLike = eqx.field(converter=_as_param_like)
    fn: Callable = eqx.field(static=True)

    @property
    def value(self) -> Array:
        """
        The derived value.
        
        Returns the raw state transformed by the derivation function.
        """
        return self.fn(jnp.asarray(self.raw_value))


class Constrained(AbstractVariable, AbstractBounded[Array]):
    """
    A constrained variable.
    
    The constraint is specified via a `parax.AbstractConstraint`.

    The constraint is automatically applied as a bijection mapping during 
    evaluation. Implements the `parax.bounded.AbstractBounded` interface
    for integration with bounded optimizers.

    Attributes:
        raw_value: The raw, unconstrained value mapping to the real number line.
        constraint: The parameter constraint defining bounds and bijector mappings.
    """
    raw_value: ParamLike = eqx.field(converter=_as_param_like)
    constraint: AbstractConstraint = eqx.field(converter=Frozen)

    def __init__(
        self,
        value: Array | None = None,
        constraint: AbstractConstraint | None = None,
        *,
        raw_value: ParamLike | None = None,
    ):
        """
        Args:
            value: The desired output (constrained) value. If provided, the internal 
                `raw_value` is computed dynamically via the constraint's inverse bijector. 
                Mutually exclusive with `raw_value`.
            constraint: A Parax constraint. If None, defaults to `parax.RealLine` (unconstrained).
            raw_value: The unconstrained optimizer-space value. Mutually exclusive with `value`.
        """
        # Error checking
        if value is None and raw_value is None:
            raise ValueError("Must provide either `value` or `raw_value`.")
        if value is not None and raw_value is not None:
            raise ValueError("Cannot provide both `value` and `raw_value`.")
        
        # Array standardization
        if raw_value is not None:
            raw_value = _as_param_like(raw_value)
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

        self.constraint = constraint
        self.raw_value = raw_value
       
    @property
    def base(self) -> Array:
        from parax.converters import as_free
        return as_free(self.constraint).bijector.forward(self.raw_value)
    
    @property
    def bounds(self) -> tuple[Array, Array]:
        from parax.converters import as_free
        constraint_bounds = as_free(self.constraint).bounds
        lower = jnp.broadcast_to(constraint_bounds[0], self.value.shape)
        upper = jnp.broadcast_to(constraint_bounds[1], self.value.shape)
        return lower, upper
    
    def update(self, base: Array) -> Array:
        from parax.converters import as_free
        new_raw = as_free(self.constraint).bijector.inverse(base)
        return eqx.tree_at(lambda x: x.raw_value, self, new_raw)
    
    @property
    def value(self) -> Array:
        return self.base
    

class Random(AbstractVariable, AbstractProbabilistic[Array]):
    """
    A random variable.
    
    The distribution is specified via a `distreqx.distributions.AbstractDistribution`.

    The variable implements the `parax.probabilistic.AbstractProbabilistic` interface
    to integrate with stochastic samplers and other algorithms.

    Attributes:
        raw_value: The raw un-probabilistic value.
        distribution: The probability distribution of `raw_value`.
    """
    raw_value: ParamLike = eqx.field(converter=_as_param_like)
    distribution: AbstractDistribution = eqx.field(converter=Frozen)

    def __init__(
        self,
        raw_value: ParamLike = None,
        distribution: AbstractDistribution | None = None,
    ):
        """
        Args:
            raw_value: The un-probabilistic raw value.
            distribution: The probability distribution. If None, defaults to `distreqx.distributions.ImproperUniform.
        """
        # Array standardization
        raw_value = _as_param_like(raw_value)
        shape = raw_value.shape
        
        # Distribution standardization
        if distribution is None:
            distribution = ImproperUniform(shape=shape)
        
        self.distribution = distribution
        self.raw_value = raw_value

    @property
    def value(self) -> Array:
        return self.raw_value


class Physical(AbstractVariable):
    """
    A physical variable with a scale or unit.
    
    Multiplies the underlying `parax.ParamLike` by an `ArrayLike` float or unit.

    Useful for scientific modeling. Applies a linear physical scale 
    (e.g., units or preconditioning) as the final evaluation step on top of 
    an underlying variable (such as a `Constrained` or `Param` instance).

    Attributes:
        variable: The underlying parameter or array being scaled.
        scale: Linear preconditioning factor or physical unit (e.g., `unxt.Quantity`).
    """
    raw_value: ParamLike = eqx.field(converter=_as_param_like)
    scale: ArrayLike = eqx.field(converter=Fixed)

    def __init__(
        self,
        raw_value: ParamLike,
        scale: Any = 1.0,
    ):
        """
        Args:
            raw_value: The underlying base variable (e.g., `Constrained`, `Param`, or Array).
            scale: Linear multiplier or unit string. If a string is provided, 
                it is converted automatically using `unxt.unit()`.
        """
        # Scale standardization
        if isinstance(scale, (float, int, Array)):
            scale = jnp.asarray(scale, dtype=float)
        elif isinstance(scale, str):
            try:
                import unxt
            except ImportError as e:
                raise Exception("Using units as scales requires the `unxt` package")
            scale = jnp.array(1.0) * unxt.unit(scale)

        self.raw_value = raw_value
        self.scale = scale

    @property
    def value(self) -> Array:
        """Returns the physically scaled value."""
        from parax.converters import as_free
        return as_free(self.scale) * self.raw_value


def map_variables(f: Callable[[AbstractVariable], Any], pytree: PyTree) -> PyTree:
    """
    Maps a callable over all `parax.AbstractVariable` nodes in a PyTree.

    Safely bypasses standard arrays or PyTree structural nodes.

    Args:
        f: The callable mapping function.
        pytree: Any JAX PyTree containing `parax.AbstractVariable` leaves.

    Returns:
        A new PyTree with the variables mapped. 
    """
    from parax.filters import is_variable
    return jax.tree.map(lambda x: f(x) if is_variable(x) else x, pytree, is_leaf=is_variable)


def map_variables_with_path(f: Callable[[Any, AbstractVariable], Any], pytree: PyTree) -> PyTree:
    """
    Maps a callable (which takes a key path) over all `parax.AbstractVariable` 
    nodes in a PyTree.

    Args:
        f: The callable mapping function taking `(path, variable)`.
        pytree: Any JAX PyTree containing `parax.AbstractVariable` leaves.

    Returns:
        A new PyTree with the variables mapped.
    """
    from parax.filters import is_variable
    return jax.tree.map_with_path(lambda p, x: f(p, x) if is_variable(x) else x, pytree, is_leaf=is_variable)


def param(
    default: ParamLike = dataclasses.MISSING,
    metadata: dict | None = None,
) -> Any:
    """
    Specifies a dataclass field for a standard Parax `Param`.

    Args:
        default: The default value. If omitted, this field becomes required 
            by the user during instantiation.
        metadata: Additional static metadata to store.
        
    Returns:
        An `equinox.field` properly configured for the field type.
    """
    if metadata is None: metadata = {}

    def converter(x: Any) -> AbstractVariable:
        if isinstance(x, AbstractVariable):
            return x
        
        return Param(x, metadata=metadata)

    field_kwargs = {"converter": converter}
    if default is not dataclasses.MISSING:
        field_kwargs["default"] = default

    return eqx.field(**field_kwargs)


def derived(
    fn: Callable,
    default: ParamLike = dataclasses.MISSING,
) -> Any:
    """
    Specifies a dataclass field for a Parax `Derived` variable.

    Args:
        fn: The callable used to transform the raw value.
        default: The default raw value. If omitted, this field becomes required.
        
    Returns:
        An `equinox.field` properly configured for the field type.
    """
    def converter(x: Any) -> AbstractVariable:
        if isinstance(x, AbstractVariable):
            return x
        return Derived(x, fn=fn)

    field_kwargs = {"converter": converter}
    if default is not dataclasses.MISSING:
        field_kwargs["default"] = default
        
    return eqx.field(**field_kwargs)


def constrained(
    default: ParamLike = dataclasses.MISSING,
    constraint: AbstractConstraint | None = None,
) -> Any:
    """
    Specifies a dataclass field for a Parax `parax.Constrained` variable.

    Args:
        default: The default constrained value. If omitted, this field becomes required.
        constraint: The abstract constraint defining base bounds and mappings.
        
    Returns:
        An `equinox.field` properly configured for the field type.
    """
    def converter(x: Any) -> AbstractVariable:
        if isinstance(x, AbstractVariable):
            return x
        return Constrained(x, constraint=constraint)

    field_kwargs = {"converter": converter}
    if default is not dataclasses.MISSING:
        field_kwargs["default"] = default
        
    return eqx.field(**field_kwargs)


def random(
    default: ParamLike = dataclasses.MISSING,
    distribution: AbstractDistribution | None = None,
) -> Any:
    """
    Specifies a dataclass field for a Parax `parax.Random` variable.

    Args:
        default: The default value. If omitted, this field becomes required.
        constraint: The abstract constraint defining base bounds and mappings.
        
    Returns:
        An `equinox.field` properly configured for the field type.
    """
    def converter(x: Any) -> AbstractVariable:
        if isinstance(x, AbstractVariable):
            return x
        return Random(x, distribution=distribution)

    field_kwargs = {"converter": converter}
    if default is not dataclasses.MISSING:
        field_kwargs["default"] = default
        
    return eqx.field(**field_kwargs)


def physical(
    default: ParamLike = dataclasses.MISSING,
    scale: Any = 1.0,
) -> Any:
    """
    Specifies a dataclass field for a Parax `Physical` parameter wrapper.

    Args:
        default: The default underlying variable (e.g. `Constrained`, `Param`, or Array). 
            If omitted, this field becomes required by the user during instantiation.
        scale: Linear preconditioning factor or unit string (e.g., "mm").
        
    Returns:
        An `equinox.field` properly configured for the field type.
    """
    def converter(x: Any) -> AbstractVariable:
        # Avoid double-wrapping if the user passes an already instantiated Physical
        if isinstance(x, Physical):
            return x
        return Physical(x, scale=scale)

    field_kwargs = {"converter": converter}
    if default is not dataclasses.MISSING:
        field_kwargs["default"] = default
        
    return eqx.field(**field_kwargs)