"""
Computable JAX arrays with metadata.
"""
from abc import abstractmethod
from typing import Any, Iterator, Callable
import dataclasses

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, ArrayLike
import equinox as eqx

from parax.constraints import AbstractConstraint, RealLine
from parax.unwrappables import AbstractUnwrappable, as_frozen
from parax.constant import AbstractConstant
from parax.metadata import AbstractHasMetadata

class AbstractVariable(AbstractUnwrappable[Array]):
    """
    The abstract interface for all model variables.

    Derive from this class and override `value` to implement
    custom variable unwrapping behaviour.

    All parameters in Parax, such as `parax.Param`,
    `parax.Constrained` etc., derive from this class.

    By inheriting from this class, your class becomes
    a `parax.AbstractUnwrappable`, and automatically has
    relevant dunder/array-like methods implemented.
    """
    @property
    @abstractmethod
    def value(self) -> Array:
        """Returns the underlying value of the variable."""
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


class AbstractConstrained(AbstractVariable, strict=True):
    """
    The abstract interface for a constrained variable.

    Used as a type check for `parax.is_constrained`.
    """
    #: The underlying constraint
    constraint: eqx.AbstractVar[AbstractConstraint]

    #: The value in constrained "base" space i.e. after the constraint is applied
    #: but before any additional transformations to the final space.
    #: Useful for bounded optimizers.
    base_value: eqx.AbstractVar[Array]

    #: Returns the final output value from a given base value.
    @abstractmethod
    def value_from_base(self, base_value: Array) -> Array:
        pass

    @property
    def value(self) -> Array:
        return self.value_from_base(self.base_value)


class Param(AbstractVariable, AbstractHasMetadata):
    """
    A simple parameter with metadata.
    
    Represents an unconstrained, trainable parameter
    with a single underlying `raw_value`.
    """
    #: The raw value used by optimizers and samplers.
    raw_value: Array = eqx.field(converter=jnp.asarray)

    #: Additional arbitrary metadata.
    metadata: dict = eqx.field(default_factory=dict, static=True, kw_only=True)

    @property
    def value(self) -> Array:
        return self.raw_value
       

class Derived(AbstractVariable, AbstractHasMetadata):
    """
    A parameter whose value is dynamically derived from its raw state 
    via an arbitrary callable.

    This is ideal for one-way transformations, projections, or normalizations 
    where a strict bijector (with an inverse) is not required or mathematically 
    possible (e.g., `jax.nn.softmax` for probabilities).
    """
    #: The raw value used by optimizers and samplers.
    raw_value: Array = eqx.field(converter=jnp.asarray)

    #: The callable used to transform the raw value.
    fn: Callable = eqx.field(static=True)
    
    #: Additional arbitrary metadata.
    metadata: dict = eqx.field(default_factory=dict, static=True, kw_only=True)

    @property
    def value(self) -> Array:
        """
        The derived value.
        
        Returns the raw state transformed by the derivation function.
        """
        return self.fn(self.raw_value)


class Constrained(AbstractConstrained, AbstractHasMetadata):
    """
    A parameter constrained by an arbitrary `parax.AbstractConstraint` constraint.

    The constraint is automatically applied as a bijection during unwrapping.
    However, since each constraint implements a `bounds` property,
    this property can easily be inspected for use in a bounded optimization.
    """
    #: The raw value used by optimizers and samplers.
    raw_value: Array = eqx.field(converter=jnp.asarray)

    #: The parameter constraint in base space. Useful to define conditioning and bounds for optimization.
    constraint: AbstractConstraint = eqx.field(converter=as_frozen)

    #: Additional arbitrary metadata.
    metadata: dict = eqx.field(default_factory=dict, static=True, kw_only=True)

    def __init__(
        self,
        value: Array | None = None,
        constraint: AbstractConstraint | None = None,
        *,
        raw_value: Array | None = None,
        metadata: dict | None = None,
    ):
        """
        Args:
            value: The output (constrained) array value. Mutually exclusive with `raw_value`.
            constraint: A parax constraint defining base bounds and mappings.
                        If None is passed, an `Unconstrained` constraint is created internally.
            raw_value: The raw (unconstrained) value. Mutually exclusive with `value`.
            metadata: Additional static metadata.        
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

        self.constraint = constraint
        self.raw_value = raw_value
        self.metadata = metadata if metadata is not None else {}
       
    @property
    def base_value(self) -> Array:
        return self.constraint.bijector.forward(self.raw_value)
    
    def value_from_base(self, base_value: Array):
        return base_value
    

class Physical(AbstractConstrained, AbstractHasMetadata):
    """
    A physically scaled and constrained parameter.

    Useful for scientific modeling.

    Combines an optimizer-friendly unbounded raw space with constraints 
    and a linear physical scale (e.g., units or preconditioning).
    """
    #: The raw value used by optimizers and samplers.
    raw_value: jax.Array = eqx.field(converter=jnp.asarray)
    
    #: Linear preconditioning factor or physical unit (e.g., unxt.Quantity).
    scale: Any = eqx.field(converter=as_frozen)
    
    #: The parameter constraint in base space.
    constraint: AbstractConstraint = eqx.field(converter=as_frozen)
    
    #: Additional arbitrary metadata.
    metadata: dict = eqx.field(default_factory=dict, static=True, kw_only=True)

    def __init__(
        self,
        base_value: Any | None = None,
        scale: Any = 1.0,
        *,
        raw_value: Any | None = None,
        constraint: AbstractConstraint | None = None,
        metadata: dict | None = None,
    ):
        """
        Args:
            base_value: The base array value. Mutually exclusive with `raw_value`.
            scale: Linear multiplier or unit string. If a string, is converted using `unxt.unit()`.
            raw_value: The unconstrained value. Mutually exclusive with `base_value`.
            constraint: A parax constraint defining base bounds and mappings.
            metadata: Additional static metadata.
        """
        if base_value is None and raw_value is None:
            raise ValueError("Must provide either `base_value` or `raw_value`.")
        if base_value is not None and raw_value is not None:
            raise ValueError("Cannot provide both `base_value` and `raw_value`.")
        
        # Scale standardization
        if isinstance(scale, (float, int, Array)):
            scale = jnp.asarray(scale, dtype=float)
        elif isinstance(scale, str):
            try:
                import unxt
            except ImportError as e:
                raise Exception("Using units as scales requires the `unxt` package")
            scale = jnp.array(1.0) * unxt.unit(scale)

        # Array standardization
        if raw_value is not None:
            raw_value = jnp.asarray(raw_value, dtype=float)
            shape = raw_value.shape
        else:
            base_value = jnp.asarray(base_value, dtype=float)
            shape = base_value.shape
        
        # Constraint standardization
        if constraint is None:
            constraint = RealLine(shape=shape)
        
        # Raw value standardization
        if base_value is not None:
            raw_value = constraint.bijector.inverse(base_value)

        self.raw_value = raw_value
        self.scale = scale
        self.constraint = constraint
        self.metadata = metadata if metadata is not None else {}

    @property
    def base_value(self) -> Array:
        """
        The value in base space.
        Returns the constraint's bijector applied to the raw value.
        """
        return self.constraint.bijector.forward(self.raw_value)
    
    def value_from_base(self, base_value: Array) -> Array:
        return base_value * self.scale


class Fixed(AbstractVariable, AbstractConstant[AbstractVariable]):
    """
    A wrapper to fix another variable.
    
    Wraps another variable while forwarding all attributes.
     
    Can be used either as a tag or to apply stopping gradients.
    """
    #: The underlying value that is being fixed.
    variable: AbstractVariable | ArrayLike

    def __init__(self, variable: AbstractVariable | ArrayLike):
        if isinstance(variable, Fixed):
            variable = variable.variable
        self.variable = variable

    def as_free(self) -> AbstractVariable:
        return self.variable

    @property
    def value(self) -> Array:
        value = self.variable
        if isinstance(self.variable, AbstractVariable):
            value = value.value
        return jax.lax.stop_gradient(value)
    
    def __getattr__(self, name):
        if hasattr(self.variable, name):
            return getattr(self.variable, name)
        return super().__getattribute__(name)
    

def map_variables(f: Callable[[AbstractVariable], Any], pytree: PyTree) -> PyTree:
    """
    Maps all `parax.AbstractVariable` variables in a PyTree using a callable.

    Args:
        f: The callable.
        pytree: Any JAX PyTree containing `parax.AbstractVariable` leaves.

    Returns:
        A PyTree with `parax.AbstractVariable` mapped using `f`. 
    """
    from parax.filters import is_variable
    return jax.tree.map(lambda x: f(x) if is_variable(x) else x, pytree, is_leaf=is_variable)


def map_variables_with_path(f: Callable[[Any, AbstractVariable], Any], pytree: PyTree) -> PyTree:
    """
    Maps all `parax.AbstractVariable` variables in a PyTree using a callable that takes a path.

    Args:
        f: The callable.
        pytree: Any JAX PyTree containing `parax.AbstractVariable` leaves.

    Returns:
        A PyTree with `parax.AbstractVariable` mapped using `f`. 
    """
    from parax.filters import is_variable
    return jax.tree.map_with_path(lambda p, x: f(p, x) if is_variable(x) else x, pytree, is_leaf=is_variable)


# ==========================================
# Parameter Dataclass Field Helpers
# ==========================================

def param(
    raw_value: Array = dataclasses.MISSING,
    metadata: dict | None = None,
) -> Any:
    """
    Specifies a field for a standard Parax `Param` within a dataclass.

    Args:
        raw_value: The default raw value. If omitted, this field becomes required by the user.
        metadata: Additional static metadata to store in the parameter.
    """
    if metadata is None: metadata = {}

    def converter(x: Any) -> AbstractVariable:
        if isinstance(x, AbstractVariable):
            return x
        
        return Param(raw_value=x, metadata=metadata)

    field_kwargs = {"converter": converter}
    if raw_value is not dataclasses.MISSING:
        field_kwargs["default"] = raw_value

    return eqx.field(**field_kwargs)


def derived(
    fn: Callable,
    raw_value: Array = dataclasses.MISSING,
    metadata: dict | None = None,
) -> Any:
    """
    Specifies a field for a Parax `Derived` variable within a dataclass.

    Args:
        fn: The callable used to transform the raw value.
        raw_value: The default raw value. If omitted, this field becomes required.
        metadata: Additional static metadata to store in the parameter.
    """
    if metadata is None: metadata = {}

    def converter(x: Any) -> AbstractVariable:
        if isinstance(x, AbstractVariable):
            return x
        return Derived(raw_value=x, fn=fn, metadata=metadata)

    field_kwargs = {"converter": converter}
    if raw_value is not dataclasses.MISSING:
        field_kwargs["default"] = raw_value
        
    return eqx.field(**field_kwargs)


def constrained(
    default_value: Array = dataclasses.MISSING,
    *,
    constraint: AbstractConstraint | None = None,
    metadata: dict | None = None,
) -> Any:
    """
    Specifies a field for a Parax `Constrained` parameter within a dataclass.

    Args:
        default_value: The default constrained value. If omitted, this field becomes required.
        constraint: The abstract constraint defining base bounds and mappings.
        metadata: Additional static metadata to store in the parameter.
    """
    if metadata is None: metadata = {}

    def converter(x: Any) -> AbstractVariable:
        if isinstance(x, AbstractVariable):
            return x
        return Constrained(value=x, constraint=constraint, metadata=metadata)

    field_kwargs = {"converter": converter}
    if default_value is not dataclasses.MISSING:
        field_kwargs["default"] = default_value
        
    return eqx.field(**field_kwargs)


def physical(
    default_base: Any = dataclasses.MISSING,
    scale: Any = 1.0,
    *,
    constraint: AbstractConstraint | None = None,
    metadata: dict | None = None,
) -> Any:
    """
    Specifies a field for a Parax `Physical` parameter within a dataclass.

    Args:
        default_base: The default base array value. If omitted, this field becomes required.
        scale: Linear preconditioning factor or unit string (e.g., "mm").
        constraint: A parax constraint defining base bounds and mappings.
        metadata: Additional static metadata to store in the parameter.
    """
    if metadata is None: metadata = {}

    def converter(x: Any) -> AbstractVariable:
        if isinstance(x, AbstractVariable):
            return x
        return Physical(base_value=x, scale=scale, constraint=constraint, metadata=metadata)

    field_kwargs = {"converter": converter}
    if default_base is not dataclasses.MISSING:
        field_kwargs["default"] = default_base
        
    return eqx.field(**field_kwargs)


def as_param(value: Any) -> Param:
    """
    Returns `value` as a `parax.Param` by creating one if needed.

    Args:
        value: An arbitrary value.

    Returns:
        The parameter.
    """    
    if isinstance(value, Param):
        return value
    return Param(raw_value=value)


def as_fixed(value: AbstractVariable) -> Param:
    """
    Returns `value` as a `parax.Fixed` variable by creating one if needed.

    Args:
        value: An arbitrary variable.

    Returns:
        A fixed version of the variable.
    """    
    if isinstance(value, Fixed):
        return value
    return Fixed(value)


Variable = AbstractVariable | ArrayLike