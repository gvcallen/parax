"""
Computable JAX arrays with metadata and constraints.

This module provides the core variable types for Parax. They act as 
array-like objects that can be directly injected into JAX computations 
or unwrapped prior to execution.
"""
from functools import reduce
from abc import abstractmethod
from typing import Any, Iterator, Callable, TypeGuard, Self

import jax
import jax.numpy as jnp
from jaxtyping import Array, Inexact
import equinox as eqx
from distreqx.distributions import AbstractDistribution
from distreqx.bijectors import AbstractBijector, Chain

from parax.constraints import AbstractConstraint, AbstractConstrainable, RealLine, infer_distribution_constraint, is_constrainable, Transformed as TransformedConstraint, intersect as intersect_constraints
from parax.constants import AbstractConstant
from parax.wrappers import AbstractUnwrappable, AbstractWrappable, as_unwrapped, as_opaque
from parax.annotation import AbstractAnnotated
from parax.bounds import AbstractBounded
from parax.probability import AbstractProbabilistic, truncate_distribution


class AbstractVariable(AbstractUnwrappable[Array]):
    """
    The abstract interface for all model variables.

    Derive from this class and override `value` to implement
    custom variable unwrapping behaviour.

    All parameters in Parax, such as `parax.Random`,
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
    
    @property
    def ndim(self) -> Any: return self.value.ndim

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


class Fixed(AbstractVariable, AbstractConstant[Param], AbstractWrappable[Array]):
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

    def free(self) -> AbstractVariable:
        return self.raw_value

    @property
    def value(self) -> Array:
        value = self.raw_value
        if isinstance(self.raw_value, AbstractVariable):
            value = value.value
        return jax.lax.stop_gradient(value)
    
    def wrap(self, value: Array) -> Self:
        return eqx.tree_at(lambda m: m.raw_value, self, value)    
   

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
        fn: The callable used to transform the raw value.
        raw_value: The raw value used by optimizers and samplers.
    """
    fn: Callable = eqx.field(converter=as_opaque)
    raw_value: Param = eqx.field(converter=as_param)

    @property
    def value(self) -> Array:
        """
        The derived value.
        
        Returns the raw state transformed by the derivation function.
        """
        return as_unwrapped(self.fn)(jnp.asarray(self.raw_value))
    
    
class Transformed(AbstractVariable, AbstractWrappable[Array]):
    """
    A variable transformed by a bijector.
     
    The parameter's value is dynamically derived via a bijective transform.
    
    Note that this simply applies forward/inverse passes during unwrapping,
    and does NOT apply any special treatment to any other variable types
    (e.g. `parax.Constrained` or `parax.Random` variables).

    Attributes:
        bijector: The bijector used to transform the raw value.
        raw_value: The raw value used by optimizers and samplers.
    """
    bijector: AbstractBijector = eqx.field(converter=as_opaque)
    raw_value: Param = eqx.field(converter=as_param)
    
    def __init__(self, bijector: AbstractBijector, raw_value: Param):
        """
        Args:
            bijector: The bijector used to transform the raw value.
            raw_value: The underlying value to be fixed.
        """
        if isinstance(raw_value, Transformed):
            bijector = Chain([bijector, as_unwrapped(raw_value.bijector)])
            raw_value = raw_value.raw_value
        
        self.bijector = bijector
        self.raw_value = raw_value

    @property
    def value(self) -> Array:
        """
        The derived value.
        
        Returns the raw state transformed by the derivation function.
        """
        return as_unwrapped(self.bijector).forward(jnp.asarray(self.raw_value))
    
    def wrap(self, value: Array) -> Self:
        new_raw = as_unwrapped(self.bijector).inverse(value)
        return eqx.tree_at(lambda x: x.raw_value, self, new_raw)
    
    
class Bounded(
    AbstractVariable,
    AbstractBounded[Array],
    AbstractWrappable[Array]
):
    """
    A bounded variable.
    
    This simply attaches bounds to an existing variable or an array,
    and does not apply any bijective constraints. For enforcing
    constraints on an array, use `parax.variables.Constrained`.
    
    Attributes:
        bounds: The parameter bounds.
        raw_value: The raw, unconstrained value on the real number line.
    """
    bounds: tuple[Array, Array] = eqx.field(converter=as_opaque)
    raw_value: Param = eqx.field(converter=jnp.asarray)

    def __init__(
        self,
        bounds: tuple[Array, Array],
        raw_value: Param | None = None,
    ):
        """
        Args:
            bounds: The parameter bounds.
            raw_value: The underlying value. Must lie within `bounds`.
        """
        if raw_value is not None:
            raw_array = jnp.asarray(raw_value)
            lower, upper = jnp.asarray(bounds)
            
            is_out_of_bounds = jnp.any((raw_array < lower) | (raw_array > upper))
            raw_value = eqx.error_if(
                raw_value,
                is_out_of_bounds,
                "Bounded variable initialized with a value outside of its specified bounds."
            )

        self.bounds = bounds
        self.raw_value = raw_value
        
    @property
    def value(self) -> Array:
        return jnp.asarray(self.raw_value)
    
    def wrap(self, value: Array) -> Self:
        return eqx.tree_at(lambda x: x.raw_value, self, jnp.asarray(value))
    

class Constrained(
    AbstractVariable,
    AbstractConstrainable[Array],
    AbstractWrappable[Array]
):
    """
    A constrained variable.
    
    The constraint is specified via a `parax.AbstractConstraint`.

    The constraint is automatically applied as a bijection mapping during 
    evaluation. Implements the `parax.bounds.AbstractBounded` interface
    for integration with bounded optimizers.

    Attributes:
        constraint: The parameter constraint defining bounds and bijector mappings.
        raw_value: The raw, unconstrained value on the real number line.
    """
    constraint: AbstractConstraint = eqx.field(converter=as_opaque)
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
            raw_value = eqx.error_if(
                raw_value,
                jnp.any(jnp.isnan(raw_value)),
                "Constraint violated for variable upon initialization (produced NaNs)."
            )

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
    AbstractConstrainable[Array],
    AbstractWrappable[Array]
):
    """
    A random variable with an optional constraint.
    
    The distribution is specified via a `distreqx.distributions.AbstractDistribution`.
    The constraint is specified via a `parax.constraint.AbstractConstraint`.

    The variable implements the `parax.probability.AbstractProbabilistic` interface
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
    distribution: AbstractDistribution = eqx.field(converter=as_opaque)
    constraint: AbstractConstraint = eqx.field(converter=as_opaque)
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
            constraint = infer_distribution_constraint(distribution)

        # Calculate unconstrained raw_value
        if value is not None:
            raw_value = constraint.bijector.inverse(jnp.asarray(value))
            raw_value = eqx.error_if(
                raw_value,
                jnp.any(jnp.isnan(raw_value)),
                "Constraint violated for variable upon initialization (produced NaNs)."
            )
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
        orig_lower, orig_upper = as_unwrapped(self.constraint).bounds
        new_lower, new_upper = as_unwrapped(constraint).bounds
        has_shrunk = jnp.any(new_lower > orig_lower) or jnp.any(new_upper < orig_upper)

        distribution = self.distribution
        value = self.value
        if has_shrunk:
            try:
                distribution = truncate_distribution(as_unwrapped(distribution), new_lower, new_upper)
            except Exception:
                import warnings
                dist_name = type(as_unwrapped(distribution)).__name__
                warnings.warn(
                    f"A constraint was applied, but the distribution ({dist_name}) "
                    f"could not be automatically truncated and will therefore be warped. "
                    f"It is recommended to use a distribution that aligns with the constraints.",
                    UserWarning, stacklevel=2
                )
                
        return Random(distribution=distribution, constraint=constraint, value=value)
    
    def wrap(self, value: Array) -> Self:
        new_raw = as_unwrapped(self.constraint).bijector.inverse(value)
        return eqx.tree_at(lambda x: x.raw_value, self, new_raw)

   
def tree_named_params(tree: Any) -> dict[str, Any]:
    """
    Extracts a dictionary of named parameters from a JAX PyTree.

    This function traverses a PyTree (such as an Equinox module)
    and constructs readable string paths for each parameter leaf,
    mapping these paths to their corresponding values.
    
    It is useful for inspecting model state, debugging, or logging.

    Parameters
    ----------
    tree : Any
        The JAX PyTree to inspect.

    Returns
    -------
    dict[str, Any]
        A dictionary where keys are string representations of the paths to the 
        leaves (e.g., '.res.R' or '.cascade[0].L') and values are the leaves 
        themselves.
    """
    leaves_with_path, _ = jax.tree.flatten_with_path(tree, is_leaf=is_param)
    return {jax.tree_util.keystr(path): leaf for path, leaf in leaves_with_path if is_param(leaf)}


def constrain_param(variable: Param, *constraints: AbstractConstraint) -> AbstractConstrainable:
    """Intelligently applies a constraint to a parameter (variable or array).

    This function acts as a smart router for applying physical bounds to variables,
    regardless of how heavily wrapped they are. It safely drills through non-constrainable
    wrappers (like `Fixed` or `Tagged`), promotes unconstrained bases (like `Real` or
    raw JAX arrays), and correctly propagates constraints backwards through bijective
    transformations.

    Args:
        variable (Param): The target variable or standard JAX inexact array.
        *constraints (AbstractConstraint): The physical constraints to apply.

    Returns:
        Param: A new instance of the variable with the constraint applied.

    Raises:
        TypeError: If the variable type is not supported for dynamic constraining.
    """
    constraint = reduce(intersect_constraints, constraints)
    
    if is_constrainable(variable):
        return variable.constrain(constraint)
    elif isinstance(variable, Transformed):
        try:
            from distreqx.bijectors import Inverse
        except:
            from parax._bijectors import Inverse
        inverse_bij = Inverse(as_unwrapped(variable.bijector))
        transformed_constraint = TransformedConstraint(constraint, inverse_bij)
        new_inner = constrain_param(variable.raw_value, transformed_constraint)
        return Transformed(variable.bijector, new_inner)
    elif isinstance(variable, (Fixed, Tagged)):
        new_raw = constrain_param(variable.raw_value, constraint)
        return eqx.tree_at(lambda x: x.raw_value, variable, new_raw)
    elif isinstance(variable, Real) or eqx.is_inexact_array(variable):
        return Constrained(constraint, value=jnp.array(variable))
    else:
        raise TypeError(
            f"Cannot dynamically inject a constraint into type {type(variable)}. "
            "Ensure the target is a valid Parax variable or JAX inexact array."
        )