"""
The main Parameter class.
"""

import dataclasses
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array
import equinox as eqx
from distreqx.distributions import AbstractDistribution
from distreqx.bijectors import AbstractBijector
import distreqx.bijectors as bij

from parax.constraints import AbstractConstraint, Unconstrained
from parax.utils import (
    format_array, 
    format_distribution,
)

class Parameter(eqx.Module):
    """
    A container for a physical Parameter.

    `Parameter` is a JAX PyTree node (`eqx.Module`) that pairs an array
    with physical metadata. It maps unconstrained optimization spaces
    to bounded physical spaces using "constraints", and allows specifying
    probability distributions (priors) via `distreqx`.

    `Parameter` also provides native support for units via `unxt`
    by passing a string, an `unxt.Unit`, or an `unxt.Quantity`.

    Note that while `__jax_array__` and various dunder methods are overriden
    to provide seemless computations with other arrays, this support is experimental,
    and PyTrees using parameters should rather be unwrapped using the various utilities in
    `parax.tree` rather than relying on the implicit behaviour.
    """
    #: The parameter value in unconstrained space used by optimizers and samplers.
    raw_value: Array = eqx.field(converter=jnp.asarray)

    #: A combined multiplier acting as both the pre-conditioning scale and the physical unit.
    #: Transforms the base space to physical space. 
    #: Can be an Array or a `quax.ArrayValue`.
    scale: Array | Any = eqx.field(default=1.0)
    
    #: If True, the parameter should be ignored during optimization.
    fixed: Array = eqx.field(default=False, converter=jnp.asarray)

    #: The probability distribution in base space. Useful to specify priors for Bayesian inference.
    distribution: AbstractDistribution | None = eqx.field(default=None)
    
    #: The parameter constraint in base space. Useful to define bounds and conditioning for optimization.
    constraint: AbstractConstraint | None = eqx.field(default=None)

    #: String identifier(s) for the parameter.
    name: str | list[str] | None = eqx.field(default=None, static=True)

    #: Additional arbitrary metadata.
    metadata: dict = eqx.field(default_factory=dict, static=True)

    def __init__(
        self,
        base_value: Any | None = None,
        scale: Any = 1.0,
        *,
        raw_value: Any | None = None,
        fixed: bool = False,
        distribution: AbstractDistribution | None = None,
        constraint: AbstractConstraint | None = None,
        name: str | list[str] | None = None,
        metadata: dict | None = None,
    ):
        """
        Args:
            base_value: The base array value.
            scale: Linear preconditioning factor for the optimizer space. Can be any array-like value
                   that multiplies with a standard array (e.g. float, `unxt.Unit` or `unxt.Quantity`.)
                   If a `float` is passed, it is converted to an array.
                   If a `string` is passed, an `unxt.Unit` is created internally.
            raw_value: The unconstrained value. Mutually exclusive with `base_value`.
            fixed: If True, indicates the parameter should be frozen during optimization.
            name: Identifier for logging or unwrapping logic.
            distribution: A distreqx distribution defined in the base space.
            constraint: A parax constraint defining base bounds and mappings.
            metadata: Additional static metadata.
        """
        # Error checking
        if base_value is None and raw_value is None:
            raise ValueError("Must provide either `base_value` or `raw_value`.")
        if base_value is not None and raw_value is not None:
            raise ValueError("Cannot provide both `base_value` and `raw_value`.")
        
        # Scale normalization
        if isinstance(scale, float):
            scale = jnp.asarray(scale)
        elif isinstance(scale, str):
            import unxt
            scale = unxt.unit(scale)

        # Raw value normalization
        if base_value is not None:
            if constraint is not None and constraint.bijector is not None:
                raw_value = constraint.bijector.inverse(base_value)
            else:
                raw_value = base_value

        self.raw_value = jnp.asarray(raw_value, dtype=float)
        self.scale = scale
        self.fixed = jnp.asarray(fixed, dtype=bool)
        self.distribution = distribution
        self.constraint = constraint
        self.name = name
        self.metadata = metadata

    def replace(self, **kwargs: Any) -> "Parameter":
        """
        Creates a new Parameter with updated fields.

        This is a convenience method for shallow, node-level updates. 
        For deep updates across a PyTree of parameters, prefer `equinox.tree_at`
        or the utilities in `parax.tree`.

        Example:
            new_param = param.replace(raw_value=jnp.array(5.0))
        """
        return dataclasses.replace(self, **kwargs)        

    @property
    def unit(self) -> Any:
        """
        Read-only access to the physical dimensions of the scale, if any.
        """
        return getattr(self.scale, "unit", None)
    
    @property
    def base_value(self) -> Any:
        """
        The value in base space.

        Returns the constraint's bijector applied to the raw value.
        """
        return self.raw_to_base_bijector.forward(self.raw_value)

    @property
    def physical_value(self) -> Any:
        """
        The value in physical space.
        
        Returns the base value multiplied by `self.scale`.
        """
        return self.raw_to_physical_bijector.forward(self.raw_value)
    
    @property
    def raw_to_base_bijector(self) -> AbstractBijector:
        """
        Bijector mapping from raw (unconstrained) space to base (constrained) space.
        """
        constraint = self.constraint if self.constraint is not None else Unconstrained()
        return constraint.bijector
    
    @property
    def base_to_physical_bijector(self) -> AbstractBijector:
        """
        Bijector mapping from base (constrained) space to physical (scaled & constrained) space.
        """
        return bij.Scale(self.scale)

    @property
    def raw_to_physical_bijector(self) -> AbstractBijector:
        """
        Bijector mapping from raw (unconstrained) space to physical (scaled & constrained) space.
        
        Composes the unconstraining bijector with a scaling bijector.
        """
        return bij.Chain([self.base_to_physical_bijector, self.raw_to_base_bijector])    

    @property
    def shape(self) -> tuple[int, ...]:
        return self.raw_value.shape

    @property
    def size(self) -> int:
        return self.raw_value.size

    def __repr__(self):
        args = []
        
        # Support None in repr for equinox partitioning
        if self.raw_value is None:
            args.append("raw_value=None")
        else:
            args.append(f"base_value={format_array(self.base_value)}")
        
        # Safely check if scale is 1.0 without triggering JAX Tracer boolean errors
        if self.scale is not None and not hasattr(self.scale, 'unit') and jnp.allclose(self.scale, jnp.array(1.0)):
            args.append(f"scale={self.scale}")
            
        if self.fixed is not False:
            args.append(f"fixed={self.fixed}")
            
        if self.distribution is not None:
            args.append(f"distribution={format_distribution(self.distribution)}")
            
        if self.constraint is not None:
            args.append(f"constraint={self.constraint}")
            
        if self.name is not None:
            args.append(f"name={repr(self.name)}")
            
        if self.metadata:
            for k, v in self.metadata.items():
                args.append(f"{k}={repr(v)}")
            
        return f"Parameter({', '.join(args)})"

    # =========================================================================
    # Interactive Math Dunders
    # =========================================================================
    
    def __jax_array__(self): return self.physical_value
    def __add__(self, other): return self.physical_value + other
    def __radd__(self, other): return other + self.physical_value
    def __sub__(self, other): return self.physical_value - other
    def __rsub__(self, other): return other - self.physical_value
    def __mul__(self, other): return self.physical_value * other
    def __rmul__(self, other): return other * self.physical_value
    def __truediv__(self, other): return self.physical_value / other
    def __rtruediv__(self, other): return other / self.physical_value
    def __pow__(self, other): return self.physical_value ** other


def asparam(value: Any) -> Parameter:
    if isinstance(value, Parameter):
        return value

    return Parameter(base_value=value)


def field(
    default: Any = dataclasses.MISSING,
    scale: Any = 1.0,
    *,
    constraint: AbstractConstraint | None = None,
    distribution: AbstractDistribution | None = None,
    fixed: Any = False,
    name: str | list[str] | None = None,
    **kwargs
) -> Any:
    """
    Specifies a Parax Parameter field within an Equinox module.

    This function defines the default physical metadata (scale, bounds, fixed state)
    for a parameter. When the module is initialized, any raw values passed to this 
    field are automatically converted into fully configured `Parameter` objects. 
    Users can override these defaults by passing an explicit `Parameter` object instead.

    Args:
        default: The default base value. If omitted, this field becomes required.
        scale: Physical unit or scaling factor (default: 1.0).
        constraint: base bounds and mapping logic.
        distribution: Prior probability distribution for Bayesian inference.
        fixed: Whether the parameter is frozen during optimization (default: False).
        name: String identifier for the parameter.
        **kwargs: Additional static metadata to store in the parameter.
    """
    def _converter(val: Any) -> Parameter:
        if isinstance(val, Parameter):
            return val
        
        return Parameter(
            base_value=val,
            scale=scale,
            constraint=constraint,
            distribution=distribution,
            fixed=fixed,
            name=name,
            **kwargs
        )

    # Build the keyword arguments for Equinox's underlying field mapping
    field_kwargs = {
        "converter": _converter,
        "metadata": kwargs
    }
    
    # Only pass 'default' to Equinox if the user actually provided one.
    # Otherwise, Equinox knows this is a strictly required argument.
    if default is not dataclasses.MISSING:
        field_kwargs["default"] = default
        
    return eqx.field(**field_kwargs)