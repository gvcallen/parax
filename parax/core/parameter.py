import json
import dataclasses
from copy import deepcopy
from typing import Any
import jax.numpy as jnp
import equinox as eqx
import numpyro.distributions as dist
from numpyro.distributions.transforms import Transform
from numpyro.distributions.distribution import Distribution

from parax.core.field import field
from parax.core.parameter_metadata import ParameterMetadata
from parax.utils import (
    format_val, 
    split_vectorized_distribution, 
    serialize_distribution, 
    deserialize_distribution, 
    format_distribution,
    serialize_transform,
    deserialize_transform,
    format_transform,
)

class Parameter(eqx.Module):
    """
    A container for a JAX parameter.

    This class serves as the fundamental building block for defining the
    tunable or fixed parameters in your own models, or within a **parax** `Module`.
    
    It is designed to be a flexible container that behaves like a standard numerical type
    (e.g., a `numpy.ndarray`) while holding additional metadata for model
    training and analysis.

    Usage
    -----
    
    * Use in mathematical operations just like a JAX/numpy array.
    * ``Parameter`` objects are JAX PyTrees, compatible with JAX transformations (jit, grad).
    * Mark as ``fixed`` (honoured by :meth:`parax.partition`).
    * Associate distributions, specified as numpyro distributions (uniform, normal, etc.).

    Attributes
    ----------
    value : jnp.ndarray
        The underlying unscaled value. Automatically converted to a float64 array.
    fixed : bool
        If True, the parameter is treated as a constant during optimization/sampling.
    metadata : ParameterMetadata or None
        The hidden structure containing all extended parameter properties.
    distribution : numpyro.distributions.Distribution or None
        The distribution associated with this parameter.
    transform : numpyro.distributions.transforms.Transform or None
        A bijector used to map unconstrained optimizer spaces to constrained parameter spaces.
    bounds : jnp.ndarray or None
        Absolute physical or numerical limits for the parameter.
    scale : float
        A scaling factor. The effective value used in calculations is ``value * scale``.
    name : str, list[str], or None
        An optional programmatic name for the parameter.
    description : str or None
        An optional human-readable description of the parameter's purpose.
    units : str or None
        An optional semantic unit for plotting and validation (e.g., "pF", "GHz").
    info : dict
        A dictionary containing any arbitrary user-defined metadata.
    """
    value: jnp.ndarray
    fixed: bool = field(static=True)
    metadata: ParameterMetadata | None = None

    def __init__(
        self, 
        value: Any, 
        fixed: bool = False, 
        metadata: ParameterMetadata | None = None, 
        **kwargs
    ):
        """
        Initializes the parameter. Core metadata and arbitrary kwargs are automatically 
        routed into the hidden `ParameterMetadata` struct. 
        
        If a transform is provided, the input `value` is assumed to be in the physical 
        (constrained) space and is automatically inverted to store the latent 
        (unconstrained) value for the optimizer.
        """
        # 1. Route the metadata first
        if metadata is not None:
            self.metadata = metadata
        else:
            name = kwargs.pop("name", None)
            description = kwargs.pop("description", None)
            units = kwargs.pop("units", None)
            distribution = kwargs.pop("distribution", None)
            transform = kwargs.pop("transform", None)
            bounds = kwargs.pop("bounds", None)
            scale = kwargs.pop("scale", 1.0)
            
            if isinstance(name, tuple):
                name = list(name)
                
            if bounds is not None:
                bounds = jnp.asarray(bounds)
                
            if (distribution is None and transform is None and bounds is None and 
                scale == 1.0 and name is None and description is None and units is None and not kwargs):
                self.metadata = None
            else:
                self.metadata = ParameterMetadata(
                    name=name,
                    description=description,
                    units=units,
                    distribution=distribution,
                    transform=transform,
                    bounds=bounds,
                    scale=scale,
                    info=kwargs 
                )

        # 2. Process the physical value into latent space
        raw_value = jnp.asarray(value)
        
        # MAGIC: Invert the physical value if a transform exists
        if self.metadata is not None and self.metadata.transform is not None:
            raw_value = self.metadata.transform.inv(raw_value)
            
        self.value = raw_value
        self.fixed = fixed

    # --- Property Overloads for Seamless API Compatibility ---
    
    @property
    def name(self) -> str | list[str] | None:
        return self.metadata.name if self.metadata is not None else None

    @property
    def description(self) -> str | None:
        return self.metadata.description if self.metadata is not None else None

    @property
    def units(self) -> str | None:
        return self.metadata.units if self.metadata is not None else None

    @property
    def distribution(self) -> Distribution | None:
        return self.metadata.distribution if self.metadata is not None else None

    @property
    def transform(self) -> Transform | None:
        return self.metadata.transform if self.metadata is not None else None

    @property
    def bounds(self) -> jnp.ndarray | None:
        return self.metadata.bounds if self.metadata is not None else None

    @property
    def scale(self) -> float:
        return self.metadata.scale if self.metadata is not None else 1.0

    @property
    def info(self) -> dict:
        return self.metadata.info if self.metadata is not None else {}    

    @property
    def shape(self) -> tuple[int, ...]:
        """
        The shape of this parameter.
        """
        return self.value.shape
    
    @property
    def size(self) -> int:
        """
        The number of dimensions for this parameter.
        """
        return self.value.size

    @property
    def mean(self) -> jnp.ndarray:
        r"""
        The unscaled mean value of the parameter's distribution.
        
        Returns
        -------
        jnp.ndarray
            The mean value, or the current value if no distribution is set.
        """        
        # ImproperUniform doesn't have a defined mean, so we fall back to the initialized value
        if isinstance(self.distribution, dist.ImproperUniform) or self.distribution is None:
            return self.value
        elif isinstance(self.distribution, dist.Delta):
            return self.distribution.v
            
        return self.distribution.mean
    
    def with_value(self, value: jnp.ndarray) -> 'Parameter':
        r"""
        Return a copy of the parameter with a new unscaled value.

        Parameters
        ----------
        value : jnp.ndarray
            The new unscaled value to set.

        Returns
        -------
        Parameter
            A copy of this object with ``value`` replaced.
        """
        return dataclasses.replace(self, value=jnp.asarray(value))

    def shifted(self, amount: jnp.ndarray) -> 'Parameter':
        r"""
        Returns a copy of this parameter with its mean shifted by an amount.

        Parameters
        ----------
        amount : jnp.ndarray
            The amount to shift by. Can be positive or negative.

        Returns
        -------
        Parameter
            A copy of this object with ``value`` replaced.
        """
        new_value = self.value + amount
        new_distribution = deepcopy(self.distribution)
        if isinstance(new_distribution, dist.Uniform):
            new_distribution.low += amount
            new_distribution.high += amount
        elif isinstance(new_distribution, dist.Normal):
            new_distribution.loc += amount
        else:
            raise Exception("Can only call 'shifted' on a normal or uniform parameter")
            
        return self.with_distribution(new_distribution).with_value(new_value)
    
    def with_distribution(self, distribution: Distribution) -> 'Parameter':
        r"""
        Return a copy of the parameter with a new distribution.

        Parameters
        ----------
        distribution : numpyro.distributions.Distribution
            The distribution to associate with this parameter.

        Returns
        -------
        Parameter
            A copy of this object with ``distribution`` replaced.

        Raises
        ------
        Exception
            If ``dist`` is not a numpyro Distribution.
        """
        if not isinstance(distribution, Distribution):
            raise Exception('Only numpyro distributions are supported as parameter distributions')
        
        if self.metadata is None:
            new_meta = ParameterMetadata(distribution=distribution)
        else:
            new_meta = dataclasses.replace(self.metadata, distribution=distribution)
            
        return dataclasses.replace(self, metadata=new_meta)
    
    def flattened(self, separator='_') -> 'list[Parameter]':
        r"""
        Flatten self into a list of scalar Parameters.
        
        If the internal parameter is scalar, the list will contain self.
        Otherwise, the parameter is split (de-vectorized) if possible.
        
        Parameters
        ----------
        separator : str, optional, default='_'
            Separator used for naming split parameters (e.g., name_0, name_1).

        Returns
        -------
        list[Parameter]
            The list of individual parameters.

        Raises
        ------
        ValueError
            If any internal distributions cannot be de-vectorized.
        """
        # Handle scalar / 0-d array. Check if name is NOT a list.
        if self.value.ndim == 0 and not isinstance(self.name, list):
            return [self]
            
        # Flatten the value
        flat_val = jnp.ravel(self.value)
        
        # Split distribution if present
        if self.distribution is not None:
            dists_split = split_vectorized_distribution(self.distribution)
        else:
            dists_split = [None] * flat_val.size

        # Generate names based on the type of self.name
        if isinstance(self.name, list):
            if len(self.name) != flat_val.size:
                raise ValueError(f"Length of name list ({len(self.name)}) must match parameter size ({flat_val.size}).")
            flat_names = self.name
        elif self.name is not None:
            flat_names = [f"{self.name}{separator}{i}" for i in range(flat_val.size)]
        else:
            flat_names = [None] * flat_val.size
                
        return [
            Parameter(
                value=val, 
                distribution=p, 
                transform=self.transform,
                bounds=self.bounds, # You may need split_vectorized_bounds if arrays get large
                fixed=self.fixed, 
                scale=self.scale, 
                name=n,
                description=self.description,
                units=self.units,
                **self.info # Safely pass along arbitrary metadata
            ) 
            for val, p, n in zip(flat_val, dists_split, flat_names)
        ]
    
    def as_fixed(self) -> 'Parameter':
        r"""
        Return a copy of self with ``fixed=True``.

        Returns
        -------
        Parameter
            The new, fixed parameter.
        """
        return dataclasses.replace(self, fixed=True)
    
    def as_free(self) -> 'Parameter':
        r"""
        Return a copy of self with ``fixed=False``.

        Returns
        -------
        Parameter
            The new, free parameter.
        """
        return dataclasses.replace(self, fixed=False)
    
    # Arithmetic and array conversions
    def __array__(self, dtype=None):
        r"""
        NumPy array interface.

        Parameters
        ----------
        dtype : Any, optional
            Desired dtype.

        Returns
        -------
        jnp.ndarray
            The scaled value as an array (``value * scale``).
        """
        val = self.value
        if self.transform is not None:
            val = self.transform(val)
            
        return jnp.asarray(val * self.scale, dtype=dtype)
    
    def __jax_array__(self, dtype=None):
        r"""
        JAX array interface.

        Parameters
        ----------
        dtype : Any, optional
            Desired dtype.

        Returns
        -------
        jnp.ndarray
            The scaled value as an array (``value * scale``).
        """
        val = self.value
        if self.transform is not None:
            val = self.transform(val)
            
        return jnp.asarray(val * self.scale, dtype=dtype)
    
    def __len__(self):
        r"""
        Length of the parameter value.

        Returns
        -------
        int
            ``1`` for scalars, otherwise ``len(value)``.
        """
        if len(self.value.shape) == 0:
            return 1 # e.g. for jax scalars
        return len(self.value)
    
    def __repr__(self):
        # Build the representation dynamically
        # 'value' is always printed as it is the core of the Parameter
        args = [f"value={format_val(self.value)}"]
        
        # Only add attributes if they deviate from the default
        if self.scale != 1.0:
            args.append(f"scale={self.scale}")
            
        if self.fixed is not False:
            args.append(f"fixed={self.fixed}")
            
        if self.distribution is not None:
            args.append(f"distribution={format_distribution(self.distribution)}")
            
        if self.transform is not None:
            # Use the custom formatter instead of raw repr
            args.append(f"transform={format_transform(self.transform)}")
            
        if self.bounds is not None:
            args.append(f"bounds={format_val(self.bounds)}")
            
        if self.name is not None:
            args.append(f"name={repr(self.name)}")
            
        if self.units is not None:
            args.append(f"units={repr(self.units)}")

        if self.description is not None:
            args.append(f"description={repr(self.description)}")
            
        if self.info:
            for k, v in self.info.items():
                args.append(f"{k}={repr(v)}")
            
        return f"Parameter({', '.join(args)})"
            
    def __add__(self, other):
        r"""Elementwise addition."""
        return jnp.add(jnp.array(self), jnp.array(other))
    
    def __sub__(self, other):
        r"""Elementwise subtraction."""
        return jnp.subtract(jnp.array(self), jnp.array(other))
    
    def __mul__(self, other):
        r"""Elementwise multiplication."""
        return jnp.multiply(jnp.array(self), jnp.array(other))

    def __truediv__(self, other):
        r"""Elementwise true division."""
        return jnp.divide(jnp.array(self), jnp.array(other))

    def __radd__(self, other):
        r"""Reflected elementwise addition."""
        return jnp.add(jnp.array(other), jnp.array(self))
    
    def __rsub__(self, other):
        r"""Reflected elementwise subtraction."""
        return jnp.subtract(jnp.array(other), jnp.array(self))

    def __rmul__(self, other):
        r"""Reflected elementwise multiplication."""
        return jnp.multiply(jnp.array(other), jnp.array(self))
    
    def __rtruediv__(self, other):
        r"""Reflected elementwise true division."""
        return jnp.divide(jnp.array(other), jnp.array(self))
    
    def copy(self):
        r"""
        Return a shallow copy.

        Returns
        -------
        Parameter
            A copy created via ``dataclasses.replace``.
        """
        return dataclasses.replace(self)
    
    def to_json(self) -> str:
        r"""
        Serialize the parameter to a JSON string.
        Omits any fields that are None or empty to keep the payload lightweight.

        Returns
        -------
        str
            A JSON-formatted string containing the parameter's data.
        """
        # 1. Base required attributes (scale is included as it defaults to 1.0, not None)
        d = {
            "value": self.value.tolist(),
            "fixed": self.fixed,
            "scale": self.scale
        }
        
        # 2. Add metadata fields only if they are actively used
        if self.distribution is not None:
            d["distribution"] = serialize_distribution(self.distribution)
            
        if self.transform is not None:
            d["transform"] = serialize_transform(self.transform)
            
        if self.bounds is not None:
            d["bounds"] = self.bounds.tolist()
            
        if self.name is not None:
            d["name"] = self.name
            
        if self.description is not None:
            d["description"] = self.description
            
        if self.units is not None:
            d["units"] = self.units
            
        if self.info: # Evaluates to False if the dictionary is empty
            d["info"] = self.info 
            
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Parameter":
        r"""
        Deserialize a parameter from a JSON string.

        Parameters
        ----------
        s : str
            The JSON string produced by :meth:`to_json`.

        Returns
        -------
        Parameter
            A reconstructed :class:`Parameter` instance.
        """
        d = json.loads(s)
        
        # Extract the fields that don't go into kwargs
        value = d.pop("value")
        fixed = d.pop("fixed", False)
        
        # Reconstruct complex NumPyro objects if they are present in the JSON
        if "distribution" in d:
            d["distribution"] = deserialize_distribution(d["distribution"])
            
        if "transform" in d:
            d["transform"] = deserialize_transform(d["transform"])
            
        # Merge arbitrary info back into the top level dictionary
        # so our smart __init__ can sweep them up via **kwargs
        info_dict = d.pop("info", {})
        d.update(info_dict)
        
        return cls(value=value, fixed=fixed, **d)
    
from typing import Any, TYPE_CHECKING

def is_param(x) -> bool:
    r"""
    Check if an object is an instance of a `Parameter`.

    Parameters
    ----------
    x
        The object to check.

    Returns
    -------
    bool
        `True` if the object is a Parameter, `False` otherwise.
    """
    return isinstance(x, Parameter)

def is_valid_param(x) -> bool:
    r"""
    Check if an object is an instance of a `Parameter` and if its value is not None.

    Parameters
    ----------
    x
        The object to check.

    Returns
    -------
    bool
        `True` if the object is a valid Parameter, `False` otherwise.
    """
    return isinstance(x, Parameter) and x.value is not None

def is_free_param(x) -> bool:
    r"""
    Check if an object is a non-fixed `Parameter`.

    Parameters
    ----------
    x
        The object to check.

    Returns
    -------
    bool
        `True` if the object is a non-fixed Parameter, `False` otherwise.
    """
    return isinstance(x, Parameter) and not x.fixed

def is_fixed_param(x) -> bool:
    r"""
    Check if an object is a fixed `Parameter`.

    Parameters
    ----------
    x
        The object to check.

    Returns
    -------
    bool
        `True` if the object is a fixed Parameter, `False` otherwise.
    """
    return isinstance(x, Parameter) and x.fixed

def as_param(x: Any | list[Any] | dict[str, Any], **kwargs) -> "Parameter":
    r"""
    Ensure an object is a `Parameter` or container over parameters.

    If the object is already a `Parameter`, it is returned unchanged.
    Otherwise, the underlying objects are converted into new `Parameter` objects.

    Parameters
    ----------
    x
        The object to convert.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor (e.g. `name`).

    Returns
    -------
    Parameter
        The object wrapped as a `Parameter`.
    """
    from parax.parameters import Free, Fixed
    
    if isinstance(x, Parameter):
        return x
    elif isinstance(x, list):
        return [as_param(xi, **kwargs) for xi in x]
    elif isinstance(x, dict):
        return {k: as_param(xi, **kwargs) for k, xi in x.items()}
    else:
        is_fixed = kwargs.pop('fixed', False)
        if is_fixed:
            return Fixed(value=x, **kwargs)
        else:
            return Free(value=x, **kwargs)