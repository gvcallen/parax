import json
import dataclasses
from typing import Any

import jax.numpy as jnp
import equinox as eqx
from distreqx.distributions import AbstractDistribution, Transformed
from distreqx.bijectors import AbstractBijector, Chain


from parax._bijectors import Inverse
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
    A container for a parameter.

    This class serves as the fundamental building block for defining
    parameters with metadata within Equinox modules. It is designed
    to be a flexible container that behaves like a standard JAX array
    (i.e.., a `jax.numpy.ndarray`) while holding additional metadata for model
    training and analysis.

    Usage
    -----
    * Use in mathematical operations just like a JAX/numpy array.
    * `Parameter` objects are JAX PyTrees, compatible with JAX transformations (jit, grad).
    * Mark as `fixed` (honored by `parax.partition`).
    * Associate distributions and transforms/bijectors using `distreqx`.
    """
    #: The underlying unscaled, untransformed, latent value.
    latent_value: jnp.ndarray
      
    #: If True, the parameter is treated as a constant during optimization/sampling.
    fixed: bool = field(default=False, static=True)
        
    #: The hidden structure containing all extended parameter properties.
    metadata: ParameterMetadata | None = None

    def __init__(
        self, 
        value: Any | None = None, 
        fixed: bool = False, 
        metadata: ParameterMetadata | None = None, 
        n: int | None = None,
        **kwargs
    ):
        """
        During initialization, core metadata and arbitrary kwargs are automatically routed into the 
        hidden `ParameterMetadata` struct. If a transform/bijector is provided, the input 
        `value` is assumed to be in the physical (constrained) space and is 
        automatically inverted to store the latent (unconstrained) value.
        
        The parameter `n` allows for vectorizing the input value and metadata 
        across `n` dimensions.
        """
        latent_value = kwargs.pop('latent_value', None)
        
        if latent_value is None and value is None:
            raise Exception("Must pass one of either `latent_value` or `value` to Parameter constructor")
        
        # 1. Handle Vectorization (n)
        if n is not None:
            if value is not None:
                value = jnp.broadcast_to(jnp.asarray(value), (n,) + jnp.shape(value))
            if latent_value is not None:
                latent_value = jnp.broadcast_to(jnp.asarray(latent_value), (n,) + jnp.shape(latent_value))

        # 2. Extract known metadata keys
        updates = {}
        for key in ["name", "distribution", "transform", "bounds", "scale"]:
            if key in kwargs:
                updates[key] = kwargs.pop(key)
                
        # Handle vectorization for specific metadata fields
        if n is not None and n != 1:
            # Vectorize bounds: if shape is (2,), it becomes (n, 2)
            if "bounds" in updates and updates["bounds"] is not None:
                b = jnp.asarray(updates["bounds"])
                updates["bounds"] = jnp.broadcast_to(b, (n,) + b.shape)
            
            # Vectorize name: if a string is passed, turn into a list of n strings
            if "name" in updates and isinstance(updates["name"], str):
                updates["name"] = [f"{updates['name']}_{i}" for i in range(n)]

        # Format specific fields
        if "name" in updates and isinstance(updates["name"], tuple):
            updates["name"] = list(updates["name"])
            
        if "bounds" in updates and updates["bounds"] is not None:
            updates["bounds"] = jnp.asarray(updates["bounds"])
            
        # Any remaining kwargs belong in the custom 'info' dict
        info_updates = kwargs if len(kwargs) > 0 else {}

        # 3. Reconcile metadata
        if metadata is not None:
            if updates or info_updates:
                new_info = dict(metadata.info) if metadata.info is not None else {}
                new_info.update(info_updates)
                
                self.metadata = dataclasses.replace(
                    metadata, 
                    **updates, 
                    info=new_info if new_info else None
                )
            else:
                self.metadata = metadata
        else:
            name = updates.get("name", None)
            distribution = updates.get("distribution", None)
            transform = updates.get("transform", None)
            bounds = updates.get("bounds", None)
            scale = updates.get("scale", 1.0)
            
            if (distribution is None and transform is None and bounds is None and 
                scale == 1.0 and name is None and not info_updates):
                self.metadata = None
            else:
                self.metadata = ParameterMetadata(
                    name=name,
                    distribution=distribution,
                    transform=transform,
                    bounds=bounds,
                    scale=scale,
                    info=info_updates if info_updates else None
                )
                
        # 4. Handle latent value extraction/inversion
        if latent_value is None:
            latent_value = jnp.asarray(value)
            if self.metadata is not None and self.metadata.transform is not None:
                # The bijector in distreqx handles vectorized inputs automatically
                latent_value = self.metadata.transform.inverse(latent_value)
                
                if jnp.any(jnp.isnan(latent_value)):
                    raise ValueError(f"Got nan while applying bijector inverse in parameter init.")
            
        self.latent_value = latent_value
        self.fixed = fixed

    @property
    def value(self) -> jnp.ndarray:
        """
        Get the unscaled physical space value.

        Returns
        -------
        jnp.ndarray
            The parameter value mapped through the transform (if any), but unscaled.
        """
        raw_val = self.latent_value
        if self.transform is not None:
            raw_val = self.transform.forward(raw_val)
            
        return jnp.asarray(raw_val)
    
    @property
    def name(self) -> str | list[str] | None:
        """
        Get the parameter name.

        Returns
        -------
        str, list of str, or None
            The name or list of names associated with the parameter.
        """
        return self.metadata.name if self.metadata is not None else None

    @property
    def distribution(self) -> AbstractDistribution | None:
        """
        Get the parameter distribution.

        Returns
        -------
        AbstractDistribution or None
            The probability distribution associated with the parameter.
        """
        return self.metadata.distribution if self.metadata is not None else None

    @property
    def transform(self) -> AbstractBijector | None:
        """
        Get the parameter transform.

        Returns
        -------
        AbstractBijector or None
            The bijector used to map between latent and physical space.
        """
        return self.metadata.transform if self.metadata is not None else None

    @property
    def bounds(self) -> jnp.ndarray | None:
        """
        Get the parameter bounds.

        Returns
        -------
        jnp.ndarray or None
            The physical bounds of the parameter.
        """
        return self.metadata.bounds if self.metadata is not None else None

    @property
    def scale(self) -> float:
        """
        Get the parameter scale.

        Returns
        -------
        float
            The multiplier applied to the physical value for numerical operations.
        """
        return self.metadata.scale if self.metadata is not None else 1.0

    @property
    def info(self) -> dict:
        """
        Get the parameter's custom metadata.

        Returns
        -------
        dict
            Any arbitrary keyword arguments passed during initialization.
        """
        if self.metadata is None or self.metadata.info is None:
            return {}
        return self.metadata.info

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Get the shape of the parameter.

        Returns
        -------
        tuple of int
            The shape of the latent array.
        """
        return self.latent_value.shape
    
    @property
    def size(self) -> int:
        """
        Get the number of elements in the parameter.

        Returns
        -------
        int
            The total size of the latent array.
        """
        return self.latent_value.size
    
    @property
    def latent_distribution(self) -> AbstractDistribution | None:
        """
        Get the parameter distribution in the latent space.

        Returns
        -------
        AbstractDistribution or None
            The physical probability distribution mapped back to the latent space 
            via the inverse of the parameter's transform.
        """
        dist = self.distribution
        transform = self.transform

        if dist is None:
            return None
        if transform is None:
            return dist
        
        return Transformed(dist, Inverse(transform))    
    
    def with_name(self, name: str) -> 'Parameter':
        """
        Return a copy of the parameter with a new physical name.

        Parameters
        ----------
        name : str
            The new name.

        Returns
        -------
        Parameter
            A copy of this object with `name` updated.
        """
        if self.metadata is None:
            new_meta = ParameterMetadata(name=name)
        else:
            new_meta = dataclasses.replace(self.metadata, name=name)
            
        return dataclasses.replace(self, metadata=new_meta)
    
    def with_value(self, value: jnp.ndarray) -> 'Parameter':
        """
        Return a copy of the parameter with a new physical value.

        Parameters
        ----------
        value : jnp.ndarray
            The new unscaled physical value to set. It will be mapped through 
            the transform inverse if one exists.

        Returns
        -------
        Parameter
            A copy of this object with `value` updated.
        """
        latent_value = jnp.asarray(value)
        if self.metadata is not None and self.metadata.transform is not None:
            latent_value = self.metadata.transform.inverse(latent_value)        
        return dataclasses.replace(self, latent_value=latent_value)
    
    def with_distribution(self, distribution: AbstractDistribution) -> 'Parameter':
        """
        Return a copy of the parameter with a new distribution.

        Parameters
        ----------
        distribution : distreqx.distributions.AbstractDistribution
            The distribution to associate with this parameter.

        Returns
        -------
        Parameter
            A copy of this object with the `distribution` replaced.

        Raises
        ------
        Exception
            If `distribution` is not a distreqx AbstractDistribution.
        """
        if not isinstance(distribution, AbstractDistribution):
            raise Exception('Only distreqx distributions are supported as parameter distributions')
        
        if self.metadata is None:
            new_meta = ParameterMetadata(distribution=distribution)
        else:
            new_meta = dataclasses.replace(self.metadata, distribution=distribution)
            
        return dataclasses.replace(self, metadata=new_meta)
    
    def with_transform(self, transform: AbstractBijector) -> 'Parameter':
        """
        Return a copy of the parameter with a new transform.

        Parameters
        ----------
        transform : distreqx.bijectors.AbstractBijector
            The transform to associate with this parameter.

        Returns
        -------
        Parameter
            A copy of this object with the `transform` replaced.

        Raises
        ------
        Exception
            If `distribution` is not a distreqx AbstractDistribution.
        """
        if not isinstance(transform, AbstractBijector):
            raise Exception('Only distreqx bijectors are supported as parameter transforms')
        
        if self.metadata is None:
            new_meta = ParameterMetadata(transform=transform)
        else:
            new_meta = dataclasses.replace(self.metadata, transform=transform)
            
        return dataclasses.replace(self, metadata=new_meta)    
    
    def transformed(self, transform: AbstractBijector) -> 'Parameter':
        """
        Return a copy of this parameter transformed.

        This method applies the given transform to the parameter's physical space. 
        It holistically updates the parameter by chaining the new transform with 
        any existing one, transforming the probability distribution, and mapping 
        the bounds. The underlying latent unconstrained value remains unchanged.

        Parameters
        ----------
        transform : distreqx.bijectors.AbstractBijector
            The transform to apply to the parameter's unscaled physical space.

        Returns
        -------
        Parameter
            A dynamically transformed Parameter object.
            
        Raises
        ------
        TypeError
            If the provided transform is not an instance of AbstractBijector.
        """
        if not isinstance(transform, AbstractBijector):
            raise TypeError("The provided transformation must be a distreqx AbstractBijector.")
        if self.latent_value is None:
            raise Exception("Cannot transform a parameter with a None latent value")

        # 1. Transform the distribution
        new_dist = self.distribution
        if new_dist is not None:
            new_dist = Transformed(new_dist, transform)
            
        # 2. Chain the transforms (applied right-to-left: first old, then new)
        old_transform = self.transform
        if old_transform is not None:
            new_transform = Chain([transform, old_transform])
        else:
            new_transform = transform
            
        # 3. Transform the bounds
        new_bounds = self.bounds
        if new_bounds is not None:
            new_bounds = transform.forward(new_bounds)
        
        # 4. Update metadata
        if self.metadata is None:
            new_meta = ParameterMetadata(
                distribution=new_dist,
                transform=new_transform,
                bounds=new_bounds
            )
        else:
            new_meta = dataclasses.replace(
                self.metadata, 
                distribution=new_dist,
                transform=new_transform,
                bounds=new_bounds
            )
            
        # The latent value remains unchanged; the chained transform handles the new physical mapping.
        return dataclasses.replace(self, metadata=new_meta)
    
    def flattened(self, separator='_') -> 'list[Parameter]':
        """
        Flatten the parameter into a list of scalar Parameters.
        
        If the internal parameter is scalar, the list will contain self.
        Otherwise, the parameter is split (de-vectorized) if possible.
        
        Parameters
        ----------
        separator : str, optional
            Separator used for naming split parameters (e.g., name_0), by default '_'.

        Returns
        -------
        list of Parameter
            The list of individual scalar parameters.

        Raises
        ------
        ValueError
            If the list of names does not match the parameter size.
        """
        if self.latent_value.ndim == 0 and not isinstance(self.name, list):
            return [self]
            
        unscaled_physical = self.value
        flat_val = jnp.ravel(unscaled_physical)
        
        if self.distribution is not None:
            if not self.distribution.event_shape:
                dists_split = [self.distribution] * flat_val.size
            else:
                dists_split = split_vectorized_distribution(self.distribution)
        else:
            dists_split = [None] * flat_val.size

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
                fixed=self.fixed, 
                distribution=p, 
                transform=self.transform,
                bounds=self.bounds,
                scale=self.scale, 
                name=n,
                **self.info 
            ) 
            for val, p, n in zip(flat_val, dists_split, flat_names)
        ]
    
    def as_fixed(self) -> 'Parameter':
        """
        Return a copy of the parameter set to fixed.

        Returns
        -------
        Parameter
            A copy with `fixed=True`.
        """
        return dataclasses.replace(self, fixed=True)
    
    def as_free(self) -> 'Parameter':
        """
        Return a copy of the parameter set to free.

        Returns
        -------
        Parameter
            A copy with `fixed=False`.
        """
        return dataclasses.replace(self, fixed=False)
    
    def __array__(self, dtype=None):
        """
        NumPy array interface.

        Returns
        -------
        numpy.ndarray
            The fully scaled and physical space array.
        """
        return jnp.asarray(self.value * self.scale, dtype=dtype)
    
    def __jax_array__(self, dtype=None):
        """
        JAX array interface.

        Returns
        -------
        jnp.ndarray
            The fully scaled and physical space array.
        """
        return jnp.asarray(self.value * self.scale, dtype=dtype)
    
    def __len__(self):
        """
        Get the length of the parameter value.

        Returns
        -------
        int
            `1` for scalars, otherwise `len(latent_value)`.
        """
        if len(self.latent_value.shape) == 0:
            return 1 
        return len(self.latent_value)
    
    def __repr__(self):
        args = [f"value={format_val(self.latent_value)}"]
        
        if self.scale != 1.0:
            args.append(f"scale={self.scale}")
            
        if self.fixed is not False:
            args.append(f"fixed={self.fixed}")
            
        if self.distribution is not None:
            args.append(f"distribution={format_distribution(self.distribution)}")
            
        if self.transform is not None:
            args.append(f"transform={format_transform(self.transform)}")
            
        if self.bounds is not None:
            args.append(f"bounds={format_val(self.bounds)}")
            
        if self.name is not None:
            args.append(f"name={repr(self.name)}")
            
        if self.info:
            for k, v in self.info.items():
                args.append(f"{k}={repr(v)}")
            
        return f"Parameter({', '.join(args)})"
            
    def __add__(self, other):
        """Elementwise addition."""
        return jnp.add(jnp.array(self), jnp.array(other))
    
    def __sub__(self, other):
        """Elementwise subtraction."""
        return jnp.subtract(jnp.array(self), jnp.array(other))
    
    def __mul__(self, other):
        """Elementwise multiplication."""
        return jnp.multiply(jnp.array(self), jnp.array(other))

    def __truediv__(self, other):
        """Elementwise true division."""
        return jnp.divide(jnp.array(self), jnp.array(other))

    def __radd__(self, other):
        """Reflected elementwise addition."""
        return jnp.add(jnp.array(other), jnp.array(self))
    
    def __rsub__(self, other):
        """Reflected elementwise subtraction."""
        return jnp.subtract(jnp.array(other), jnp.array(self))

    def __rmul__(self, other):
        """Reflected elementwise multiplication."""
        return jnp.multiply(jnp.array(other), jnp.array(self))
    
    def __rtruediv__(self, other):
        """Reflected elementwise true division."""
        return jnp.divide(jnp.array(other), jnp.array(self))
    
    def copy(self):
        """
        Return a shallow copy.

        Returns
        -------
        Parameter
            A copied instance.
        """
        return dataclasses.replace(self)
    
    def to_json(self) -> str:
        """
        Serialize the parameter to a JSON string.
        
        Omits any fields that are None or empty to keep the payload lightweight.

        Returns
        -------
        str
            A JSON-formatted string containing the parameter's data.
        """
        d = {
            "fixed": self.fixed,
            "scale": self.scale
        }
        
        if self.latent_value is not None:
            d["value"] = self.value.tolist()
        else:
            d["value"] = None
        
        if self.distribution is not None:
            d["distribution"] = serialize_distribution(self.distribution)
            
        if self.transform is not None:
            d["transform"] = serialize_transform(self.transform)
            
        if self.bounds is not None:
            d["bounds"] = self.bounds.tolist()
            
        if self.name is not None:
            d["name"] = self.name
            
        if self.info: 
            d["info"] = self.info 
            
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Parameter":
        """
        Deserialize a parameter from a JSON string.

        Parameters
        ----------
        s : str
            The JSON string produced by `to_json`.

        Returns
        -------
        Parameter
            A reconstructed `Parameter` instance.
        """
        d = json.loads(s)
        
        value = d.pop("value", None)
        fixed = d.pop("fixed", False)
        
        if "distribution" in d:
            d["distribution"] = deserialize_distribution(d["distribution"])
            
        if "transform" in d:
            d["transform"] = deserialize_transform(d["transform"])
            
        info_dict = d.pop("info", {})
        d.update(info_dict)
        
        return cls(value=value, fixed=fixed, **d)
    

def is_param(x: Any) -> bool:
    """
    Check if an object is an instance of a Parameter.

    Parameters
    ----------
    x : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is a Parameter, False otherwise.
    """
    return isinstance(x, Parameter)

def is_valid_param(x: Any) -> bool:
    """
    Check if an object is a Parameter and has a valid latent value.

    Parameters
    ----------
    x : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is a Parameter with a non-None value, False otherwise.
    """
    return isinstance(x, Parameter) and x.latent_value is not None

def is_free_param(x: Any) -> bool:
    """
    Check if an object is a free (unfixed) Parameter.

    Parameters
    ----------
    x : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is a non-fixed Parameter, False otherwise.
    """
    return is_valid_param(x) and not x.fixed

def is_fixed_param(x: Any) -> bool:
    """
    Check if an object is a fixed Parameter.

    Parameters
    ----------
    x : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is a fixed Parameter, False otherwise.
    """
    return is_valid_param(x) and x.fixed

def as_param(x: Any | list[Any] | dict[str, Any], **kwargs) -> "Parameter":
    """
    Ensure an object is a Parameter or container of Parameters.
    
    If the object is already a Parameter, it is returned unchanged. Otherwise, 
    the underlying objects are converted into new Parameter objects.

    Parameters
    ----------
    x : Any, list, or dict
        The object to convert.
    **kwargs
        Additional keyword arguments passed to the Parameter constructor.

    Returns
    -------
    Parameter, list of Parameter, or dict of Parameter
        The object wrapped as Parameter(s).
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