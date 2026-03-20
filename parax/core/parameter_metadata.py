import jax.numpy as jnp
import equinox as eqx

from numpyro.distributions.transforms import Transform
from numpyro.distributions.distribution import Distribution

from parax.core.field import field

class ParameterMetadata(eqx.Module):
    """
    Hidden struct to hold all parameter metadata. 
    Keeps the core Parameter class lightweight for basic users.
    """
    name: str | list[str] | None = field(default=None, static=True)
    description: str | None = field(default=None, static=True)
    units: str | None = field(default=None, static=True)
    
    distribution: Distribution | None = field(default=None)
    transform: Transform | None = field(default=None, static=True)
    bounds: jnp.ndarray | None = field(default=None)
    scale: float = field(default=1.0, static=True)
    
    info: dict = field(default_factory=dict, static=True)