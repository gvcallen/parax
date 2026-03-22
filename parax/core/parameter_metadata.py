import jax.numpy as jnp
import equinox as eqx
from distreqx.distributions import AbstractDistribution
from distreqx.bijectors import AbstractBijector

from parax.core.field import field

class ParameterMetadata(eqx.Module):
    """
    Hidden struct to hold all parameter metadata. 
    Keeps the core Parameter class lightweight for basic users.
    Adds metadata only that `parax` interacts with.
    Contains an additional `info` field for additional user metadata.
    """
    name: str | list[str] | None = field(default=None, static=True)
    
    distribution: AbstractDistribution | None = field(default=None)
    bijector: AbstractBijector | None = field(default=None)
    scale: float = field(default=1.0)
    bounds: jnp.ndarray | None = field(default=None)
    
    info: dict = field(default_factory=dict, static=True)