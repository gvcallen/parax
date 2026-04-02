import jax.numpy as jnp
import equinox as eqx
from distreqx.distributions import AbstractDistribution
from distreqx.bijectors import AbstractBijector

from parax.field import field

class ParameterMetadata(eqx.Module):
    """
    Hidden struct to hold all parameter metadata. 

    This keeps the core `Parameter` class lightweight for basic users 
    by compartmentalizing the extended properties that `parax` interacts with. 
    It also contains an `info` field to store arbitrary user-defined metadata.

    Attributes
    ----------
    name : str, list, or None
        The identifier(s) for the parameter. Must either be a single string
        or a list matching the shape of the underlying array.
    distribution : distreqx.distributions.AbstractDistribution or None
        The probability distribution associated with the parameter
        in unscaled physical space.
    transform : distreqx.bijectors.AbstractBijector or None
        The transform used to map from the latent space to the unscaled physical space.
    bounds : jnp.ndarray or None
        The boundaries of the parameter in unscaled physical space.
        Can be used as hard constraints for bounded optimizers.
    scale : float
        A scalar multiplier applied to the unscaled physical value to convert it
        to a JAX array to be used in calculations. Defaults to 1.0.
    info : dict
        A dictionary for storing additional, arbitrary user-defined metadata. 
        Marked as static.
    """
    name: str | list | None = field(default=None, static=True)
    
    distribution: AbstractDistribution | None = field(default=None)
    transform: AbstractBijector | None = field(default=None)
    bounds: jnp.ndarray | None = field(default=None)
    scale: float = field(default=1.0)
    
    info: dict = field(default_factory=dict, static=True)