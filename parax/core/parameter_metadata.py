import jax.numpy as jnp
import equinox as eqx
from distreqx.distributions import AbstractDistribution
from distreqx.bijectors import AbstractBijector

from parax.core.field import field

class ParameterMetadata(eqx.Module):
    """
    Hidden struct to hold all parameter metadata. 

    This keeps the core `Parameter` class lightweight for basic users 
    by compartmentalizing the extended properties that `parax` interacts with. 
    It also contains an `info` field to store arbitrary user-defined metadata.

    Attributes
    ----------
    name : str, list of str, or None
        The identifier(s) for the parameter. Marked as static so it is 
        ignored during JAX transformations.
    distribution : distreqx.distributions.AbstractDistribution or None
        The probability distribution associated with the parameter.
    bijector : distreqx.bijectors.AbstractBijector or None
        The bijector used to map between the unconstrained latent space 
        and the constrained physical space.
    scale : float
        A scalar multiplier applied to the physical value. Defaults to 1.0.
    bounds : jnp.ndarray or None
        The numeric physical boundaries of the parameter.
    info : dict
        A dictionary for storing additional, arbitrary user-defined metadata. 
        Marked as static.
    """
    name: str | list[str] | None = field(default=None, static=True)
    
    distribution: AbstractDistribution | None = field(default=None)
    bijector: AbstractBijector | None = field(default=None)
    scale: float = field(default=1.0)
    bounds: jnp.ndarray | None = field(default=None)
    
    info: dict = field(default_factory=dict, static=True)