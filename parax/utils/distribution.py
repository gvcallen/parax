import jax
import jax.numpy as jnp
import numpy as np

import distreqx.distributions as dist
from distreqx.distributions import AbstractDistribution

from parax.utils.array import format_array

def serialize_distribution(d: AbstractDistribution | None) -> dict | None:
    """
    Serialize a distreqx distribution to a lightweight dictionary.
    """
    if d is None:
        return None
        
    params = {}
    # Since distreqx uses Equinox, distributions are dataclass-like.
    # vars() cleanly iterates over the initialized properties.
    for k, v in vars(d).items():
        if k.startswith("_"):
            continue
            
        if isinstance(v, (jax.Array, jnp.ndarray, np.ndarray)):
            params[k] = v.tolist()
        elif isinstance(v, AbstractDistribution):
            # Recursively handle nested distributions (e.g., in an Independent wrapper)
            params[k] = serialize_distribution(v)
        else:
            # Fallback for primitives (int, str, float)
            params[k] = v
            
    return {"class": d.__class__.__name__, "params": params}

def deserialize_distribution(dct: dict | None) -> AbstractDistribution | None:
    """
    Deserialize a distreqx distribution from a dictionary.
    """
    if dct is None:
        return None
        
    cls = getattr(dist, dct["class"], None)
    if cls is None:
        raise ValueError(f"Unknown distribution class: {dct['class']}")
        
    params = dct["params"].copy()
    for k, v in params.items():
        if isinstance(v, dict) and "class" in v:
            # Reconstruct nested distribution
            params[k] = deserialize_distribution(v)
        else:
            params[k] = jnp.asarray(v)
            
    return cls(**params)

def format_distribution(d: AbstractDistribution) -> str:
    """Format a distreqx distribution dynamically."""
    class_name = d.__class__.__name__
    args = []
    
    # Iterate dynamically over Equinox module fields
    for k, v in vars(d).items():
        if k.startswith("_"):
            continue
        args.append(f"{k}={format_array(v)}")
        
    if args:
        return f"{class_name}({', '.join(args)})"
    return repr(d)


def hypercube_to_physical(d: dist.AbstractDistribution, u: jnp.ndarray) -> jnp.ndarray:
    """
    Maps a vector `u` from the unit hypercube [0, 1]^d to the target parameter space.
    (Commonly used as the Prior Transform in nested sampling).
    """
    if hasattr(d, 'icdf') and callable(d.icdf):
        return d.icdf(u)
    elif isinstance(d, dist.Transformed):
        base_x = hypercube_to_physical(d.distribution, u)
        return d.bijector.forward(base_x) 
    else:
        raise NotImplementedError(
            f"Analytical hypercube mapping is not yet supported for {type(d)}. "
            f"Ensure the distribution has a .icdf() method or is wrapped in a Bijector."
        )

def physical_to_hypercube(d: dist.AbstractDistribution, x: jnp.ndarray) -> jnp.ndarray:
    """
    Maps a vector `x` from the physical parameter space back to the unit hypercube [0, 1]^d.
    (The mathematical inverse of the hypercube_to_physical mapping).
    """
    if hasattr(d, 'cdf') and callable(d.cdf):
        return d.cdf(x)
    elif isinstance(d, dist.Transformed):
        base_x = d.bijector.inverse(x)
        return physical_to_hypercube(d.distribution, base_x)
    else:
        raise NotImplementedError(
            f"Analytical physical-to-hypercube mapping is not supported for {type(d)}. "
            f"Ensure the distribution has a .cdf() method or is wrapped in a Bijector."
        )