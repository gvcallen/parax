import jax
import jax.numpy as jnp
import numpy as np

import distreqx.distributions as dists
from distreqx.distributions import AbstractDistribution

from parax.utils.array import format_array

def split_vectorized_distribution(d: AbstractDistribution) -> list[AbstractDistribution]:
    """
    Split an arbitrarily shaped batch of univariate distreqx distributions into a list of scalar distributions.
    
    Handles broadcasting of distribution parameters to the implicit batch shape.
    
    Parameters
    ----------
    d : distreqx.distributions.AbstractDistribution
        A distribution with a scalar event shape and arbitrary implicit batch shape.

    Returns
    -------
    list[distreqx.distributions.AbstractDistribution]
        A flat list of scalar distributions corresponding to the flattened batch.
    """
    # Verify scalar event shape (if explicitly defined by the distreqx distribution)
    if hasattr(d, "event_shape") and d.event_shape != ():
        raise ValueError(f"Cannot split distribution with event_shape={d.event_shape} (likely Multivariate)")

    # 1. Inspect the PyTree leaves to find the implicit batch shape
    leaves = jax.tree_util.tree_leaves(d)
    
    def is_broadcastable(leaf):
        return isinstance(leaf, (jax.Array, jnp.ndarray, np.ndarray, float, int, bool))
    
    arrays = [l for l in leaves if is_broadcastable(l)]
    
    if not arrays:
        return [d]

    # 2. Determine the broadcasted batch shape from the distribution's parameters
    try:
        batch_shape = jnp.broadcast_shapes(*[jnp.shape(a) for a in arrays])
    except ValueError as e:
        raise ValueError("Distribution parameters cannot be broadcasted to a common batch shape.") from e

    if not batch_shape:  # It's already a pure scalar distribution
        return [d]

    total_size = int(jnp.prod(jnp.array(batch_shape)))

    # 3. Broadcast and flatten all valid arrays/numbers within the PyTree
    def broadcast_and_flatten(leaf):
        if is_broadcastable(leaf):
            return jnp.ravel(jnp.broadcast_to(leaf, batch_shape))
        return leaf  # Leave static configurations untouched

    flat_d = jax.tree_util.tree_map(broadcast_and_flatten, d)

    # 4. Reconstruct individual scalar distributions via PyTree unstacking
    split_dists = []
    for i in range(total_size):
        scalar_d = jax.tree_util.tree_map(
            lambda leaf: leaf[i] if isinstance(leaf, (jax.Array, jnp.ndarray, np.ndarray)) else leaf, 
            flat_d
        )
        split_dists.append(scalar_d)

    return split_dists

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
        
    cls = getattr(dists, dct["class"], None)
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