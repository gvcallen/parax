import jax.numpy as jnp
from distreqx import bijectors
from distreqx.bijectors import AbstractBijector, Chain

from parax.utils.string import format_val

def serialize_bijector(b: AbstractBijector | None) -> dict | None:
    r"""
    Serialize a distreqx bijector to a lightweight dictionary.

    Parameters
    ----------
    b : distreqx.bijectors.AbstractBijector or None
        The bijector to serialize.

    Returns
    -------
    dict or None
        A dictionary with ``class`` and ``params`` keys, or ``None``.
    """
    if b is None:
        return None
        
    # Special handling for composed bijectors which nest other bijectors
    if isinstance(b, Chain):
        return {
            "class": "Chain",
            "bijectors": [serialize_bijector(p) for p in b.bijectors]
        }
        
    params = {}
    for k, v in b.__dict__.items():
        # Skip private attributes and cached inverses
        if k.startswith("_") or k == "inv":
            continue
            
        if isinstance(v, jnp.ndarray):
            params[k] = v.tolist()
        elif isinstance(v, AbstractBijector):
            # Recursively serialize nested bijectors if they exist
            params[k] = serialize_bijector(v)
        else:
            params[k] = v
            
    return {"class": b.__class__.__name__, "params": params}


def deserialize_bijector(dct: dict | None) -> AbstractBijector | None:
    r"""
    Deserialize a distreqx bijector from a dictionary.

    Parameters
    ----------
    dct : dict or None
        A dictionary produced by [`serialize_bijector`][].

    Returns
    -------
    distreqx.bijectors.AbstractBijector or None
        The reconstructed bijector, or ``None``.
    
    Raises
    ------
    ValueError
        If the bijector class is unknown.
    """
    if dct is None:
        return None
        
    cls_name = dct["class"]
    
    # Special handling for composed bijectors
    if cls_name == "Chain":
        parts = [deserialize_bijector(p) for p in dct.get("bijectors", [])]
        return Chain(parts)
        
    cls = getattr(bijectors, cls_name, None)
    if cls is None:
        raise ValueError(f"Unknown bijector class: {cls_name}")
        
    params = dct.get("params", {})
    for k, v in params.items():
        if isinstance(v, dict):
            if "class" in v:
                # Recursively deserialize nested bijectors
                params[k] = deserialize_bijector(v)
                
    return cls(**params)


def format_bijector(b: AbstractBijector) -> str:
    """
    Format a distreqx bijector dynamically.
    """
    if isinstance(b, Chain):
        parts_str = ", ".join([format_bijector(p) for p in b.bijectors])
        return f"Chain([{parts_str}])"
        
    class_name = b.__class__.__name__
    args = []
    
    # AbstractBijectors don't have standard constraints dicts,
    # so we explicitly check for the most common defining mathematical properties.
    for param_name in ["shift", "scale", "loc", "base", "concentration0", "concentration1"]:
        if hasattr(b, param_name):
            val = getattr(b, param_name)
            args.append(f"{param_name}={format_val(val)}")
            
    if args:
        return f"{class_name}({', '.join(args)})"
        
    # Fallback for bijectors with no standard parameters (e.g., Exp)
    return f"{class_name}()"