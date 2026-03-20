import jax.numpy as jnp
import numpyro.distributions.transforms as transforms
from numpyro.distributions.transforms import Transform, ComposeTransform
from numpyro.distributions import constraints

from parax.utils.string import format_val

def serialize_transform(t: Transform | None) -> dict | None:
    r"""
    Serialize a numpyro transform to a lightweight dictionary.

    Parameters
    ----------
    t : numpyro.distributions.transforms.Transform or None
        The transform to serialize.

    Returns
    -------
    dict or None
        A dictionary with ``class`` and ``params`` keys, or ``None``.
    """
    if t is None:
        return None
        
    # Special handling for composed transforms which nest other transforms
    if isinstance(t, ComposeTransform):
        return {
            "class": "ComposeTransform",
            "parts": [serialize_transform(p) for p in t.parts]
        }
        
    params = {}
    for k, v in t.__dict__.items():
        # Skip private attributes and cached inverses
        if k.startswith("_") or k == "inv":
            continue
            
        if isinstance(v, jnp.ndarray):
            params[k] = v.tolist()
        elif isinstance(v, constraints.Constraint):
            # Just store the name of the constraint (e.g., '_Real', '_Positive')
            params[k] = {"__constraint__": type(v).__name__}
        elif isinstance(v, Transform):
            # Recursively serialize nested transforms if they exist
            params[k] = serialize_transform(v)
        else:
            params[k] = v
            
    return {"class": t.__class__.__name__, "params": params}


def deserialize_transform(dct: dict | None) -> Transform | None:
    r"""
    Deserialize a numpyro transform from a dictionary.

    Parameters
    ----------
    dct : dict or None
        A dictionary produced by :func:`serialize_transform`.

    Returns
    -------
    numpyro.distributions.transforms.Transform or None
        The reconstructed transform, or ``None``.
    
    Raises
    ------
    ValueError
        If the transform class is unknown.
    """
    if dct is None:
        return None
        
    cls_name = dct["class"]
    
    # Special handling for composed transforms
    if cls_name == "ComposeTransform":
        parts = [deserialize_transform(p) for p in dct.get("parts", [])]
        return ComposeTransform(parts)
        
    cls = getattr(transforms, cls_name, None)
    if cls is None:
        raise ValueError(f"Unknown transform class: {cls_name}")
        
    params = dct.get("params", {})
    for k, v in params.items():
        if isinstance(v, dict):
            if "__constraint__" in v:
                # Map the string name back to the numpyro constraint object
                constraint_name = v["__constraint__"].lower() # e.g. '_Real' -> 'real'
                constraint_name = constraint_name.strip('_') 
                params[k] = getattr(constraints, constraint_name)
            elif "class" in v:
                # Recursively deserialize nested transforms
                params[k] = deserialize_transform(v)
                
    return cls(**params)


def format_transform(t: Transform) -> str:
    """
    Format a numpyro transform dynamically.
    """
    if isinstance(t, ComposeTransform):
        parts_str = ", ".join([format_transform(p) for p in t.parts])
        return f"ComposeTransform([{parts_str}])"
        
    class_name = t.__class__.__name__
    args = []
    
    # Transforms don't have an `arg_constraints` dictionary like distributions do,
    # so we explicitly check for the most common defining mathematical properties.
    for param_name in ["loc", "scale", "base", "concentration0", "concentration1"]:
        if hasattr(t, param_name):
            val = getattr(t, param_name)
            args.append(f"{param_name}={format_val(val)}")
            
    if args:
        return f"{class_name}({', '.join(args)})"
        
    # Fallback for transforms with no standard parameters (e.g., ExpTransform)
    return f"{class_name}()"