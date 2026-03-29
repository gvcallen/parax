import inspect
import importlib
import builtins
from typing import Any

import jax.numpy as jnp
from distreqx import bijectors
from distreqx.bijectors import AbstractBijector, Chain

from parax.utils.string import format_val

def serialize_transform(t: Any | None) -> dict | None:
    r"""
    Serialize an arbitrary transform (bijector or callable) to a lightweight dictionary.

    Parameters
    ----------
    t : Any or None
        The transform to serialize. Can be a distreqx bijector, a standard function,
        or a stateful callable class.

    Returns
    -------
    dict or None
        A dictionary with ``class``, ``module``, and ``params`` keys, or ``None``.
        
    Raises
    ------
    ValueError
        If attempting to serialize an anonymous lambda function.
    """
    if t is None:
        return None
        
    # Special handling for composed bijectors which nest other bijectors
    if isinstance(t, Chain):
        return {
            "class": "Chain",
            "module": "distreqx.bijectors",
            "transforms": [serialize_transform(p) for p in t.bijectors]
        }
        
    # Handle stateless functions / standard callables
    if inspect.isfunction(t) or inspect.isbuiltin(t):
        name = getattr(t, "__name__", "unknown")
        if name == "<lambda>":
            raise ValueError("Cannot serialize lambda functions. Please use named functions.")
            
        return {
            "class": "function",
            "module": getattr(t, "__module__", "builtins"),
            "name": name
        }
        
    # Handle stateful instances (like distreqx Bijectors or custom callable classes)
    params = {}
    if hasattr(t, '__dict__'):
        for k, v in t.__dict__.items():
            # Skip private attributes and cached inverses
            if k.startswith("_") or k == "inv":
                continue
                
            if isinstance(v, jnp.ndarray):
                params[k] = v.tolist()
            elif isinstance(v, AbstractBijector) or callable(v):
                # Recursively serialize nested transforms if they exist
                params[k] = serialize_transform(v)
            else:
                params[k] = v
            
    return {
        "class": t.__class__.__name__, 
        "module": t.__class__.__module__,
        "params": params
    }


def deserialize_transform(dct: dict | None) -> Any | None:
    r"""
    Deserialize a transform from a dictionary.

    Parameters
    ----------
    dct : dict or None
        A dictionary produced by [`serialize_transform`][].

    Returns
    -------
    Any or None
        The reconstructed transform (bijector, function, or callable), or ``None``.
    
    Raises
    ------
    ValueError
        If the transform class or function cannot be located.
    """
    if dct is None:
        return None
        
    cls_name = dct["class"]
    mod_name = dct.get("module", "builtins")
    
    # Special handling for composed bijectors
    if cls_name == "Chain":
        parts = [deserialize_transform(p) for p in dct.get("transforms", [])]
        return Chain(parts)
        
    # Handle stateless functions
    if cls_name == "function":
        func_name = dct.get("name")
        if mod_name == "builtins":
            return getattr(builtins, func_name)
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, func_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Could not deserialize function {func_name} from {mod_name}") from e

    # Handle stateful instances (Bijectors / Custom Callables)
    cls = None
    if mod_name and mod_name != "builtins":
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
        except (ImportError, AttributeError):
            pass
            
    # Fallback to distreqx bijectors if dynamic import failed
    if cls is None:
        cls = getattr(bijectors, cls_name, None)
        
    if cls is None:
        raise ValueError(f"Unknown transform class: {cls_name} in module {mod_name}")
        
    params = dct.get("params", {})
    for k, v in params.items():
        if isinstance(v, dict) and "class" in v:
            # Recursively deserialize nested transforms
            params[k] = deserialize_transform(v)
                
    return cls(**params)


def format_transform(t: Any) -> str:
    """
    Format a transform or bijector dynamically for string representation.
    """
    if isinstance(t, Chain):
        parts_str = ", ".join([format_transform(p) for p in t.bijectors])
        return f"Chain([{parts_str}])"
        
    if inspect.isfunction(t) or inspect.isbuiltin(t):
        return getattr(t, "__name__", "callable")
        
    class_name = t.__class__.__name__
    args = []
    
    # AbstractBijectors and custom classes might not have standard constraints dicts,
    # so we explicitly check for the most common defining mathematical properties.
    for param_name in ["shift", "scale", "loc", "base", "concentration0", "concentration1"]:
        if hasattr(t, param_name):
            val = getattr(t, param_name)
            args.append(f"{param_name}={format_val(val)}")
            
    if args:
        return f"{class_name}({', '.join(args)})"
        
    # Fallback for transforms with no standard parameters (e.g., Exp)
    return f"{class_name}()"