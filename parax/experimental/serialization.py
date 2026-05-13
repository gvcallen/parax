import importlib
from typing import Any
import dataclasses

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree

def _serialize_generic(node: Any) -> Any:
    """Recursively converts any PyTree/Equinox node into a JSON-serializable dict."""
    # 1. Base Case: Standard Python primitives
    if isinstance(node, (int, float, str, bool, type(None))):
        return node
        
    # NEW: Handle native Python complex numbers
    if isinstance(node, complex):
        return {
            "__type__": "complex",
            "real": node.real,
            "imag": node.imag
        }
        
    # 2. Dynamic Nodes: Equinox Modules
    if hasattr(node, "__dataclass_fields__"):
        state = {}
        # Notice we are using .items() now to get the Field object
        for field_name, field in node.__dataclass_fields__.items():
            val = getattr(node, field_name)
            
            # --- THE FIX ---
            # Compare the current value against the field's explicit default.
            # If it perfectly matches, we skip saving it entirely!
            if field.default is not dataclasses.MISSING:
                try:
                    # try/except because comparing JAX arrays can sometimes raise errors
                    if val == field.default:
                        continue 
                except Exception:
                    pass
            
            # Also handle default_factories (e.g., default_factory=list)
            if field.default_factory is not dataclasses.MISSING:
                try:
                    if val == field.default_factory():
                        continue
                except Exception:
                    pass
            # ---------------
                    
            state[field_name] = _serialize_generic(val)
                
        return {
            "__type__": "dynamic_node",
            "__module__": type(node).__module__,
            "__class__": type(node).__name__,
            "state": state
        }
        
    # 3. Base Case: Pure Arrays (jax.Array, np.ndarray, etc.)
    if eqx.is_array_like(node):
        arr = jnp.asarray(node)
        
        # NEW: Handle complex arrays by splitting real and imaginary parts
        if jnp.iscomplexobj(arr):
            return {
                "__type__": "complex_array",
                "dtype": str(getattr(arr, "dtype", "complex64")),
                "real": arr.real.tolist(),
                "imag": arr.imag.tolist()
            }
        # Handle standard real arrays
        else:
            return {
                "__type__": "array",
                "dtype": str(getattr(arr, "dtype", "float32")),
                "data": arr.tolist()
            }
        
    # 4. Standard Containers
    if isinstance(node, dict):
        return {k: _serialize_generic(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_serialize_generic(x) for x in node]
        
    raise TypeError(
        f"Cannot serialize {type(node)}. Ensure all custom classes are wrapped in eqx.Module."
    )

def _deserialize_generic(data: Any) -> Any:
    """Recursively reconstructs PyTrees/Equinox modules from a JSON dict."""
    
    if isinstance(data, (int, float, str, bool, type(None))):
        return data
        
    if isinstance(data, list):
        return [_deserialize_generic(x) for x in data]
        
    if isinstance(data, dict):
        
        # NEW: Rebuild native complex number
        if data.get("__type__") == "complex":
            return complex(data["real"], data["imag"])
            
        # NEW: Rebuild complex array
        if data.get("__type__") == "complex_array":
            real_part = jnp.array(data["real"])
            imag_part = jnp.array(data["imag"])
            return jnp.array(real_part + 1j * imag_part, dtype=data["dtype"])

        # Rebuild standard Array
        if data.get("__type__") == "array":
            return jnp.array(data["data"], dtype=data["dtype"])
            
        # Rebuild Dynamic Equinox Module
        if data.get("__type__") == "dynamic_node":
            module = importlib.import_module(data["__module__"])
            cls = getattr(module, data["__class__"])
            
            instance = object.__new__(cls) 
            
            for field_name, field_data in data["state"].items():
                val = _deserialize_generic(field_data)
                object.__setattr__(instance, field_name, val)
                
            return instance
            
        # Standard dictionary
        return {k: _deserialize_generic(v) for k, v in data.items()}


def save(filepath: str, model: PyTree) -> None:
    import json
    serialized_tree = _serialize_generic(model)
    with open(filepath, "w") as f:
        json.dump(serialized_tree, f, indent=4)

def load(filepath: str) -> PyTree:
    import json
    with open(filepath, "r") as f:
        serialized_tree = json.load(f)
    return _deserialize_generic(serialized_tree)