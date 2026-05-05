import importlib
import json
from typing import Any

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree

def _serialize_generic(node: Any) -> Any:
    """Recursively converts any PyTree/Equinox node into a JSON-serializable dict."""
    
    # 1. Base Case: Standard Python primitives
    if isinstance(node, (int, float, str, bool, type(None))):
        return node
        
    # 2. Dynamic Nodes: Equinox Modules (This MUST come before array-likes)
    # This prevents Parax variables (which implement __jax_array__) from 
    # being accidentally swallowed as raw arrays.
    if hasattr(node, "__dataclass_fields__"):
        state = {}
        for field_name in node.__dataclass_fields__:
            val = getattr(node, field_name)
            state[field_name] = _serialize_generic(val)
            
        return {
            "__type__": "dynamic_node",
            "__module__": type(node).__module__,
            "__class__": type(node).__name__,
            "state": state
        }
        
    # 3. Base Case: Pure Arrays (jax.Array, np.ndarray, etc.)
    if eqx.is_array_like(node):
        # jnp.asarray safely strips JAX tracers/wrappers and ensures .tolist() exists
        arr = jnp.asarray(node)
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
        # Rebuild Array
        if data.get("__type__") == "array":
            return jnp.array(data["data"], dtype=data["dtype"])
            
        # Rebuild Dynamic Equinox Module
        if data.get("__type__") == "dynamic_node":
            module = importlib.import_module(data["__module__"])
            cls = getattr(module, data["__class__"])
            
            # THE TRICK: Bypass __init__ to prevent argument signature mismatches.
            # Create an uninitialized instance of the class in memory.
            instance = object.__new__(cls) 
            
            # Recursively deserialize all internal fields
            for field_name, field_data in data["state"].items():
                val = _deserialize_generic(field_data)
                
                # Equinox modules are frozen. We must use object.__setattr__ 
                # to forcefully inject the state into the husk.
                object.__setattr__(instance, field_name, val)
                
            return instance
            
        # Standard dictionary
        return {k: _deserialize_generic(v) for k, v in data.items()}


def save(model: PyTree, filepath: str) -> None:
    serialized_tree = _serialize_generic(model)
    with open(filepath, "w") as f:
        json.dump(serialized_tree, f, indent=4)

def load(filepath: str) -> PyTree:
    with open(filepath, "r") as f:
        serialized_tree = json.load(f)
    return _deserialize_generic(serialized_tree)