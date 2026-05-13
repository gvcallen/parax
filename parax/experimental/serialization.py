import importlib
import json
import dataclasses
from typing import Any

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree

def _serialize_generic(node: Any) -> Any:
    """Recursively converts any PyTree/Equinox node into a JSON-serializable dict.
    Falls back to jsonpickle for unregistered or non-standard objects if installed.
    """
    # 1. Base Case: Standard Python primitives
    if isinstance(node, (int, float, str, bool, type(None))):
        return node
        
    # 2. Base Case: Native Python complex numbers
    if isinstance(node, complex):
        return {
            "__type__": "__complex__",
            "__real__": node.real,
            "__imag__": node.imag
        }
        
    # 3. Dynamic Nodes: Equinox Modules (Dataclasses)
    if hasattr(node, "__dataclass_fields__"):
        state = {}
        for field_name, field in node.__dataclass_fields__.items():
            val = getattr(node, field_name)
            
            # Skip saving if the value exactly matches the field's explicit default
            if field.default is not dataclasses.MISSING:
                try:
                    if val == field.default:
                        continue 
                except Exception:
                    pass
            
            # Skip saving if the value matches the default_factory output
            if field.default_factory is not dataclasses.MISSING:
                try:
                    if val == field.default_factory():
                        continue
                except Exception:
                    pass
                    
            state[field_name] = _serialize_generic(val)
                
        return {
            "__type__": "__dynamic_node__",
            "__module__": type(node).__module__,
            "__class__": type(node).__name__,
            "__state__": state
        }
        
    # 4. Base Case: Pure Arrays (jax.Array, np.ndarray, etc.)
    if eqx.is_array_like(node):
        arr = jnp.asarray(node)
        
        # Handle complex arrays
        if jnp.iscomplexobj(arr):
            return {
                "__type__": "__complex_array__",
                "__dtype__": str(getattr(arr, "dtype", "complex64")),
                "__real__": arr.real.tolist(),
                "__imag__": arr.imag.tolist()
            }
        # Handle standard real arrays
        else:
            return {
                "__type__": "__array__",
                "__dtype__": str(getattr(arr, "dtype", "float32")),
                "__data__": arr.tolist()
            }
        
    # 5. Standard Containers
    if isinstance(node, dict):
        return {k: _serialize_generic(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_serialize_generic(x) for x in node]
        
    # 6. Fallback: jsonpickle for arbitrary objects
    try:
        import jsonpickle
    except ImportError:
        raise TypeError(
            f"Cannot serialize object of type {type(node)}.\n"
            "Ensure all custom classes are wrapped in `eqx.Module`, or install "
            "`jsonpickle` (`pip install jsonpickle`) to enable automatic fallback serialization."
        )

    try:
        pickled_string = jsonpickle.encode(node)
        return {
            "__type__": "__jsonpickle__",
            "__payload__": json.loads(pickled_string) 
        }
    except Exception as e:
        raise TypeError(
            f"Cannot serialize {type(node)}. Custom traversal failed, "
            f"and jsonpickle fallback also raised an error: {e}"
        )

def _deserialize_generic(data: Any) -> Any:
    """Recursively reconstructs PyTrees/Equinox modules from a JSON dict."""
    
    if isinstance(data, (int, float, str, bool, type(None))):
        return data
        
    if isinstance(data, list):
        return [_deserialize_generic(x) for x in data]
        
    if isinstance(data, dict):
        node_type = data.get("__type__")
        
        # Rebuild native complex number
        if node_type == "__complex__":
            return complex(data["__real__"], data["__imag__"])
            
        # Rebuild complex array
        if node_type == "__complex_array__":
            real_part = jnp.array(data["__real__"])
            imag_part = jnp.array(data["__imag__"])
            return jnp.array(real_part + 1j * imag_part, dtype=data["__dtype__"])

        # Rebuild standard Array
        if node_type == "__array__":
            return jnp.array(data["__data__"], dtype=data["__dtype__"])
            
        # Rebuild Dynamic Equinox Module
        if node_type == "__dynamic_node__":
            module = importlib.import_module(data["__module__"])
            cls = getattr(module, data["__class__"])
            
            instance = object.__new__(cls) 
            
            for field_name, field_data in data["__state__"].items():
                val = _deserialize_generic(field_data)
                object.__setattr__(instance, field_name, val)
                
            return instance

        # Rebuild jsonpickle fallback
        if node_type == "__jsonpickle__":
            try:
                import jsonpickle
            except ImportError:
                raise ImportError(
                    "This model contains a node serialized with `jsonpickle`, but the library "
                    "is not currently installed. Please run `pip install jsonpickle` to load this file."
                )
            
            pickled_string = json.dumps(data["__payload__"])
            return jsonpickle.decode(pickled_string)
            
        # Standard dictionary
        return {k: _deserialize_generic(v) for k, v in data.items()}

def save(filepath: str, model: PyTree) -> None:
    """Serializes an Equinox model or PyTree and saves it to a JSON file."""
    serialized_tree = _serialize_generic(model)
    with open(filepath, "w") as f:
        json.dump(serialized_tree, f, indent=4)

def load(filepath: str) -> PyTree:
    """Loads a serialized Equinox model or PyTree from a JSON file."""
    with open(filepath, "r") as f:
        serialized_tree = json.load(f)
    return _deserialize_generic(serialized_tree)