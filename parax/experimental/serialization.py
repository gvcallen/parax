"""
(experimental) IO helpers e.g. for module loading and saving.
"""
import os
import json
from typing import BinaryIO, Any

import jax.tree_util as jtu
import jsonpickle
import jsonpickle.handlers

from parax.constrained import Param
import equinox as eqx

def _eqx_getstate(self):
    """Strip out JAX's mangled ghosts before serialization."""
    return {
        k: v for k, v in self.__dict__.items()
        if not k.startswith(f"_{self.__class__.__name__}__") and k != "__orig_class__"
    }

def _eqx_setstate(self, state):
    """Bypass Equinox's frozen lock during deserialization."""
    for k, v in state.items():
        if not k.startswith("py/"):
            object.__setattr__(self, k, v)

# Patch Equinox modules so jsonpickle handles them smoothly
eqx.Module.__getstate__ = _eqx_getstate
eqx.Module.__setstate__ = _eqx_setstate


class ParameterHandler(jsonpickle.handlers.BaseHandler):
    """
    Custom jsonpickle handler to explicitly use parax.Parameter's 
    to_json() and from_json() serialization logic.
    """
    def flatten(self, obj: Param, data: dict) -> dict:
        try:
            # 1. Attempt to use the class's native JSON serialization
            param_dict = json.loads(obj.to_json())
            data.update(param_dict)
            
        except ValueError as e:
            # 2. Catch the JAX 'None' crash caused by uninitialized parameters
            if "None is not a valid value" in str(e):
                # Fallback: manually serialize the object's __dict__, bypassing .to_json()
                for k, v in obj.__dict__.items():
                    data[k] = self.context.flatten(v, reset=False)
            else:
                raise e
                
        return data

    def restore(self, data: dict) -> Param:
        # If it fell back to __dict__ serialization, we might need to handle it natively
        clean_data = {k: v for k, v in data.items() if not k.startswith("py/")}
        
        try:
            json_str = json.dumps(clean_data)
            return Param.from_json(json_str)
        except Exception:
            # Fallback if from_json fails on the natively serialized dict
            param = Param.__new__(Param)
            param.__dict__.update(clean_data)
            return param

# Register the custom handler with jsonpickle globally
ParameterHandler.handles(Param)


def load(source: str | os.PathLike | BinaryIO) -> Any:
    """
    Load a Parax PyTree (e.g., a Module, or a dict/list of Modules) from a file.

    Parameters
    ----------
    source : str, os.PathLike, or BinaryIO
        The path to the saved file or an open file-like object containing the data.

    Returns
    -------
    Any
        The deserialized PyTree (Module, dict, list, etc.).
        
    Raises
    ------
    TypeError
        If the root object or any nested submodules fail to load and 
        silently degrade into dictionaries (usually due to moved classes).
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source, "r", encoding="utf8") as f:
            data = f.read()
    else:
        data = source.read()

    decoded = jsonpickle.decode(data)
    
    # Recursively check for nested degraded objects across the entire PyTree
    def _verify_no_degraded_modules(obj, current_path="root"):
        if isinstance(obj, dict):
            # If a dict has 'py/object', jsonpickle failed to resolve the class path
            if 'py/object' in obj:
                failed_class = obj['py/object']
                raise TypeError(
                    f"Degraded object found at path '{current_path}'. "
                    f"Failed to instantiate the class '{failed_class}'"
                    "Did you move or rename this class in your codebase?"
                )
            for k, v in obj.items():
                _verify_no_degraded_modules(v, f"{current_path}[{repr(k)}]")
                
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _verify_no_degraded_modules(v, f"{current_path}[{i}]")
                
        elif isinstance(obj, eqx.Module):
            # Safely traverse Equinox/Parax modules and dataclasses
            for f in obj.__dataclass_fields__:
                _verify_no_degraded_modules(getattr(obj, f), f"{current_path}.{f}")

    _verify_no_degraded_modules(decoded)
    
    return decoded


def save(target: str | os.PathLike | BinaryIO, tree: Any):
    """
    Save a Parax PyTree (e.g., a Module, or a dict/list of Modules) to a file.
    
    Parameters
    ----------
    target : str, os.PathLike, or BinaryIO
        The path to the saved file or an open file-like object.
    tree : Any
        The PyTree containing Parax modules to save.
    """
    # 1. Map over the PyTree, converting only Parax Modules into their saveable forms
    def to_saveable(node):
        return node
        
    # NB: we treat all equinox Modules as leafs so JAX doesnt mangle their internals.
    # Note however that if there is a parax module inside an equinox module this wont work
    tree_save = jtu.tree_map(
        to_saveable, 
        tree, 
        is_leaf=lambda x: isinstance(x, (eqx.Module))
    )
    
    # 2. Encode the standardized tree
    data = jsonpickle.encode(tree_save, unpicklable=True)
    
    # 3. Write to file
    if isinstance(target, (str, os.PathLike)):
        with open(target, "w", encoding="utf8") as f:
            f.write(data)
    else:
        target.write(data)

__all__ = ['save', 'load']