"""
IO helpers e.g. for module loading and saving.
"""
import os
import jsonpickle
from typing import BinaryIO

from parax.module import Module

def load(source: str | BinaryIO) -> Module:
    """
    Load a Parax module from a file or file-like object.

    Parameters
    ----------
    source : str or BinaryIO
        The path to the saved module file or an open file-like object 
        containing the module data.

    Returns
    -------
    Module
        The deserialized Parax module instance.
        
    Raises
    ------
    TypeError
        If the root module or any nested submodules fail to load and 
        silently degrade into dictionaries (usually due to moved classes).
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source, "r", encoding="utf8") as f:
            data = f.read()
    else:
        data = source.read()

    decoded = jsonpickle.decode(data)    
    
    # 1. Check if the root object degraded
    if not isinstance(decoded, Module):
        raise TypeError(
            f"Failed to load module. Expected a Parax Module, but got '{type(decoded).__name__}'. "
            "This almost always happens because the module's class was moved to a different module, "
            "renamed, or deleted. jsonpickle cannot find the import path and silently degraded it to a dict."
        )

    # 2. Recursively check for nested degraded modules
    def _verify_no_degraded_modules(obj, current_path="root"):
        if isinstance(obj, dict):
            # Parax modules have tell-tale internal fields. If a dict has these, it's a dead module.
            if '_param_groups' in obj or '_separator' in obj or 'z0' in obj:
                raise TypeError(
                    f"Degraded submodule found at path '{current_path}'. "
                    "A nested module failed to instantiate and became a dictionary. "
                    "Did you move or rename a component class in your codebase?"
                )
            for k, v in obj.items():
                _verify_no_degraded_modules(v, f"{current_path}.{k}")
                
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _verify_no_degraded_modules(v, f"{current_path}[{i}]")
                
        elif hasattr(obj, '__dataclass_fields__'):
            # Safely traverse Equinox modules and dataclasses
            for f in obj.__dataclass_fields__:
                _verify_no_degraded_modules(getattr(obj, f), f"{current_path}.{f}")

    # _verify_no_degraded_modules(decoded)

    return decoded

def save(target: str | BinaryIO, module: Module):
    """
    Save a Parax module to a file or file-like object.
    ...
    """
    module_save = module._saveable()
    data = jsonpickle.encode(module_save)
    
    if isinstance(target, (str, os.PathLike)):
        with open(target, "w", encoding="utf8") as f:
            f.write(data)
    else:
        target.write(data)

__all__ = ['save', 'load']