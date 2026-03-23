"""
IO helpers e.g. for module loading and saving.
"""
import os
import json
from typing import BinaryIO

import jsonpickle
import jsonpickle.handlers

from parax.core.module import Module
from parax.core.parameter import Parameter


class ParameterHandler(jsonpickle.handlers.BaseHandler):
    """
    Custom jsonpickle handler to explicitly use parax.Parameter's 
    to_json() and from_json() serialization logic.
    """
    def flatten(self, obj: Parameter, data: dict) -> dict:
        # 1. Get the JSON string from your custom method and parse it into a dict
        param_dict = json.loads(obj.to_json())
        
        # 2. Merge it into the jsonpickle data payload. 
        # `data` already contains {"py/object": "parax.core.parameter.Parameter"}
        data.update(param_dict)
        return data

    def restore(self, data: dict) -> Parameter:
        # 1. Strip out jsonpickle's internal tracking metadata
        clean_data = {k: v for k, v in data.items() if not k.startswith("py/")}
        
        # 2. Re-encode to a JSON string and pass to your classmethod
        json_str = json.dumps(clean_data)
        return Parameter.from_json(json_str)

# Register the custom handler with jsonpickle globally
ParameterHandler.handles(Parameter)


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
            # If a dict has 'py/object', jsonpickle failed to resolve the class path
            if 'py/object' in obj:
                failed_class = obj['py/object']
                raise TypeError(
                    f"Degraded object found at path '{current_path}'. "
                    f"jsonpickle failed to instantiate the class '{failed_class}' and reverted to a dictionary. "
                    "Did you move or rename this class in your codebase?"
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

    _verify_no_degraded_modules(decoded)

    return decoded

def save(target: str | BinaryIO, module: Module):
    """
    Save a Parax module to a file or file-like object.
    
    Parameters
    ----------
    target : str or BinaryIO
        The path to the saved module file or an open file-like object.
    module : Module
        The Parax module to save.
    """
    module_save = module.saveable()
    data = jsonpickle.encode(module_save, unpicklable=True)
    
    if isinstance(target, (str, os.PathLike)):
        with open(target, "w", encoding="utf8") as f:
            f.write(data)
    else:
        target.write(data)

__all__ = ['save', 'load']