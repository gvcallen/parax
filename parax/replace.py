from typing import Any, Callable
import dataclasses
from dataclasses import is_dataclass

import jax
from jaxtyping import PyTree

def tree_replace(
    pytree: PyTree, 
    *,
    is_leaf: Callable[[Any], bool] | None = is_dataclass, 
    **kwargs: PyTree
) -> PyTree:
    """
    Creates a new PyTree with its leaves replaced by leaves in parallel PyTrees.

    This extends `dataclasses.replace` to work over JAX PyTrees. 
    
    Note: The nodes defined as "leaves" by the `is_leaf` callable MUST be 
    valid dataclasses (e.g., Equinox Modules). If `is_leaf` is None, JAX will 
    recurse down to raw arrays/numbers, causing `dataclasses.replace` to fail.

    Args:
        pytree: The base PyTree to update.
        is_leaf: A function that returns True for the nodes that should be treated 
                 as dataclass leaves (e.g., `parax.is_param`). Defaults to None,
                 in which case the root PyTree is replaced (standard `replace` behaviour.)
        **kwargs: Parallel PyTrees containing the new values for specific fields.
    """
    if not kwargs:
        return pytree
        
    field_names = list(kwargs.keys())
    field_trees = list(kwargs.values())

    def _replace_leaf(x: Any, *new_values: Any) -> Any:
        if not dataclasses.is_dataclass(x):
            raise TypeError(f"Expected a dataclass leaf, but got {type(x)}. Check your `is_leaf` function.")
            
        updates = {name: val for name, val in zip(field_names, new_values)}
        return dataclasses.replace(x, **updates)

    return jax.tree.map(
        _replace_leaf,
        pytree,
        *field_trees,
        is_leaf=is_leaf
    )