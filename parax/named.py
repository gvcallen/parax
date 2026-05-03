import dataclasses
from typing import Sequence, Callable, Iterator
from jaxtyping import PyTree

import jax
import jax.numpy as jnp

from parax.utils.tree import path_to_pseudoname
from parax.variables import AbstractVariable
from parax.filters import is_variable


def iterate(
    pytree: PyTree,
    param_filter: str | Sequence[str] | Callable[[str], bool] = None,
    *,
    include_fixed: bool = False,
) -> Iterator[tuple[str, AbstractVariable]]:
    path_and_nodes, _ = jax.tree.flatten_with_path(pytree, is_leaf=is_variable)
    allowed_param_ids = None

    filter_is_seq_str = False
    filter_is_seq_param = False
    filter_is_callable = False
    filter_ids = None

    if param_filter is not None:
        if isinstance(param_filter, str):
            param_filter = {param_filter} 
            filter_is_seq_str = True
        elif isinstance(param_filter, AbstractVariable):
            filter_ids = {id(param_filter)}
            filter_is_seq_param = True
        elif isinstance(param_filter, Sequence):
            if len(param_filter) > 0:
                if isinstance(param_filter[0], str):
                    param_filter = set(param_filter) 
                    filter_is_seq_str = True
                elif isinstance(param_filter[0], Param):
                    filter_ids = {id(p) for p in param_filter}
                    filter_is_seq_param = True
        elif isinstance(param_filter, Callable):
            filter_is_callable = True
        else:
            raise Exception(f"Unknown filter type passed for parameters: {param_filter}")

    # 4. The Single Lazy Pass
    for path, param in path_and_nodes:
        if not is_free_array(param):
            continue
        if not include_fixed and getattr(param, "fixed", False):
            continue

        if allowed_param_ids is not None and id(param) not in allowed_param_ids:
            continue

        if filter_is_seq_param and id(param) not in filter_ids:
            continue

        name = path_to_pseudoname(path)

        if filter_is_seq_str and name not in param_filter:
            continue
        if filter_is_callable and not param_filter(name):
            continue

        # 5. Flattening & Yielding
        yield name, param


def update(
    pytree: PyTree,
    params: dict[str, Param] | dict[str, float] | jnp.ndarray | None = None,
    check_missing: bool = False,
    check_unknown: bool = True,
    fix_others = False,
    remove_metadata: bool = False,
    **param_kwargs: dict[str, Param] | dict[str, float],
) -> PyTree:
    # 2. Dictionary / Kwargs Update Path
    params = params if params is not None else {}
    params = dict(params)
    params.update(param_kwargs)
    
    new_params = dict(iterate(pytree, include_fixed=True))
    unknown_params = set(params.keys() - new_params.keys())
    if check_unknown and len(unknown_params) != 0:
        raise Exception(f"Error: the following parameters were passed but are not in the module: {unknown_params}")
    params = {k: v for k, v in params.items() if k not in unknown_params}
    
    if check_missing or fix_others:
        missing_params = set(new_params.keys() - params.keys())
        if check_missing and len(missing_params) != 0:
            raise Exception(f"Error: the following module parameters were missing: {missing_params}")
        if fix_others:
            for missing_param_name in missing_params:
                new_params[missing_param_name] = dataclasses.replace(new_params[missing_param_name], fixed=True)                    

    for name, value in params.items():
        if isinstance(value, Param):
            if remove_metadata:
                new_params[name] = dataclasses.replace(value, metadata=None)
            else:
                new_params[name] = value

        else:
            if remove_metadata:
                new_params[name] = Param(jnp.asarray(value))
            else:
                new_params[name] = new_params[name].with_value(jnp.asarray(value))                
            
    # Fast tree mapping bypasses string iteration logic completely
    def map_fn(path, node):
        if is_free_array(node):
            name = path_to_pseudoname(path)
            if name in new_params:
                new_param = new_params[name]
                if new_param.name is None:
                    new_param = dataclasses.replace(new_param, name=node.name)
                return new_param
        return node
        
    return jax.tree.map_with_path(map_fn, pytree, is_leaf=is_free_array)