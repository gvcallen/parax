"""
The main module class.

This module defines [`parax.Module`][], a frozen, JAX-compatible, Equinox module.

"""

from functools import partial
from copy import copy, deepcopy
from typing import Callable, Sequence, Iterator, Self, ClassVar, Any
import dataclasses
from dataclasses import fields, is_dataclass
from typing_extensions import dataclass_transform
import operator

import jax
import jax.numpy as jnp
from jax import flatten_util
from jax.tree_util import GetAttrKey, DictKey, SequenceKey, FlattenedIndexKey
import equinox as eqx
from distreqx.distributions import AbstractDistribution, Uniform as UniformDistribution, Joint
from distreqx.bijectors import AbstractBijector

from parax.parameter import Parameter, is_valid_param, as_param
from parax.parameter_group import ParameterGroup
from parax.field import field
from parax.tree import partition
from parax.utils import get_first_underlying_type, nodes_by_type

@dataclass_transform(field_specifiers=(field, eqx.field, dataclasses.field))
class ModuleMeta(type(eqx.Module)):
    def __new__(mcs, name, bases, namespace, **kwargs):
        annotations = namespace.get('__annotations__', {})
        
        for field_name, field_types in annotations.items():
            field_kwargs = {}
            default = namespace.get(field_name, dataclasses.MISSING)
            
            field_type = get_first_underlying_type(field_types)
            is_param_type = field_type is not None and isinstance(field_type, type) and issubclass(field_type, Parameter)
            
            # 1. Handle explicit Field declarations (e.g., x = prx.field(...))
            if isinstance(default, dataclasses.Field):
                if is_param_type:
                    has_converter = default.metadata is not None and "converter" in default.metadata
                    
                    # If they didn't provide their own converter, inject ours
                    if not has_converter:
                        valid_keys = getattr(default, '__slots__', None) or vars(default).keys()
                        
                        field_kwargs = {
                            k: getattr(default, k) for k in valid_keys
                            if not k.startswith('_') and k not in ('name', 'type')
                        }
                        
                        field_kwargs['converter'] = lambda x, fn=field_name: as_param(x, fixed=False)
                        
                        if 'metadata' in field_kwargs and field_kwargs['metadata']:
                            field_kwargs['metadata'] = dict(field_kwargs['metadata'])
                            field_kwargs['metadata'].pop('converter', None)
                        
                        field_kwargs = {k: v for k, v in field_kwargs.items() if v is not dataclasses.MISSING}
                        
                        # USE CUSTOM FIELD: Repackage the explicitly defined field using prx.field
                        namespace[field_name] = field(**field_kwargs)
                continue

            # 2. Handle standard type hints (e.g., x: Parameter = 10)
            if is_param_type:
                if default is not dataclasses.MISSING and not isinstance(default, Parameter):                
                    if isinstance(default, tuple):
                        raise Exception(f"Expected a parameter for default '{field_name}' in class {name} but found a tuple.")
                    field_kwargs['default'] = default
                
                field_kwargs['converter'] = lambda x, fn=field_name: as_param(x, fixed=False)
            
            # 3. Apply default_factory to avoid Python's mutable default trap
            if default is not dataclasses.MISSING:
                if any(isinstance(default, m_type) for m_type in {list, dict, tuple, eqx.Module, jnp.ndarray}):
                    field_kwargs['default_factory'] = lambda default=default: deepcopy(default)
                    field_kwargs.pop('default', None)
            
            # 4. Inject the final configured field back into the namespace
            if len(field_kwargs) != 0:
                namespace[field_name] = field(**field_kwargs)
                
        return super().__new__(mcs, name, bases, namespace, **kwargs)

# 2. Assign the metaclass to your base Module
class Module(eqx.Module, metaclass=ModuleMeta):
    """
    An extension of an Equinox ``Module``.

    This class extends an Equinox ``Module`` with additional helpful features and methods.
    
    One feature includes the ability to inspect and modify parameters using strings based
    on their module path. This is helpful for modifying deep, hierachical modules
    using unique identifiers.
    
    Another feature is the fact that attributes marked with the `Parameter` type
    are automatically given parameter-converters. This ensures that they remain parameters after construction
    (e.g. when initializing them with a float).
    
    Usage
    -----
    - Define your model by sub-classing the module and adding custom parameters and/or sub-modules.
    - Construct modules by passing parameters and/or submodules to the initializer (like a dataclass).
    - Retrieve parameter information via methods such as [`parax.Module.named_params`][], [`parax.Module.param_names`][], [`parax.Module.flat_params`][], etc..
    - Use `with_xxx` functions to modify fields, modules and parameters within the module e.g. [`parax.Module.with_params`][], [`parax.Module.with_fields`][].

    Methods & Properties Summary
    ----------------------------

    **Introspection Properties**

    **Module Inspection & Manipulation**

    | Method | Description |
    |---|---|
    | [`children`][parax.Module.children] | Returns the immediate submodules. |
    | [`submodules`][parax.Module.submodules] | Returns all nested submodules (depth-first). |
    | [`sampled`][parax.Module.sampled] | Return a new module with parameters drawn from this module's distribution. |

    **Parameter Inspection**

    | Method | Description |
    |---|---|
    | [`named_params`][parax.Module.named_params] | Named module parameter objects as a dict. |
    | [`named_param_values`][parax.Module.named_param_values] | Named module parameter values as a dict of jax arrays. |
    | [`param_names`][parax.Module.param_names] | Module parameter names as a list. |
    | [`param`][parax.Module.param] | A single module parameter object by name. |
    | [`params`][parax.Module.params] | Module parameters as a list. |
    | [`param_value`][parax.Module.param_value] | A single module parameter value by name. |
    | [`param_values`][parax.Module.param_values] | Module parameter values as a list of jax arrays. |
    | [`named_flat_params`][parax.Module.named_flat_params] | Named flattened module parameter objects as a dict. |
    | [`named_flat_param_values`][parax.Module.named_flat_param_values] | Named flattened module parameter values as a dict. |
    | [`flat_param_names`][parax.Module.flat_param_names] | Flattened parameter names as a list. |
    | [`flat_params`][parax.Module.flat_params] | Flattened parameters as a list. |
    | [`flat_param_values`][parax.Module.flat_param_values] | Flattened module parameter values as a flat array. |
    | [`param_groups`][parax.Module.param_groups] | Return all parameter groups relevant to this module. |

    **Parameter Manipulation**

    | Method | Description |
    |---|---|
    | [`with_params`][parax.Module.with_params] | Return a module with parameters updated. |
    | [`with_mapped_params`][parax.Module.with_mapped_params] | Apply a map function to parameters. |
    | [`with_transformed_params`][parax.Module.with_transformed_params] | Apply a map function to parameters. |
    | [`with_fixed_params`][parax.Module.with_fixed_params] | Return a module with specified parameters fixed. |
    | [`with_free_params`][parax.Module.with_free_params] | Return a module with specified parameters free. |
    | [`with_free_params_only`][parax.Module.with_free_params_only] | Return a module with ONLY the specified parameters free. |
    | [`with_all_params_fixed`][parax.Module.with_all_params_fixed] | Return a module with all parameters fixed. |
    | [`with_all_params_free`][parax.Module.with_all_params_free] | Return a module with all parameters free. |

    **Parameter Group Manipulation**

    | Method | Description |
    |---|---|
    | [`with_param_groups`][parax.Module.with_param_groups] | Return a module with parameter groups appended. |
    | [`with_demoted_param_groups`][parax.Module.with_demoted_param_groups] | Recursively demote parameter groups to deepest submodule. |
    | [`with_no_param_groups`][parax.Module.with_no_param_groups] | Return a module with all parameter groups removed. |

    **Distribution Manipulation**

    | Method | Description |
    |---|---|
    | [`with_mapped_distributions`][parax.Module.with_mapped_distributions] | Apply a map function to the parameter distributions. |
    | [`with_uniform_distributions`][parax.Module.with_uniform_distributions] | Return a module with uniform distributions set. |

    **Field & Module Manipulation**

    | Method | Description |
    |---|---|
    | [`with_defaults`][parax.Module.with_defaults] | Return this module type with default initialization args. |
    | [`with_modules`][parax.Module.with_modules] | Combines this module with free parameters in other modules. |
    | [`with_fields`][parax.Module.with_fields] | Return a copy with dataclass-style field replacements. |
    | [`with_name`][parax.Module.with_name] | Return a copy of this module with a different name. |
    | [`with_submodule_fields`][parax.Module.with_submodule_fields] | Dataclass-style field replacements on a nested sub-module. |
    | [`with_free_submodules`][parax.Module.with_free_submodules] | Free all parameters in the given submodules. |
    | [`with_free_submodules_only`][parax.Module.with_free_submodules_only] | Returns a module with ONLY the specified submodules freed. |
    | [`with_fixed_submodules`][parax.Module.with_fixed_submodules] | Fix all parameters in the given submodules. |
    
    **Function Tools**

    | Method | Description |
    |---|---|
    | [`func_jacobian`][parax.Module.func_jacobian] | Calculate the Jacobian of a function w.r.t parameters. |
    | [`func_sensitivity`][parax.Module.func_sensitivity] | Calculate the sensitivity of a function w.r.t parameters. |
    | [`func_samples`][parax.Module.func_samples] | Evaluate a function over parameter samples. |    

    Attributes
    ----------
    name : str or None
        An optional name for the module instance.
    """
    # Public init fields
    name: str | None = field(default=None, kw_only=True, static=True)
    _param_groups: list[ParameterGroup] = field(default_factory=list, kw_only=True, repr=False, static=True, init=False)
    
    # Class variables
    _separator: ClassVar[str] = '_'
    _transparent: ClassVar[bool] = False

    # ---- Internal initialization methods -------------------------------------------------
    
    def __init_subclass__(cls, transparent: bool = False, **kwargs):
        """Customize subclass construction."""        
        super().__init_subclass__(**kwargs)
        cls._transparent = transparent
            
    def path_to_param_name(self, path) -> str:
        """Convert a PyTree path to a fully-qualified parameter name."""
        name_fields = []
        node = self

        for item in path:
            if isinstance(item, GetAttrKey):
                k = item.name
                next_node = getattr(node, k)
                
                # 1. Determine transparency
                is_transparent = getattr(node, '_transparent', False)
                if not is_transparent and is_dataclass(node):
                    field_obj = next((f for f in fields(node) if f.name == k), None)
                    if field_obj is not None:
                        is_transparent = field_obj.metadata.get('transparent', False)

                # 2. Extract user override
                explicit_name = getattr(next_node, 'name', None)
                
                # 3. Rule application
                if is_transparent:
                    if explicit_name is not None:
                        name_fields.append(explicit_name)
                else:
                    name_fields.append(explicit_name if explicit_name is not None else k)
                        
                node = next_node
                
            elif isinstance(item, DictKey):
                k = item.key
                node = node[k]
                name_fields.append(str(k))
                    
            elif isinstance(item, (SequenceKey, FlattenedIndexKey)):
                idx = item.idx if hasattr(item, 'idx') else item.key
                node = node[idx]
                explicit_name = getattr(node, 'name', None)
                if explicit_name is not None:
                    name_fields.append(explicit_name)
                else:
                    name_fields.append(str(idx))
                    
            else:
                raise Exception(f"Unsupported key type in path: {type(item)}")
                
        return self._separator.join(name_fields)
    
    def saveable(self: Self) -> Self:
        def strip_unsaveable_recursive(obj, memo=None):
            if memo is None:
                memo = {}
            
            obj_id = id(obj)
            if obj_id in memo:
                return memo[obj_id]

            # Case 1: It's an Equinox Module
            if isinstance(obj, Module):
                updates = {}
                
                for f in dataclasses.fields(obj):
                    if f.metadata.get('save', True) is False:
                        if f.default is not dataclasses.MISSING:
                            new_val = f.default
                        elif f.default_factory is not dataclasses.MISSING:
                            new_val = f.default_factory()
                        else:
                            new_val = None
                    else:
                        val = getattr(obj, f.name)
                        new_val = strip_unsaveable_recursive(val, memo)
                        
                    updates[f.name] = new_val
                
                for k, v in updates.items():
                    object.__setattr__(obj, k, v)

                memo[obj_id] = obj
                return obj

            # Case 2: Standard containers
            elif isinstance(obj, (list, tuple)):
                obj = type(obj)(strip_unsaveable_recursive(x, memo) for x in obj)
                memo[obj_id] = obj
                return obj

            elif isinstance(obj, dict):
                obj = {k: strip_unsaveable_recursive(v, memo) for k, v in obj.items()}
                memo[obj_id] = obj
                return obj

            # Case 3: Leaf nodes
            else:
                return obj

        clean_module = strip_unsaveable_recursive(copy(self))
        return clean_module
    
    def iter_params(
        self,
        param_filter: str | Sequence[str] | Parameter | Sequence[Parameter] | Callable[[str], bool] = None,
        *,
        include_fixed: bool = False,
        flatten: bool = False,
        submodules: 'Module | Sequence[Module] | str | Sequence[str] | None' = None,
    ) -> Iterator[tuple[str, Parameter]]:
        """Iterate over (name, Parameter) pairs in internal order."""
        
        # 1. Direct Flattening
        path_and_nodes, _ = jax.tree.flatten_with_path(self, is_leaf=is_valid_param)

        # 2. Pre-process submodule IDs outside the loop (Fast JAX C++ traversal)
        allowed_param_ids = None
        if submodules is not None:
            if isinstance(submodules, (Module, str)):
                submodules = [submodules]
                
            resolved_submodules = [getattr(self, sm) if isinstance(sm, str) else sm for sm in submodules]
            if not isinstance(resolved_submodules[0], Module):
                raise Exception(f"Got unknown type when expecting a module or string. Type was: {type(resolved_submodules[0])}")

            allowed_param_ids = set()
            for sm in resolved_submodules:
                sm_nodes, _ = jax.tree.flatten(sm, is_leaf=is_valid_param)
                for node in sm_nodes:
                    if is_valid_param(node) and (include_fixed or not getattr(node, "fixed", False)):
                        allowed_param_ids.add(id(node))

        # 3. Pre-process filters into O(1) lookups
        filter_is_seq_str = False
        filter_is_seq_param = False
        filter_is_callable = False
        filter_ids = None

        if param_filter is not None:
            if isinstance(param_filter, str):
                param_filter = {param_filter} 
                filter_is_seq_str = True
            elif isinstance(param_filter, Parameter):
                filter_ids = {id(param_filter)}
                filter_is_seq_param = True
            elif isinstance(param_filter, Sequence):
                if len(param_filter) > 0:
                    if isinstance(param_filter[0], str):
                        param_filter = set(param_filter) 
                        filter_is_seq_str = True
                    elif isinstance(param_filter[0], Parameter):
                        filter_ids = {id(p) for p in param_filter}
                        filter_is_seq_param = True
            elif isinstance(param_filter, Callable):
                filter_is_callable = True
            else:
                raise Exception(f"Unknown filter type passed for parameters: {param_filter}")

        # 4. The Single Lazy Pass
        for path, param in path_and_nodes:
            if not is_valid_param(param):
                continue
            if not include_fixed and getattr(param, "fixed", False):
                continue

            if allowed_param_ids is not None and id(param) not in allowed_param_ids:
                continue

            if filter_is_seq_param and id(param) not in filter_ids:
                continue

            name = self.path_to_param_name(path)

            if filter_is_seq_str and name not in param_filter:
                continue
            if filter_is_callable and not param_filter(name):
                continue

            # 5. Flattening & Yielding
            if flatten and (param.size > 1 or isinstance(param.name, list)):
                flattened_params = param.flattened(separator=self._separator)
                for i, subparam in enumerate(flattened_params):
                    suffix = subparam.name if subparam.name is not None else str(i)
                    yield f"{name}{self._separator}{suffix}", subparam
            else:
                yield name, param
    
    @property
    def num_params(self) -> int:
        """Number of free parameters.

        Returns
        -------
        int
        """
        return len(self.params())

    @property
    def num_flat_params(self) -> int:
        """Number of free, **flattened** parameters.

        Returns
        -------
        int
        """
        return len(self.flat_params())          
    
    # ---- Magic methods and copying --------------------------------------------------

    def copy(self: Self) -> Self:
        """Returns a deepcopy of self.

        Returns
        -------
        Module
        """        
        return deepcopy(self)   
    
    def __getitem__(self, key: str | Sequence[str]):
        if isinstance(key, str):
            return self.param_value(key)
        else:
            named_param_values = self.named_param_values()
            for k in key:
                if k not in named_param_values.keys():
                    raise Exception(f"Parameter name '{k}' was passed but is not a free parameter")
            return [v for k, v in named_param_values.items() if k in key]
        
    def __repr__(self):
        module_param_fields = []
        other_fields = []
        base_fields = []
        
        for f in dataclasses.fields(self):
            if f.repr is False:
                continue
            
            val = getattr(self, f.name)
            val_repr = repr(val)
            
            indented_val_repr = val_repr.replace('\n', '\n  ')
            formatted_field = f"  {f.name}={indented_val_repr}"
            
            if f.name == "name":
                base_fields.append(formatted_field)
            elif isinstance(val, (Module, Parameter)):
                module_param_fields.append(formatted_field)
            else:
                other_fields.append(formatted_field)
            
        all_fields_str = module_param_fields + other_fields + base_fields
        joined_fields = ",\n".join(all_fields_str)
        
        return f"{self.__class__.__name__}(\n{joined_fields}\n)"

    # ---- Module inspection --------------------------------------------------    
    
    def children(self) -> list['Module']:
        """Returns the immediate submodules.

        Returns
        -------
        list[Module]
        """
        return [node for node in eqx.tree_flatten_one_level(self)[0] if isinstance(node, Module)]
    
    def submodules(self) -> list['Module']:
        """Returns all nested submodules (depth-first), excluding ``self``.

        Returns
        -------
        list[Module]
        """
        return nodes_by_type(self, Module)[1:]         

    def sampled(self, key=None, **kwargs) -> 'Module':
        """Returns a new module with parameters sampled from this parameter's distribution.
        
        Returns
        -------
        Module
        """
        dist = self.flat_distribution()
        flat_param_samples = dist.sample(key, sample_shape=(1,))[0]
        return self.with_params(flat_param_samples)
    
    def merged(self: Self, modules: Self | Sequence[Self]) -> Self:
        """Merge this module with free parameters and parameter groups
        in other modules.
        
        This is useful to combine separate modules obtained from fitting
        the same initial module with different free parameters.

        Parameters
        ----------
        modules : Module or Sequence[Module]
            The other modules to combine this module with.

        Returns
        -------
        Module
        """  
        if not isinstance(modules, Sequence):
            modules = [modules]

        combined = self
        for other in modules:
            combined = combined.with_params(other.named_params())
            combined = combined.with_param_groups(other.param_groups(explicit_only=True))
        return combined    
        
    # ---- Parameter inspection -------------------------------------------------- 
    
    def named_params(self, param_filter: str | Sequence[str] | Parameter | Sequence[Parameter] | Callable[[str], bool] = None, *, include_fixed=False, submodules: 'Module' | Sequence['Module'] | str | Sequence[str] | None = None) -> dict[str, Parameter]:
        """Named module parameters as a dict.

        Keys are fully-qualified parameter names.
        The order matches the internal flattened array order.

        Parameters
        ----------
        param_filter : str | Sequence[str] | Parameter | Sequence[Parameter] | Callable[[str], bool], default=None
            A filter indicating which parameters to return. For the default case, all parameters are returned.
        include_fixed : bool, default=False
            Include fixed parameters.
        submodules : Module | Sequence[Module] | str | Sequence[str] | None, optional
            Restrict to parameters used by the given submodule(s). If strings are
            provided, ``getattr(self, name)`` is used.

        Returns
        -------
        dict[str, Parameter]
        """
        return dict(self.iter_params(param_filter=param_filter, include_fixed=include_fixed, submodules=submodules))
    
    def named_param_values(self, scaled=False, **kwargs) -> dict[str, jnp.ndarray]:
        """Named module parameter values as a dict of jax arrays.

        See [`parax.Module.named_params`][].

        Parameters
        ----------
        scaled : bool, default=False
            Whether or not to scale the returned values by the parameter scales.
        **kwargs
            Additional key-word arguments as in  [`parax.Module.named_params`][].

        Returns
        -------
        dict[str, jnp.ndarray]
        """     
        if scaled:
            return {n: jnp.array(p) for n, p in (self.iter_params(**kwargs))}
        else:
            return {n: p.latent_value for n, p in (self.iter_params(**kwargs))}    

    def param_names(self, *args, **kwargs) -> list[str]:
        """
        Return module parameter names as a list.

        See [`parax.Module.named_params`][].
        """
        return list(self.named_params(*args, **kwargs).keys())

    def param(self, name: str, *args, **kwargs) -> Parameter:
        """
        Return a single module parameter by name.

        See [`parax.Module.named_params`][].
        """
        return self.named_params(*args, **kwargs)[name]
    
    def params(self, *args, **kwargs) -> list[Parameter]:
        """
        Return module parameters as a list.

        See [`parax.Module.named_params`][].
        """
        return list(self.named_params(*args, **kwargs).values())
    
    def param_value(self, name: str, *args, **kwargs) -> jnp.ndarray:
        """
        Return a single module parameter value by name as a single jax array.

        See [`parax.Module.named_param_values`][].
        """
        return self.named_param_values(*args, **kwargs)[name]

    def param_values(self, *args, **kwargs) -> list[jnp.ndarray]:
        """
        Return module parameter values as a list of jax arrays.

        See [`parax.Module.named_param_values`][].
        """
        return list(self.named_param_values(*args, **kwargs).values())
    
    def named_flat_params(self, include_fixed=False, submodules: 'Module' | Sequence['Module'] | str | Sequence[str] | None = None) -> dict[str, Parameter]:
        """Named flattened module parameters as a dict.

        Flat parameters are a de-vectorized version of
        the internal parameters of the module. The returned
        parameter objects therefore are not necessarily
        equal to the internal module objects.
        
        Keys are fully-qualified parameter names with de-vectorized suffixes added.
        The order matches the internal flattened array order.

        Parameters
        ----------
        include_fixed : bool, default=False
            Include fixed parameters.
        submodules : Module | Sequence[Module] | str | Sequence[str] | None, optional
            Restrict to parameters used by the given submodule(s). If strings are
            provided, ``getattr(self, name)`` is used.

        Returns
        -------
        dict[str, Parameter]
        """
        return dict(self.iter_params(flatten=True, include_fixed=include_fixed, submodules=submodules))
    
    def named_flat_param_values(self, scaled=False, return_floats=False, **kwargs) -> dict[str, jnp.ndarray]:
        """Named flattened module parameter values as a dict of jax arrays.

        See [`parax.Module.named_flat_params`][].

        Parameters
        ----------
        scaled : bool, default=False
            Whether or not to scale the returned values by the parameter scales.
        **kwargs
            Additional key-word arguments as in  [`parax.Module.named_params`][].

        Returns
        -------
        dict[str, jnp.ndarray]
        """     
        if scaled:
            retval = {n: jnp.array(p) for n, p in (self.iter_params(flatten=True, **kwargs))}
        else:
            retval = {n: p.latent_value for n, p in (self.iter_params(flatten=True, **kwargs))}
            
        if return_floats:
            import numpy as np
            retval = {k: float(np.array(v)) for k, v in retval.items()}
        return retval
         
    def flat_param_names(self, *args, **kwargs) -> list[str]:
        """
        Return flattened parameter names as a list.

        See [`parax.Module.named_flat_params`][].
        """
        return list(self.named_flat_params(*args, **kwargs).keys())    
    
    def flat_params(self, *args, **kwargs) -> list[Parameter]:
        """
        Return flattened parameters as a list.

        See [`parax.Module.named_flat_params`][].
        """
        return list(self.named_flat_params(*args, **kwargs).values())
    
    def flat_param_values(self, *args, **kwargs) -> jnp.ndarray:
        """
        Return flattened module parameter values as a jax arrays.

        See [`parax.Module.named_flat_param_values`][].
        """
        return jnp.array(list(self.named_flat_param_values(*args, **kwargs).values())).reshape(-1)
    
    # ---- Grouped Parameter Inspection -----------------------------------------

    def named_grouped_params(self, include_fixed=False) -> dict[str, dict[str, Parameter]]:
        """
        Returns a dictionary of parameter groups, where each group is a 
        dictionary mapping parameter names to their Parameter objects.
        
        The outer dictionary keys are the ParameterGroup's `name` attribute. 
        If the group has no name, its index in `param_groups()` is used.
        """
        groups = self.param_groups(include_fixed=include_fixed)
        flat_params = self.named_flat_params(include_fixed=include_fixed)
        
        result = {}
        for i, group in enumerate(groups):
            group_key = getattr(group, 'name', None)
            if group_key is None:
                group_key = str(i)
                
            result[group_key] = {name: flat_params[name] for name in group.param_names}
            
        return result

    def named_grouped_param_values(self, scaled=False, include_fixed=False, **kwargs) -> dict[str, dict[str, jax.Array]]:
        """
        Returns a dictionary of parameter groups, where each group is a 
        dictionary mapping parameter names to their physical JAX array values.
        """
        groups = self.param_groups(include_fixed=include_fixed)
        flat_vals = self.named_flat_param_values(scaled=scaled, include_fixed=include_fixed, **kwargs)
        
        result = {}
        for i, group in enumerate(groups):
            group_key = getattr(group, 'name', None)
            if group_key is None:
                group_key = str(i)
                
            result[group_key] = {name: flat_vals[name] for name in group.param_names}
            
        return result

    def grouped_params(self, include_fixed=False) -> dict[str, Parameter | list[Parameter]]:
        """
        Returns a dictionary of parameter groups mapping to their Parameter objects.
        
        If a group contains multiple parameters, they are returned as a list.
        If a group contains a single parameter, the Parameter object is returned directly.
        """
        named_grouped = self.named_grouped_params(include_fixed=include_fixed)
        
        result = {}
        for group_key, param_dict in named_grouped.items():
            param_list = list(param_dict.values())
            # Squeeze single-element lists to match statistical event shapes
            result[group_key] = param_list[0] if len(param_list) == 1 else param_list
            
        return result

    def grouped_param_values(self, scaled=False, include_fixed=False, **kwargs) -> dict[str, jax.Array]:
        """
        Returns a dictionary of parameter groups mapping to their stacked JAX arrays.
        
        This structure mathematically matches the `event_shape` of `.distribution()` 
        and is the required format for evaluating log-probabilities and sampling.
        """
        named_grouped = self.named_grouped_param_values(scaled=scaled, include_fixed=include_fixed, **kwargs)
        
        result = {}
        for group_key, val_dict in named_grouped.items():
            array_list = list(val_dict.values())
            
            # Stack into a single array for Multivariate distributions
            stacked_array = jnp.stack(array_list)
            
            # Squeeze scalar parameters back to shape () for Univariate distributions
            if len(array_list) == 1:
                stacked_array = jnp.squeeze(stacked_array, axis=0)
                
            result[group_key] = stacked_array
            
        return result    

    def param_groups(self, include_fixed=False, explicit_only=False) -> list[ParameterGroup]:
        """Return all parameter groups relevant to this module, including submodules.

        This function recursively traverses submodules to collect their parameter groups,
        adjusting parameter names to match the current module's scope.

        Priority is given to groups defined in the parent module. If a parameter is 
        grouped explicitly in `self._param_groups`, it will be removed from any 
        groups returned by submodules.

        Parameters
        ----------
        include_fixed : bool, default=False
            Include groups involving fixed parameters.

        Returns
        -------
        list[ParameterGroup]
        """
        if explicit_only:
            return deepcopy(self._param_groups)
        
        all_valid_params = self.named_flat_params(include_fixed=include_fixed)
        valid_param_names = set(all_valid_params.keys())

        groups = []
        for group in self._param_groups:
            if not set(group.param_names).isdisjoint(valid_param_names):
                groups.append(deepcopy(group))

        path_and_nodes, _ = jax.tree_util.tree_flatten_with_path(
            self, 
            is_leaf=lambda x: isinstance(x, Module) and x is not self
        )

        for path, node in path_and_nodes:
            if isinstance(node, Module) and node is not self:
                relative_name = self.path_to_param_name(path)
                prefix = f"{relative_name}{self._separator}" if relative_name else ""
                sub_groups = node.param_groups(include_fixed=include_fixed)

                for sub_group in sub_groups:
                    new_names = [prefix + name for name in sub_group.param_names]
                    lifted_group = dataclasses.replace(sub_group, param_names=new_names)
                    groups.append(lifted_group)

        final_groups = []
        seen_params = set()

        for group in groups:
            valid_names = [name for name in group.param_names if name not in seen_params]
            if valid_names:
                if len(valid_names) != len(group.param_names):
                    group = dataclasses.replace(group, param_names=valid_names)
                final_groups.append(group)
                seen_params.update(valid_names)

        for name, param in all_valid_params.items():
            if name not in seen_params:
                final_groups.append(ParameterGroup(param_names=[name], distribution=param.distribution))
                seen_params.add(name)

        return final_groups
    
    def grouped_distribution(self) -> Joint:
        """
        (experimental) Returns a distreqx.Joint distribution where the PyTree structure 
        matches `self.grouped_param_values()`.
        """
        groups = self.param_groups(include_fixed=False)
        
        dist_dict = {}
        for i, group in enumerate(groups):
            group_key = getattr(group, 'name', None) or str(i)
            dist_dict[group_key] = group.distribution
            
        return Joint(dist_dict)
    
    def validate_params(self: Self) -> None:
        """
        Validates that all parameters in the module hierarchy have unique names.
        
        Raises
        ------
        ValueError
            If duplicate parameter names or flattened parameter names are detected.
        """
        seen_names = set()
        duplicates = set()
        
        # 1. Check standard parameter names
        for name, _ in self.iter_params(flatten=False):
            if name in seen_names:
                duplicates.add(name)
            seen_names.add(name)

        seen_flat = set()
        flat_duplicates = set()
        
        # 2. Check flattened parameter names (suffix collisions)
        for name, _ in self.iter_params(flatten=True):
            if name in seen_flat:
                flat_duplicates.add(name)
            seen_flat.add(name)

        # 3. Aggregate errors and report
        error_msgs = []
        if duplicates:
            error_msgs.append(f"- Duplicate standard names: {', '.join(duplicates)}")
        if flat_duplicates:
            error_msgs.append(f"- Duplicate flattened names: {', '.join(flat_duplicates)}")

        if error_msgs:
            raise ValueError(
                f"Parameter name collision detected in {self.__class__.__name__}!\n"
                + "\n".join(error_msgs)
                + "\nEnsure that custom 'name' attributes on submodules and parameters are unique within their scope."
            )
    
    # ---- Parameter Manipulation --------------------------------------------------            

    def with_params(
        self: Self,
        params: dict[str, Parameter] | dict[str, float] | jnp.ndarray | None = None,
        check_missing: bool = False,
        check_unknown: bool = True,
        fix_others = False,
        include_fixed = False,
        **param_kwargs: dict[str, Parameter] | dict[str, float],
    ) -> Self:
        """Return a new module with parameters updated.

        This is a multi-purpose function that updates parameters differently
        based on the types pass.

        Parameters
        ----------
        params : dict[str, Parameter] | dict[str, float] | jnp.ndarray | None, optional
            Parameter updates. If an array, **all** values must be provided
            (matching ``flat_params`` order). You may also pass keyword args.
        check_missing : bool, default=False
            Require that all module parameters are specified.
        check_unknown : bool, default=True
            Error if unknown parameter keys are provided.
        fix_others : bool, default=False
            Fix any parameters not explicitly passed.
        include_fixed : bool, default=False
            Include fixed parameters when interpreting ``params`` mapping.
        **param_kwargs : dict
            Additional parameter updates by name.

        Returns
        -------
        Self

        Raises
        ------
        Exception
            If shape/order mismatches, unknown/missing names (when checked),
            or if arrays are found outside of Parameters.
        """
        # 1. High-Efficiency Array Update Path
        if params is not None and not isinstance(params, dict):
            if len(param_kwargs) > 0:
                raise ValueError("Cannot pass both a flat array and explicit keyword arguments to with_params.")
                
            params_array = jnp.asarray(params)
            
            # Use `eqx.partition` to perfectly isolate parameter values while preserving exact tree structure
            dynamic, static = partition(self, include_fixed=include_fixed, param_objects=False)
            flat_dynamic, unflatten_fn = flatten_util.ravel_pytree(dynamic)
            
            if flat_dynamic.size != params_array.size:
                raise Exception(f"Array size mismatch: Expected {flat_dynamic.size} elements, but got {params_array.size}.")
                
            new_dynamic = unflatten_fn(params_array)
            return eqx.combine(new_dynamic, static)
            
        # 2. Dictionary / Kwargs Update Path
        params = params if params is not None else {}
        params = dict(params)
        params.update(param_kwargs)
        
        new_params = self.named_params(include_fixed=True)
        
        parent_keys = set(new_params.keys())
        input_keys = set(params.keys())
        potential_flat_keys = input_keys - parent_keys
        
        if potential_flat_keys:
            parents_to_scan = [p for p in parent_keys if p not in params]
            for parent_name in parents_to_scan:
                parent_param = new_params[parent_name]
                if parent_param.size > 0: 
                    sub_params = parent_param.flattened(separator=self._separator)
                    updates_found = False
                    new_sub_values = []
                    
                    for i, sub_p in enumerate(sub_params):
                        suffix = sub_p.name if sub_p.name is not None else str(i)
                        flat_name = f"{parent_name}{self._separator}{suffix}"
                        
                        if flat_name in params:
                            val = params[flat_name]
                            if hasattr(val, 'item') and getattr(val, "size", 1) == 1:
                                val = val.item()
                            try:
                                val = float(val)
                            except Exception:
                                raise Exception(f"Value for flat parameter '{flat_name}' must be convertible to float. Got: {val}")
                            new_sub_values.append(val)
                            del params[flat_name]
                            updates_found = True
                        else:
                            new_sub_values.append(sub_p.latent_value)
                    
                    if updates_found:
                        new_val_flat = jnp.array(new_sub_values)
                        new_val_shaped = new_val_flat.reshape(parent_param.latent_value.shape)
                        
                        # UPDATED: Use .with_value() instead of dataclasses.replace
                        params[parent_name] = parent_param.with_value(new_val_shaped)            
        
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
            if isinstance(value, Parameter):
                new_params[name] = value
            else:
                # UPDATED: Route primitive/array updates through .with_value()
                new_params[name] = new_params[name].with_value(jnp.asarray(value))
                
        # Fast tree mapping bypasses string iteration logic completely
        def map_fn(path, node):
            if is_valid_param(node):
                name = self.path_to_param_name(path)
                if name in new_params:
                    new_param = new_params[name]
                    if new_param.name is None:
                        new_param = dataclasses.replace(new_param, name=node.name)
                    return new_param
            return node
            
        return jax.tree_util.tree_map_with_path(map_fn, self, is_leaf=is_valid_param)

    def with_mapped_params(
        self: Self, 
        mapper: Callable[[Parameter], Parameter], 
        param_filter: str | Sequence[str] | Parameter | Sequence[Parameter] | Callable[[str], bool] | None = None, 
        *, 
        map_others: Callable[[Parameter], Parameter] | None = None,
        prefixes: bool = False,
        include_fixed: bool = False,
        ignore_unknown: bool = False,
    ) -> Self:
        """Return a module with specified parameters mapped.

        Parameters
        ----------
        mapper : Callable[[Parameter], Parameter]
            The map to apply to each parameter in the filter (or all if no filter).
        param_filter : str | Sequence[str] | Callable[[str], bool] | None, default=None
            Parameter names to map. If None, applies mapper to all parameters.
        map_others : Callable[[Parameter], Parameter] | None, default=None
            An optional map to apply to all parameters NOT in the filter.
        prefixes : bool, default=False
            Specifies that, when a string or list of strings is passed
            in `param_filter`, these must be interpreted as parameter prefixes
            to map and not full path names. Defaults to `False.`            

        Returns
        -------
        Self
        """
        current_param_names = set(self.param_names(include_fixed=include_fixed))
        
        if param_filter is None:
            resolved_filter = current_param_names
        elif isinstance(param_filter, Callable):
            resolved_filter = {p for p in current_param_names if param_filter(p)}
        else:
            # Safely cast single items or dicts to lists
            if isinstance(param_filter, str):
                param_filter = [param_filter]
            elif isinstance(param_filter, Parameter):
                param_filter = [param_filter]
            elif isinstance(param_filter, dict):
                param_filter = list(param_filter.keys())
            else:
                param_filter = list(param_filter)

            # Safely check index 0 only if the list has elements
            if param_filter and isinstance(param_filter[0], str) and prefixes:
                for prefix in param_filter:
                    if not any(name.startswith(prefix) for name in current_param_names):
                        if not ignore_unknown:
                            raise ValueError(f"Specified prefix '{prefix}' does not match any parameters in the module")
                valid_prefixes = tuple(param_filter)
                resolved_filter = {p for p in current_param_names if p.startswith(valid_prefixes)}
                
            elif param_filter and isinstance(param_filter[0], Parameter):
                param_ids = {id(p) for p in param_filter}
                resolved_filter = {name for name, p in self.named_params(include_fixed=include_fixed).items() if id(p) in param_ids}
                
            else:
                resolved_filter = set(param_filter)
                
            for param_name in resolved_filter:
                if param_name not in current_param_names:
                    raise ValueError(f"Specified parameter '{param_name}' not found in module")
        
        # Directly map using JAX natively
        def map_fn(path, node):
            if is_valid_param(node):
                if not include_fixed and getattr(node, "fixed", False):
                    return node
                name = self.path_to_param_name(path)
                if name in resolved_filter:
                    return mapper(node)
                elif map_others is not None:
                    return map_others(node)
            return node
            
        return jax.tree_util.tree_map_with_path(map_fn, self, is_leaf=is_valid_param)
    
    def with_transformed_params(
        self: Self, 
        bijector: AbstractBijector, 
        param_filter: str | Sequence[str] | Parameter | Sequence[Parameter] | Callable[[str], bool] | None = None, 
        **kwargs
    ) -> Self:
        """
        Return a module with a distreqx bijector applied to the specified parameters.

        This utilizes the underlying `transformed` method on the matched Parameters, 
        which updates their physical values, bounds, and distributions simultaneously 
        while preserving the unconstrained latent values.

        Parameters
        ----------
        bijector : distreqx.bijectors.AbstractBijector
            The bijector to apply.
        param_filter : str | Sequence[str] | Callable[[str], bool] | None, default=None
            Parameter names to transform. If None, applies to all parameters.

        Returns
        -------
        Self
        """
        return self.with_mapped_params(
            mapper=lambda p: p.transformed(bijector), 
            param_filter=param_filter, 
            **kwargs
        )    
        
    def with_fixed_params(self: Self, param_filter: str | Sequence[str] | Parameter | Sequence[Parameter] | Callable[[str], bool], free_others: bool = False, **kwargs) -> Self:
        """Return a module with specified parameters fixed.

        This maps each parameter in the filter, calling [`parax.Parameter.as_fixed`][] on each.

        See [`parax.Module.with_mapped_params`][].

        Parameters
        ----------
        free_others : bool, default=False
            Also free all parameters not in the filter.        

        Returns
        -------
        Self
        """
        map_others = None
        if free_others:
            map_others = lambda p: p.as_free()

        kwargs.setdefault('include_fixed', True) 
        
        return self.with_mapped_params(lambda p: p.as_fixed(), param_filter=param_filter, map_others=map_others, **kwargs)
    
    def with_free_params(self: Self, param_filter: str | Sequence[str] | Parameter | Sequence[Parameter] | Callable[[str], bool], *, fix_others: bool = False, **kwargs) -> Self:
        """Free the specified parameters.

        This maps each parameter in the filter, calling [`parax.Parameter.as_free`][] on each.

        See [`parax.Module.with_mapped_params`][].

        Parameters
        ----------
        fix_others : bool, default=False
            Also fix all parameters not in the filter.

        Returns
        -------
        Self
        """
        map_others = None
        if fix_others:
            map_others = lambda p: p.as_fixed()

        kwargs.setdefault('include_fixed', True) 

        return self.with_mapped_params(lambda p: p.as_free(), param_filter=param_filter, map_others=map_others, **kwargs)
    
    def with_free_params_only(self: Self, param_filter: str | list[str] | Callable[[str], bool], **kwargs) -> Self:
        """Returns a module with only the specified parameters freed.
        
        This is an alias for calling [`parax.Module.with_free_params`][]
        with `fix_others=True`.

        See [`parax.Module.with_free_params`][].
        """
        kwargs.setdefault('fix_others', True)
        if kwargs['fix_others'] == False:
            raise Exception("Cannot pass fix_others == False for `with_free_params_only`.")
        return self.with_free_params(param_filter, **kwargs)

    def with_all_params_fixed(self: Self, **kwargs) -> Self:
        """Returns a module with all parameters fixed.
        
        This is an alias for calling [`parax.Module.with_free_params`][]
        with `fix_others=True` and no parameters passed.

        See [`parax.Module.with_free_params`][].
        """
        kwargs.setdefault('fix_others', True)
        if kwargs['fix_others'] == False:
            raise Exception("Cannot pass fix_others == False for `with_all_params_fixed`.")
        return self.with_free_params({}, **kwargs)

    def with_all_params_free(self: Self, **kwargs) -> Self:
        """Returns a module with all parameters free.
        
        This is an alias for calling [`parax.Module.with_free_params`][]
        with all parameters passed.

        See [`parax.Module.with_free_params`][].
        """
        kwargs.setdefault('include_fixed', True)
        if kwargs['include_fixed'] == False:
            raise Exception("Cannot pass include_fixed == False for `with_all_params_free`.")        
        return self.with_free_params(self.param_names(include_fixed=True), **kwargs)
    
    # ---- Parameter group manipulation --------------------------------------------------            
    
    def with_param_groups(self: Self, param_groups: ParameterGroup | list[ParameterGroup]) -> Self:
        """Return a module with parameter groups appended, replacing existing relationships.
        
        This method implements an "atomic replacement" policy. If *any* parameter in 
        an existing group is claimed by a new group, the *entire* existing group is 
        removed. 
        
        This ensures that groups defining joint distributions are not left in an 
        invalid broken state (e.g. having a dimension removed). Parameters that were 
        in the removed group but not in the new group will revert to being ungrouped 
        (handled by `param_groups` as singleton groups).

        Parameters
        ----------
        param_groups : ParameterGroup or list[ParameterGroup]
            Group(s) to add.

        Returns
        -------
        Self
        """       
        if not isinstance(param_groups, list):
            param_groups = [param_groups]
        
        new_claimed_params = set()
        for group in param_groups:
            new_claimed_params.update(group.param_names)

        current_groups = self._param_groups if self._param_groups is not None else []
        kept_existing_groups = []
        
        for group in current_groups:
            existing_group_params = set(group.param_names)
            if existing_group_params.isdisjoint(new_claimed_params):
                kept_existing_groups.append(group)

        new_list = kept_existing_groups + param_groups
        new_module = copy(self)
        object.__setattr__(new_module, '_param_groups', new_list)
        return new_module
    
    def with_demoted_param_groups(self: Self) -> Self:
        """Recursively demote parameter groups to the deepest possible submodule.

        This method identifies parameter groups where every parameter belongs to the same 
        immediate submodule. It moves those groups to the submodule, stripping the prefix.
        It then recursively calls this method on the submodules to ensure groups continue 
        moving down the hierarchy as far as possible.

        Returns
        -------
        Self
            A new module instance with parameter groups distributed to their lowest 
            relevant submodules.
        """
        submodule_prefixes = {} 
        for f in dataclasses.fields(self):
            if isinstance(getattr(self, f.name), Module):
                prefix = f.name + self._separator
                submodule_prefixes[prefix] = f.name

        groups_to_keep = []
        submodule_groups = {name: [] for name in submodule_prefixes.values()}
        
        current_groups = self._param_groups if self._param_groups is not None else []

        for group in current_groups:
            demoted = False
            for prefix, field_name in submodule_prefixes.items():
                if all(name.startswith(prefix) for name in group.param_names):
                    new_names = [name[len(prefix):] for name in group.param_names]
                    new_group = dataclasses.replace(group, param_names=new_names)
                    submodule_groups[field_name].append(new_group)
                    demoted = True
                    break
            
            if not demoted:
                groups_to_keep.append(group)

        new_fields = {}
        for prefix, field_name in submodule_prefixes.items():
            child_module: Module = getattr(self, field_name)
            
            groups_to_push = submodule_groups[field_name]
            if groups_to_push:
                child_module = child_module.with_param_groups(groups_to_push)
            
            child_module = child_module.with_demoted_param_groups()
            new_fields[field_name] = child_module

        new_module = self.with_fields(**new_fields)
        object.__setattr__(new_module, '_param_groups', groups_to_keep)
        return new_module
    
    def with_no_param_groups(self: Self) -> Self:
        """Return a new module with all parameter groups removed recursively.

        This clears the `_param_groups` of the current module and traverses
        all nested submodules (and sequences of submodules) to remove their 
        parameter groups as well.

        Returns
        -------
        Self
            A new module instance with no parameter groups.
        """
        new_fields = {} 
        
        for f in dataclasses.fields(self):
            if f.name == '_param_groups':
                continue
                
            child = getattr(self, f.name)
            
            if isinstance(child, Module):
                new_fields[f.name] = child.with_no_param_groups()
                
            elif isinstance(child, (list, tuple)):
                if any(isinstance(x, Module) for x in child):
                    new_fields[f.name] = type(child)(
                        x.with_no_param_groups() if isinstance(x, Module) else x 
                        for x in child
                    )
                    
        new_module = self.with_fields(**new_fields)
        object.__setattr__(new_module, '_param_groups', [])
        return new_module
    
    # ---- Distribution manipulation --------------------------------------------------

    def with_mapped_distributions(
        self: Self, 
        mapper: Callable[[AbstractDistribution], AbstractDistribution], 
        dist_filter: Callable[[AbstractDistribution], bool] | None = None, 
        *, 
        map_others: Callable[[AbstractDistribution], AbstractDistribution] | None = None,
        param_groups: bool = False
    ) -> Self:
        """Return a module with a function applied to its parameter distributions.

        This method allows for bulk-updates of distributions, such as widening variances 
        or changing distribution types.

        If ``param_groups`` is False, the mapping is applied to the distributions 
        of individual parameters (flattened).

        If ``param_groups`` is True, the mapping is applied to the distributions 
        of [`parax.ParameterGroup`][] objects. This mode is recursive: it will traverse 
        the module tree and apply the mapping to all explicit parameter groups in all submodules.

        Parameters
        ----------
        mapper : Callable[[AbstractDistribution], AbstractDistribution]
            Function that takes a distribution and returns a new one.
        dist_filter : Callable[[AbstractDistribution], bool] | None, default=None
            A predicate function. If provided, the mapping is only applied to 
            distributions where ``dist_filter(dist)`` is True. If None, applies to all.
        map_others : Callable[[AbstractDistribution], AbstractDistribution] | None, default=None
            An optional map to apply to all distributions NOT in the filter.
        param_groups : bool, default=False
            If True, map distributions on parameter groups (recursively). 
            If False, map distributions on individual parameters (flat).

        Returns
        -------
        Self
            A new module with updated distributions.
        """
        mapped_module = self

        if param_groups:
            current_groups = self._param_groups if self._param_groups is not None else []
            for group in current_groups:
                if dist_filter is None or dist_filter(group.distribution):
                    mapped_module = mapped_module.with_param_groups(group.with_distribution(mapper(group.distribution)))
                elif map_others is not None:
                    mapped_module = mapped_module.with_param_groups(group.with_distribution(map_others(group.distribution)))

            new_submodules = {}
            for f in dataclasses.fields(mapped_module):
                child = getattr(mapped_module, f.name)
                if isinstance(child, Module):
                    updated_child = child.with_mapped_distributions(
                        mapper, 
                        dist_filter, 
                        map_others=map_others, 
                        param_groups=True
                    )
                    new_submodules[f.name] = updated_child
            
            if new_submodules:
                mapped_module = mapped_module.with_fields(**new_submodules)

        else:
            def map_fn(node):
                if is_valid_param(node):
                    if dist_filter is None or dist_filter(node.distribution):
                        return node.with_distribution(mapper(node.distribution))
                    elif map_others is not None:
                        return node.with_distribution(map_others(node.distribution))
                return node
                
            mapped_module = jax.tree_util.tree_map(map_fn, self, is_leaf=is_valid_param)
                    
        return mapped_module
    
    def with_uniform_distributions(self, percentage: float, param_filter: str | Sequence[str] | Parameter | Sequence[Parameter] | Callable[[str], bool] = None, *, respect_bounds=False, remove_param_groups=True, zero_values='keep', **kwargs) -> Self:
        """Return a module with uniform distributions set centered on current parameter values.

        The distributions are defined with bounds calculated as ``value * (1.0 +/- percentage)``.

        Parameters
        ----------
        percentage : float
            The fractional width of the uniform distribution (e.g. 0.1 = 10%).
        param_filter: str | Sequence[str] | Callable[[str], bool], default=None
            The parameters to be updated with new uniform distributions. For the default case, all are updated.
        respect_bounds: bool, default=False
            Whether or not the `min` and `max` bounds of the current distributions should be respected.
            If `True`, new bounds will not go larger than past these bounds.
        remove_param_groups: bool, default=True
            Whether to remove parameter groups recursively when setting the uniform distributions.
            Otherwise, the joint distribution of the module may not be the desired uniform distribution.
        zero_values: str, default='keep'
            How to treat zero values. Currently the only option is to keep them and their bounds as is.

        Returns
        -------
        Self
            A new module with updated parameter distributions.
        """        
        current_param_names = set(self.param_names(param_filter, **kwargs))
        
        def map_fn(path, param: Parameter):
            if is_valid_param(param):
                name = self.path_to_param_name(path)
                if name in current_param_names:
                    value = jnp.asarray(param.value)
                    
                    # Calculate bounds element-wise natively in JAX
                    base_min = jnp.where(value > 0.0, value * (1.0 - percentage), value * (1.0 + percentage))
                    base_max = jnp.where(value > 0.0, value * (1.0 + percentage), value * (1.0 - percentage))
                    
                    if zero_values == 'keep':
                        new_min = jnp.where(value == 0.0, 0.0, base_min)
                        new_max = jnp.where(value == 0.0, 0.0, base_max)
                    else:
                        raise Exception("Unknown option for 'zero_values'")
                    
                    if respect_bounds:
                        param_min = getattr(param, 'min', -jnp.inf) if param.bounds is None else param.bounds[0]
                        param_max = getattr(param, 'max', jnp.inf) if param.bounds is None else param.bounds[1]
                        new_min = jnp.maximum(new_min, param_min)
                        new_max = jnp.minimum(new_max, param_max)

                    distribution = UniformDistribution(new_min, new_max)
                    return param.with_distribution(distribution)
            return param

        new_module = jax.tree_util.tree_map_with_path(map_fn, self, is_leaf=is_valid_param)
        if remove_param_groups:
            new_module = new_module.with_no_param_groups()
        return new_module
    
    # ---- Field and module manipulation --------------------------------------------------            
    
    @classmethod
    def with_defaults(cls, *args, **kwargs) -> type[Self]:
        """Return this module type with default initialization arguments.
        
        This method is very useful in utilizing an existing module
        with default values, without having to create a new
        module type via inheritance.

        Arguments are forwarded as if they were passed to `__init__`.

        Returns
        -------
        type[Module]
        """            
        class DefaultsWrapper:
            def __init__(self, p):
                self.p = p   

            def __call__(self, *call_args, **call_kwargs):
                baked_args = deepcopy(self.p.args)
                baked_kwargs = deepcopy(self.p.keywords)
                
                final_args = baked_args + call_args
                final_kwargs = {**baked_kwargs, **call_kwargs}
                
                return self.p.func(*final_args, **final_kwargs)

            def with_defaults(self, *new_args, **new_kwargs):
                merged_args = self.p.args + new_args
                merged_kwargs = {**self.p.keywords, **new_kwargs} if self.p.keywords else new_kwargs
                return DefaultsWrapper(partial(self.p.func, *merged_args, **merged_kwargs))
                
        return DefaultsWrapper(partial(cls, *args, **kwargs))    
    
    def with_fields(self: Self, *args, **kwargs) -> Self:
        """
        Return a copy of this module with dataclass-style field replacements.

        Parameters are forwarded to `dataclasses.replace`.
        """
        new_module = dataclasses.replace(self, *args, **kwargs)
        
        for f in dataclasses.fields(self):
            if not f.init:
                val = getattr(self, f.name)
                object.__setattr__(new_module, f.name, deepcopy(val))
                
        return new_module
    
    def with_attrs(self: Self, *args: Any, **kwargs: Any) -> Self:
        """
        Return a copy of the module with one or more attributes replaced.

        This is similar to `eqx.tree_at` but uses string paths.

        Usage
        -----
        # 1. Single attribute update (path, value)
        model.with_attrs('a.b.c', 10)

        # 2. Batch nested updates via dictionary
        model.with_attrs({'a.b.c': 10, 'x.y.z': 20})
        
        # 3. Top-level attributes via keyword arguments
        model.with_attrs(name="new_model", _transparent=True)
        
        # 4. Combined dict and kwargs
        model.with_attrs({'a.b.c': 10}, name="new_model")
        """
        all_updates = {}

        # Parse positional arguments
        if len(args) == 2 and isinstance(args[0], str):
            all_updates[args[0]] = args[1]
        elif len(args) == 1 and isinstance(args[0], dict):
            all_updates.update(args[0])
        elif len(args) > 0:
            raise ValueError(
                "Invalid positional arguments. Please provide either a single "
                "(path, value) pair or a dictionary of updates."
            )

        # Add any top-level kwargs
        all_updates.update(kwargs)

        # Fast exit if nothing to update
        if not all_updates:
            return self

        # Extract paths and their corresponding values in a consistent order
        paths = tuple(all_updates.keys())
        values = tuple(all_updates.values())
        
        # CORRECTED: A single callable that returns a tuple of nodes
        def where_fn(tree):
            return tuple(operator.attrgetter(p)(tree) for p in paths)
        
        return eqx.tree_at(where_fn, self, values)
    
    def with_submodules(self: Self, *args: Any, **kwargs: Any) -> Self:
        """
        Return a copy of the module with one or more submodules replaced.

        This method accepts paths formatted in the exact same way as parameter names 
        (e.g. 'submodule1_submodule2_submodule3'), respecting transparency and custom names.

        Usage
        -----
        # Single replacement
        model.with_submodules('layer1_attention', new_attention_module)

        # Batch replacement
        model.with_submodules({
            'layer1_attention': new_attn_1,
            'layer2_attention': new_attn_2
        })
        """
        all_updates = {}

        # Parse positional arguments
        if len(args) == 2 and isinstance(args[0], str):
            all_updates[args[0]] = args[1]
        elif len(args) == 1 and isinstance(args[0], dict):
            all_updates.update(args[0])
        elif len(args) > 0:
            raise ValueError(
                "Invalid positional arguments. Please provide either a single "
                "(path, value) pair or a dictionary of updates."
            )

        # Add any top-level kwargs
        all_updates.update(kwargs)

        if not all_updates:
            return self

        # 1. Gather all submodules and map their string paths to absolute JAX paths
        name_to_jax_path = {}
        
        def traverse(node, current_path):
            # Flatten exactly one module-level deep to get paths to immediate submodules
            leaves, _ = jax.tree_util.tree_flatten_with_path(
                node, 
                is_leaf=lambda x: isinstance(x, Module) and x is not node
            )
            for sub_path, leaf in leaves:
                if isinstance(leaf, Module):
                    full_path = current_path + sub_path
                    # Map the absolute JAX path to your custom string format
                    str_name = self.path_to_param_name(full_path)
                    name_to_jax_path[str_name] = full_path
                    # Recurse into the nested module
                    traverse(leaf, full_path)
                    
        traverse(self, ())

        # 2. Match requested updates to JAX paths
        paths_to_update = []
        values_to_update = []
        
        for str_name, new_module in all_updates.items():
            if str_name not in name_to_jax_path:
                raise ValueError(f"Submodule path '{str_name}' not found in module.")
            paths_to_update.append(name_to_jax_path[str_name])
            values_to_update.append(new_module)

        # 3. Create an extractor callable for eqx.tree_at using the JAX paths
        def where_fn(tree):
            extracted = []
            for jax_path in paths_to_update:
                node = tree
                for key in jax_path:
                    if isinstance(key, GetAttrKey):
                        node = getattr(node, key.name)
                    elif isinstance(key, DictKey):
                        node = node[key.key]
                    elif isinstance(key, (SequenceKey, FlattenedIndexKey)):
                        # Compatibility for different JAX versions
                        idx = getattr(key, 'idx', getattr(key, 'key', None))
                        node = node[idx]
                extracted.append(node)
            return tuple(extracted)
            
        # Execute the atomic replacement
        return eqx.tree_at(where_fn, self, tuple(values_to_update))    
    
    def with_name(self: Self, name: str | None) -> Self:
        """
        Return a copy of this module with a different name.
        """
        return self.with_fields(name=name)
    
    def with_submodule_fields(self: Self, submodule: str | Sequence[str], *args, **kwargs) -> Self:
        """
        Return a copy of this module with dataclass-style field replacements on a nested sub-module.

        Parameters are forwarded to `dataclasses.replace`.

        Parameters
        ----------
        submodule : str | Sequence[str]
            The name of the submodule (or sequence of names) to traverse.
            Can be a single string with a path e.g. 'submodule1.submodule2',
            or a list of submodules e.g. ['submodule1', 'submodule2'].
        """
        if isinstance(submodule, str) and submodule.find('.'):
            path = submodule.split('.')
        else:
            path = [submodule] if isinstance(submodule, str) else list(submodule)
        
        if not path:
            return self.with_fields(*args, **kwargs)

        target_key = path[0]

        if len(path) == 1:
            updated_child = getattr(self, target_key).with_fields(*args, **kwargs)
        else:
            child = getattr(self, target_key)
            updated_child = child.with_submodule_fields(path[1:], *args, **kwargs)

        return self.with_fields(**{target_key: updated_child})  
    
    def with_free_submodules(self: Self, submodules: 'Module' | Sequence['Module'] | str | Sequence[str], fix_others=False, include_fixed=True) -> Self:
        """Free all parameters in the given submodules.

        Submodules parameters are obtained using [`parax.Module.param_names`][].,
        and subsequently freed using [`parax.Module.with_free_params`][].
        
        Parameters
        ----------
        submodules : Module | Sequence[Module] | str | Sequence[str]
            Submodules whose parameters should be free.
        include_fixed : bool, default=True
            Include fixed parameters in the submodule.
        fix_others : bool, default=False
            Fix all other submodules.

        Returns
        -------
        Self
        """        
        module_param_names = self.param_names(include_fixed=include_fixed, submodules=submodules)
        return self.with_free_params(module_param_names, fix_others=fix_others)
    
    def with_free_submodules_only(self: Self, *args, include_fixed=False, **kwargs) -> Self:
        """Returns a module with only the specified submodules freed.
        
        This is an alias for calling [`parax.Module.with_free_submodules`][]
        with `fix_others=True` and `include_fixed=False` by default.

        See [`parax.Module.with_free_params`][].
        """     
        kwargs.setdefault('fix_others', True)
        if kwargs['fix_others'] == False:
            raise Exception("Cannot pass fix_others == False for `with_free_submodules_only`.")
        return self.with_free_submodules(*args, include_fixed=include_fixed, **kwargs)
    
    def with_fixed_submodules(self: Self, submodules: 'Module' | Sequence['Module'] | str | Sequence[str]) -> Self:
        """Fix all parameters in the given submodules.

        Submodules parameters are obtained using [`parax.Module.param_names`][].,
        and subsequently fixed using [`parax.Module.with_fixed_params`][].
        
        Parameters
        ----------
        submodules : Module | Sequence[Module] | str | Sequence[str]
            Submodules whose parameters should be fixed.

        Returns
        -------
        Self
        """        
        module_param_names = self.param_names(include_fixed=True, submodules=submodules)
        return self.with_fixed_params(module_param_names)
    
    # ---- Function tools --------------------------------------------------        

    @eqx.filter_jit
    def func_jacobian(
        self: Self, 
        func: Callable[['Module'], jnp.ndarray], 
        args: Any
    ) -> dict[str, jnp.ndarray]:
        """Calculate the Jacobian of an arbitrary function with respect to free parameters.

        This uses forward-mode automatic differentiation to compute the gradients 
        of the provided function with respect to each free parameter in the module.

        Parameters
        ----------
        func : Callable[[Module], jnp.ndarray]
            Function to differentiate. Must take a Module and args
            and return a jnp.ndarray of any shape.
        args : Any
            The args to pass to `func`.

        Returns
        -------
        dict[str, jnp.ndarray]
            A dictionary mapping flat parameter names to their gradient 
            arrays. Each array has the same shape as the output of `func`.
        """
        def func_from_flat(flat_params_array: jnp.ndarray) -> jnp.ndarray:
            sampled_module = self.with_params(flat_params_array)
            return func(sampled_module, args)

        jac_array = jax.jacfwd(func_from_flat)(self.flat_param_values())
        jac_moved = jnp.moveaxis(jac_array, -1, 0)
        param_names = self.flat_param_names()
        
        return {name: jac_moved[i] for i, name in enumerate(param_names)}
    
    @eqx.filter_jit
    def func_sensitivity(
        self: Self, 
        func: Callable[['Module'], jnp.ndarray], 
        args: Any,
        kind: str = 'relative',
        norm: int | str | None = None
    ) -> dict[str, jnp.ndarray] | jnp.ndarray:
        r"""Calculate the sensitivity of an arbitrary function w.r.t parameters.

        Supported kinds:
        - 'relative': (dy/dtheta) * (theta/y). Fractional change in output per 
          fractional change in parameter. Blows up if y is zero.
        - 'semi-relative': (dy/dtheta) * theta. Change in output per 
          fractional change in parameter. Stable if y is zero.
        - 'absolute': (dy/dtheta). Raw gradient.

        Parameters
        ----------
        func : Callable[[Module], jnp.ndarray]
            Function to evaluate.
        args : Any
            The args to pass to `func`.
        kind : str, default='relative'
            The type of sensitivity to calculate ('relative', 'semi-relative', 'absolute').
        norm : int | str | None, default=None
            If provided, aggregates the parameter sensitivities into a single scalar 
            metric using the specified norm (e.g., 2 for L2 norm, jnp.inf for max norm).

        Returns
        -------
        dict[str, jnp.ndarray] | jnp.ndarray
            If `norm` is None, returns a dictionary mapping flat parameter names 
            to sensitivity arrays.
            If `norm` is specified, returns a 0D scalar jax array representing 
            the global sensitivity metric.
        """
        def func_from_flat(flat_params_array: jnp.ndarray) -> jnp.ndarray:
            sampled_module = self.with_params(flat_params_array)
            return func(sampled_module, args)

        theta = self.flat_param_values()
        jac_array = jax.jacfwd(func_from_flat)(theta)
        
        if kind == 'absolute':
            sens_array = jac_array
            
        elif kind == 'semi-relative':
            sens_array = jac_array * theta
            
        elif kind == 'relative':
            y_nom = func(self, args)
            y_safe = jnp.where(y_nom == 0, 1e-15, y_nom)
            sens_array = jac_array * (theta / y_safe[..., None])
            
        else:
            raise ValueError(f"Unknown sensitivity kind: '{kind}'. "
                             f"Supported: 'relative', 'semi-relative', 'absolute'") 
        
        if norm is not None:
            return jnp.linalg.norm(sens_array, ord=norm)
            
        sens_moved = jnp.moveaxis(sens_array, -1, 0)
        param_names = self.flat_param_names()
        
        return {name: sens_moved[i] for i, name in enumerate(param_names)}

    @eqx.filter_jit
    def func_samples(
        self, 
        func: Callable[['Module'], jnp.ndarray], 
        args: Any,
        *,
        key: jax.Array, 
        num_samples: int = 1000
    ) -> jnp.ndarray:
        """
        Evaluates an arbitrary function over samples drawn from the 
        module's distribution.

        Parameters
        ----------
        func : Callable[[Module], jnp.ndarray]
            A function that takes a Module instance and returns a JAX array.
        args : Any
            The args to pass to `func`.            
        key : jax.Array
            JAX random key for sampling.
        num_samples : int, default=1000
            Number of modules to sample from the joint distribution.

        Returns
        -------
        jnp.ndarray
            The function evaluated over all samples. Shape will be 
            (num_samples, *func_output_shape).
        """
        dist = self.flat_distribution()
        flat_param_samples = dist.sample(key, sample_shape=(num_samples,))

        def evaluate_single(flat_params_array):
            sampled_module = self.with_params(flat_params_array)
            return func(sampled_module, args)

        return jax.vmap(evaluate_single)(flat_param_samples)  