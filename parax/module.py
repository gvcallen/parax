"""
The main module class.

This module defines :class:`parax.Module`, a frozen, JAX-compatible, Equinox module.

"""

import inspect
from functools import partial
from copy import copy, deepcopy
from typing import Callable, Sequence, Iterator, Self, ClassVar, Any
import dataclasses
from dataclasses import fields, is_dataclass
from functools import update_wrapper
from typing_extensions import dataclass_transform

import jax
import jax.numpy as jnp
from jax import flatten_util
from jax.tree_util import GetAttrKey, DictKey, SequenceKey, FlattenedIndexKey
import equinox as eqx
import numpyro.distributions as dist
from numpyro.distributions import Distribution

from parax.parameter import Parameter, is_valid_param, as_param
from parax.parameter_group import ParameterGroup
from parax.field import field
from parax.partition import partition
from parax.distributions import JointDistribution
from parax.utils import get_first_underlying_type, is_convertible_to_float, nodes_by_type

@dataclass_transform(field_specifiers=(field, eqx.field, dataclasses.field))
class ModuleMeta(type(eqx.Module)):
    def __new__(mcs, name, bases, namespace, **kwargs):
        annotations = namespace.get('__annotations__', {})
        
        for field_name, field_types in annotations.items():
            field_kwargs = {}
            default = namespace.get(field_name, dataclasses.MISSING)
            
            field_type = get_first_underlying_type(field_types)
            is_param_type = field_type is not None and isinstance(field_type, type) and issubclass(field_type, Parameter)
            
            # 1. Handle explicit Field declarations (e.g., x = prf.field(...))
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
                        
                        # USE CUSTOM FIELD: Repackage the explicitly defined field using prf.field
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
    Overview
    --------
    The main module class.

    Derive from this class to define your parametric Model.

    The module is an Equinox ``Module`` (immutable, dataclass-like) and is
    treated as a JAX PyTree. Parameters are declared using standard dataclass
    field syntax with types like :class:`parax.Parameter`.

    Usage
    -----
    - Define your model by sub-classing the module and adding custom parameters and/or sub-modules.
    - Construct modules by passing parameters and/or submodules to the initializer (like a dataclass).
    - Retrieve parameter information via methods such as :meth:`.named_params`, :meth:`.param_names`, :meth:`.flat_params`, etc..
    - Use `with_xxx` functions to modify fields, modules and parameters within the module e.g. :meth:`.with_params`, :meth:`.with_fields`.

    Methods & Properties Summary
    ----------------------------

    **Introspection Properties**
    
    ================================= ====================================================================
    Method / Property                 Description
    ================================= ====================================================================
    :attr:`DEFAULT_NAMED_PARAMS`      Mapping from parax.Parameter name to :class:`Parameter`.
    :attr:`DEFAULT_PARAM_NAMES`       Default parameter names for the module.
    :attr:`DEFAULT_PARAMS`            Default parameters for the module.
    :attr:`num_params`                Number of free parameters.
    :attr:`num_flat_params`           Number of free, flattened parameters.
    ================================= ====================================================================

    **Function Tools**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`func_jacobian`             Calculate the Jacobian of a function w.r.t parameters.
    :meth:`func_sensitivity`          Calculate the sensitivity of a function w.r.t parameters.
    :meth:`func_samples`              Evaluate a function over parameter samples.
    ================================= ====================================================================

    **Module Inspection & Manipulation**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`children`                  Returns the immediate submodules.
    :meth:`submodules`                Returns all nested submodules (depth-first).
    :meth:`partition`                 Partition module into parameters and static trees.
    :meth:`sampled`                   Return a new module with parameters drawn from this module's distribution.
    ================================= ====================================================================

    **Parameter Inspection**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`named_params`              Named module parameter objects as a dict.
    :meth:`named_param_values`        Named module parameter values as a dict of jax arrays.
    :meth:`param_names`               Module parameter names as a list.
    :meth:`param`                     A single module parameter object by name.
    :meth:`params`                    Module parameters as a list.
    :meth:`param_value`               A single module parameter value by name.
    :meth:`param_values`              Module parameter values as a list of jax arrays.
    :meth:`named_flat_params`         Named flattened module parameter objects as a dict.
    :meth:`named_flat_param_values`   Named flattened module parameter values as a dict.
    :meth:`flat_param_names`          Flattened parameter names as a list.
    :meth:`flat_params`               Flattened parameters as a list.
    :meth:`flat_param_values`         Flattened module parameter values as a flat array.
    :meth:`flat_param_bounds`         Flattened module parameter bounds as jax arrays.
    :meth:`param_groups`              Return all parameter groups relevant to this module.
    :meth:`distribution`              Joint distribution over (flattened) parameters.
    ================================= ====================================================================

    **Parameter Manipulation**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`with_params`               Return a module with parameters updated.
    :meth:`with_mapped_params`        Apply a map function to parameters.
    :meth:`with_fixed_params`         Return a module with specified parameters fixed.
    :meth:`with_free_params`          Return a module with specified parameters free.
    :meth:`with_free_params_only`     Return a module with ONLY the specified parameters free.
    :meth:`with_all_params_fixed`     Return a module with all parameters fixed.
    :meth:`with_all_params_free`      Return a module with all parameters free.
    ================================= ====================================================================

    **Parameter Group Manipulation**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`with_param_groups`         Return a module with parameter groups appended.
    :meth:`with_demoted_param_groups` Recursively demote parameter groups to deepest submodule.
    :meth:`with_no_param_groups`      Return a module with all parameter groups removed.
    ================================= ====================================================================

    **Distribution Manipulation**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`with_mapped_distributions` Apply a map function to the parameter distributions.
    :meth:`with_uniform_distributions` Return a module with uniform distributions set.
    ================================= ====================================================================
    
    **Field & Module Manipulation**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`with_defaults`             Return this module type with default initialization args.
    :meth:`with_modules`              Combines this module with free parameters in other modules.
    :meth:`with_fields`               Return a copy with dataclass-style field replacements.
    :meth:`with_name`                 Return a copy of this module with a different name.
    :meth:`with_submodule_fields`     Dataclass-style field replacements on a nested sub-module.
    :meth:`with_free_submodules`      Free all parameters in the given submodules.
    :meth:`with_free_submodules_only` Returns a module with ONLY the specified submodules freed.
    :meth:`with_fixed_submodules`     Fix all parameters in the given submodules.
    :meth:`with_tied_submodules`      Tie submodules structurally to a shared module.
    :meth:`tied`                      Return the module with self tied to a shared module.
    :meth:`with_injected_params`      Inject parameters from a shared module into target submodules.
    ================================= ====================================================================

    Attributes
    ----------
    name : str or None
        An optional name for the module instance.
    param_groups : str
        Parameter groups to initialize the module with.

    """
    # Public init fields
    name: str | None = field(default=None, kw_only=True, static=True)
    _param_groups: list[ParameterGroup] = field(default_factory=list, kw_only=True, repr=False, static=True, init=False)
    
    # Claa variables
    _separator: ClassVar[bool] = '_'
    _transparent: ClassVar[bool] = False

    # ---- Internal initialization methods -------------------------------------------------

    def __init_subclass__(cls, transparent: bool = False, **kwargs):
        """Customize subclass construction."""        
        super().__init_subclass__(**kwargs)

        cls._transparent = transparent

        if '__init__' in cls.__dict__:
            user_init = cls.__dict__['__init__']
            sig = inspect.signature(user_init)
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            
            def wrapped_init(self, *args, **init_kwargs):
                user_kwargs = {}
                base_kwargs = {}
                
                valid_fields = {f.name for f in dataclasses.fields(type(self))}
                for k, v in init_kwargs.items():
                    if accepts_kwargs or k in sig.parameters:
                        user_kwargs[k] = v
                    elif k in valid_fields or k in {"name"}:
                        base_kwargs[k] = v
                    else:
                        raise TypeError(f"{type(self).__name__}.__init__() got an unexpected keyword argument '{k}'")
                
                for k, v in base_kwargs.items():
                    object.__setattr__(self, k, v)
                    
                user_init(self, *args, **user_kwargs)
                
                # SIMPLIFIED: Only apply defaults/converters to fields the user's init missed!
                for f in dataclasses.fields(type(self)):
                    if not hasattr(self, f.name):
                        val = dataclasses.MISSING
                        
                        # Grab the default from the metaclass blueprint
                        if f.default is not dataclasses.MISSING:
                            val = f.default
                        elif f.default_factory is not dataclasses.MISSING:
                            val = f.default_factory()
                        
                        # If a default existed, convert it and set it
                        if val is not dataclasses.MISSING:
                            converter = f.metadata.get("converter") if hasattr(f, "metadata") else None
                            if converter is not None:
                                val = converter(val)
                            object.__setattr__(self, f.name, val)
                
                if hasattr(self, '__post_init__'):
                    self.__post_init__()

            update_wrapper(wrapped_init, user_init)
            cls.__init__ = wrapped_init       

        else:
            user_post_init = getattr(cls, '__post_init__', None)
            
            def wrapped_post_init(self, *args, **kwargs_pi):
                if user_post_init is not None:
                    user_post_init(self, *args, **kwargs_pi)
                
            cls.__post_init__ = wrapped_post_init
            
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
                    # If transparent, we ONLY add a namespace if the user explicitly named the child.
                    # Otherwise, we pass right through without appending the internal variable name.
                    if explicit_name is not None:
                        name_fields.append(explicit_name)
                else:
                    name_fields.append(explicit_name if explicit_name is not None else k)
                        
                node = next_node
                
            elif isinstance(item, DictKey):
                k = item.key
                node = node[k]
                
                # Rule: Dictionaries ALWAYS use the key.
                name_fields.append(str(k))
                    
            elif isinstance(item, (SequenceKey, FlattenedIndexKey)):
                idx = item.idx if hasattr(item, 'idx') else item.key
                node = node[idx]
                
                # Rule: Sequences promote explicitly named modules, 
                # otherwise use just the index "{idx}".
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
                    # Check for our custom save flag
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
                
                # Override non-init fields
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
        params_tree, _ = partition(self, include_fixed=include_fixed, param_objects=True)
        path_and_params, _ = jax.tree.flatten_with_path(params_tree, is_leaf=is_valid_param)
        params: list[tuple[str, Parameter]] = [(self.path_to_param_name(path), param) for path, param in path_and_params]

        # Parameter filtering
        if param_filter is not None:
            # Normalization
            if isinstance(param_filter, str):
                param_filter = [param_filter]

            # Apply filter
            if isinstance(param_filter, Sequence) and isinstance(param_filter[0], str):
                params = [(k, v) for k, v in params if k in param_filter]
            elif isinstance(param_filter, Sequence) and isinstance(param_filter[0], Parameter):
                filter_ids = [id(v) for v in param_filter]
                params = [(k, v) for k, v in params if id(v) in filter_ids]
            elif isinstance(param_filter, Callable):
                params = [(k, v) for k, v in params if param_filter(k)]
            else:
                raise Exception(f"Unknown filter type passed for parameters: {param_filter}")

        # Submodule filtering
        if submodules is not None:
            if isinstance(submodules, (Module, str)):
                submodules: list[Module] = [submodules]
            if isinstance(submodules[0], str):
                submodules: list[Module] = [getattr(self, name) for name in submodules]
            if not isinstance(submodules[0], Module):
                raise Exception(f"Got unknown type when expecting a module or string. Type was: {submodules}")

            allowed = {id(p) for sm in submodules for p in sm.params(include_fixed=include_fixed)}
            params = [(k, v) for k, v in params if id(v) in allowed]

        # Flatten multi-dimensional parameters if requested
        if flatten:
            flat_params: list[tuple[str, Parameter]] = []
            for name, param in params:
                # Updated check: look for list instance instead of flat_names
                if param.size > 1 or isinstance(param.name, list):
                    flattened_params = param.flattened(separator=self._separator)
                    for i, subparam in enumerate(flattened_params):
                        suffix = subparam.name if subparam.name is not None else str(i)
                        flat_params.append((f"{name}{self._separator}{suffix}", subparam))
                else:
                    flat_params.append((name, param))
            params = flat_params

        yield from params
    
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

        # Calculate the Jacobian. By default, JAX appends the parameter dimension 
        # to the end of the output shape: (*func_shape, num_params)
        jac_array = jax.jacfwd(func_from_flat)(self.flat_param_values())
        
        # Move the parameter dimension to the front: (num_params, *func_shape)
        jac_moved = jnp.moveaxis(jac_array, -1, 0)
        
        param_names = self.flat_param_names()
        
        # Map each slice to its corresponding parameter name
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
        # 1. Setup the flat parameter evaluation
        def func_from_flat(flat_params_array: jnp.ndarray) -> jnp.ndarray:
            sampled_module = self.with_params(flat_params_array)
            return func(sampled_module, args)

        theta = self.flat_param_values()
        
        # Calculate raw Jacobian. JAX appends the parameter dimension to the end: 
        # jac_array shape: (*func_shape, num_params)
        jac_array = jax.jacfwd(func_from_flat)(theta)
        
        # 2. Scale the Jacobian entirely via array broadcasting
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
        
        # 3. Apply the norm if requested
        if norm is not None:
            return jnp.linalg.norm(sens_array, ord=norm)
            
        # 4. Otherwise, pack it back into the dictionary format
        # Move the parameter dimension to the front: (num_params, *func_shape)
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
        prng_key : jax.Array
            JAX random key for sampling.
        num_samples : int, default=1000
            Number of modules to sample from the joint distribution.

        Returns
        -------
        jnp.ndarray
            The function evaluated over all samples. Shape will be 
            (num_samples, *func_output_shape).
        """
        # 1. Get the joint distribution and sample it
        dist = self.distribution()
        flat_param_samples = dist.sample(key, sample_shape=(num_samples,))

        # 2. Define the single-sample evaluation, passing freq explicitly
        def evaluate_single(flat_params_array):
            sampled_module = self.with_params(flat_params_array)
            return func(sampled_module, args)

        # 3. Vectorize over the samples
        return jax.vmap(evaluate_single)(flat_param_samples)  
    
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
            # Skip hidden/internal fields like _separator
            if f.repr is False:
                continue
            
            val = getattr(self, f.name)
            val_repr = repr(val)
            
            # Indent multi-line strings (like nested modules) for perfect alignment
            indented_val_repr = val_repr.replace('\n', '\n  ')
            formatted_field = f"  {f.name}={indented_val_repr}"
            
            # Sort into the three buckets:
            
            # 1. Base fields (name) go at the very bottom
            if f.name == "name":
                base_fields.append(formatted_field)
                
            # 2. Modules and Parameters go at the very top
            elif isinstance(val, (Module, Parameter)):
                module_param_fields.append(formatted_field)
                
            # 3. Everything else (bools, ints, floats) goes in the middle
            else:
                other_fields.append(formatted_field)
            
        # Combine the lists in the requested order
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
        dist = self.distribution()
        flat_param_samples = dist.sample(key, sample_shape=(1,))[0]
        return self.with_params(flat_param_samples)
       
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

        See :meth:`.named_params`.

        Parameters
        ----------
        scaled : bool, default=False
            Whether or not to scale the returned values by the parameter scales.
        **kwargs
            Additional key-word arguments as in  :meth:`.named_params`.

        Returns
        -------
        dict[str, jnp.ndarray]
        """     
        if scaled:
            return {n: jnp.array(p) for n, p in (self.iter_params(**kwargs))}
        else:
            return {n: p.value for n, p in (self.iter_params(**kwargs))}    

    def param_names(self, *args, **kwargs) -> list[str]:
        """
        Return module parameter names as a list.

        See :meth:`.named_params`.
        """
        return list(self.named_params(*args, **kwargs).keys())

    def param(self, name: str, *args, **kwargs) -> Parameter:
        """
        Return a single module parameter by name.

        See :meth:`.named_params`.
        """
        return self.named_params(*args, **kwargs)[name]
    
    def params(self, *args, **kwargs) -> list[Parameter]:
        """
        Return module parameters as a list.

        See :meth:`.named_params`.
        """
        return list(self.named_params(*args, **kwargs).values())
    
    def param_value(self, name: str, *args, **kwargs) -> jnp.ndarray:
        """
        Return a single module parameter value by name as a single jax array.

        See :meth:`.named_param_values`.
        """
        return self.named_param_values(*args, **kwargs)[name]

    def param_values(self, *args, **kwargs) -> list[jnp.ndarray]:
        """
        Return module parameter values as a list of jax arrays.

        See :meth:`.named_param_values`.
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

        See :meth:`.named_flat_params`.

        Parameters
        ----------
        scaled : bool, default=False
            Whether or not to scale the returned values by the parameter scales.
        **kwargs
            Additional key-word arguments as in  :meth:`.named_params`.

        Returns
        -------
        dict[str, jnp.ndarray]
        """     
        if scaled:
            retval = {n: jnp.array(p) for n, p in (self.iter_params(flatten=True, **kwargs))}
        else:
            retval = {n: p.value for n, p in (self.iter_params(flatten=True, **kwargs))}
            
        if return_floats:
            retval = {k: float(np.array(v)) for k, v in retval.items()}
        return retval
         
    def flat_param_names(self, *args, **kwargs) -> list[str]:
        """
        Return flattened parameter names as a list.

        See :meth:`.named_flat_params`.
        """
        return list(self.named_flat_params(*args, **kwargs).keys())    
    
    def flat_params(self, *args, **kwargs) -> list[Parameter]:
        """
        Return flattened parameters as a list.

        See :meth:`.named_flat_params`.
        """
        return list(self.named_flat_params(*args, **kwargs).values())
    
    def flat_param_values(self, *args, **kwargs) -> jnp.ndarray:
        """
        Return flattened module parameter values as a jax arrays.

        See :meth:`.named_flat_param_values`.
        """
        return jnp.array(list(self.named_flat_param_values(*args, **kwargs).values())).reshape(-1)

    def flat_param_bounds(self, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Return flattened module parameter bounds as jax arrays.
        
        Note that a minimum and maximum percentile is used to get the bounds
        for any non-uniform distribution.

        Equivalent to getting the bounds from :meth:`.distribution`,
        which key-word arguments are forwarded to.
        """
        return self.distribution(**kwargs).bounds
    
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
        
        # 0. Identify valid parameters for the current mode (Free vs All)
        # We use named_flat_params to get the definitive list of "active" parameters.
        # This handles the logic for whether parameters are fixed or not.
        all_valid_params = self.named_flat_params(include_fixed=include_fixed)
        valid_param_names = set(all_valid_params.keys())

        # 1. Start with local, explicit groups defined in this module
        # We only keep groups that contain at least one parameter that is valid 
        # (i.e. not fixed, unless include_fixed=True).
        groups = []
        for group in self._param_groups:
            # We check if the group overlaps with the valid parameters.
            # If the intersection is empty, it means all parameters in the group 
            # are fixed (or don't exist), so we exclude the group.
            if not set(group.names).isdisjoint(valid_param_names):
                groups.append(deepcopy(group))

        # 2. Traverse submodules to get their groups recursively
        # We use tree_flatten_with_path to find all Module instances within self.
        # We treat Module instances as leaves so we don't traverse into their individual parameters here.
        path_and_nodes, _ = jax.tree_util.tree_flatten_with_path(
            self, 
            is_leaf=lambda x: isinstance(x, Module) and x is not self
        )

        for path, node in path_and_nodes:
            # Check if the node is a submodule (and not self, though is_leaf handles that mostly)
            if isinstance(node, Module) and node is not self:
                # Calculate the prefix for this submodule (e.g., "amplifier_")
                relative_name = self.path_to_param_name(path)
                prefix = f"{relative_name}{self._separator}" if relative_name else ""

                # Recursively get groups from the submodule
                sub_groups = node.param_groups(include_fixed=include_fixed)

                # "Lift" the submodule groups into the current namespace
                for sub_group in sub_groups:
                    new_names = [prefix + name for name in sub_group.names]
                    # Create a new group with the updated names
                    lifted_group = dataclasses.replace(sub_group, param_names=new_names)
                    groups.append(lifted_group)

        # 3. Deduplication and Conflict Resolution
        # We prioritize groups that appear earlier in the list (Parent groups > Submodule groups).
        # We filter the list to ensure every parameter appears in exactly one group.
        
        final_groups = []
        seen_params = set()

        for group in groups:
            # Find parameters in this group that haven't been claimed by a higher-priority group
            valid_names = [name for name in group.param_names if name not in seen_params]
            
            # If the group has valid parameters left, add it
            if valid_names:
                # If the group shrank (because parent claimed some params), update it
                if len(valid_names) != len(group.param_names):
                    group = dataclasses.replace(group, param_names=valid_names)
                
                final_groups.append(group)
                seen_params.update(valid_names)

        # 4. Handle Orphans
        # Any parameter in the entire module that wasn't caught in the steps above 
        # (mostly local parameters of `self` that weren't in `_param_groups`) gets a singleton group.
        all_params = self.named_flat_params(include_fixed=include_fixed)
        
        for name, param in all_params.items():
            if name not in seen_params:
                final_groups.append(ParameterGroup(names=[name], distribution=param.distribution))
                seen_params.add(name)

        return final_groups
    
    def distribution(self, param_groups: bool = True) -> JointDistribution:
        """Joint distribution over (flattened) parameters.
        
        Parameters
        ----------
        param_groups : bool, optional
            Whether or not to use the internal parameter groups
            to create the joint distribution. Defaults to ``True``.
        
        Returns
        -------
        JointParameterDistribution
        """
        if param_groups:
            groups = self.param_groups()
            group_names = [pg.names for pg in groups]
            group_dists = [pg.distribution for pg in groups]
        else:
            named_flat_params = self.named_flat_params()
            group_names = [[name] for name in named_flat_params.keys()]
            group_dists = [param.distribution for param in named_flat_params.values()]
            
        return JointDistribution(distributions=group_dists, distribution_names=group_names, param_names=self.flat_param_names())
    
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
        # Deal with the sample case i.e. an array-like object
        if not isinstance(params, dict) and len(param_kwargs) == 0:
            params = jnp.array(params)
            
            params_tree, static = partition(self, include_fixed=include_fixed)
            params_out, unravel_fn = flatten_util.ravel_pytree(params_tree)
            
            if jnp.isscalar(params_out) or params_out.shape[0] == 0:
                raise Exception("Error: no free module parameters")
            
            params_tree_recon = unravel_fn(params)
            return eqx.combine(params_tree_recon, static)

        params = params if params is not None else {}
        params.update(param_kwargs)
    
        # Generate an ordered, input flat params array for verification
        new_params = self.named_params(include_fixed=True)

        # ---- NEW: Pre-process to handle flattened keys with suffixes (e.g., 'z_real') ----
        # We must identify keys in `params` that are not in `new_params` (parents)
        # but ARE in the flattened view.
        
        parent_keys = set(new_params.keys())
        input_keys = set(params.keys())
        
        # Keys that are not top-level parameters
        potential_flat_keys = input_keys - parent_keys
        
        if potential_flat_keys:
            # Iterate over parents to find which flattened keys belong to them.
            # We only search parents that aren't ALREADY being fully replaced.
            parents_to_scan = [p for p in parent_keys if p not in params]
            
            for parent_name in parents_to_scan:
                parent_param = new_params[parent_name]
                
                # Optimization: only checking multi-dimensional parameters
                if parent_param.size > 0: 
                    # We must replicate the _iter_params name generation logic exactly
                    sub_params = parent_param.flattened(separator=self._separator)
                    
                    updates_found = False
                    new_sub_values = []
                    
                    # Reconstruct the value array from current values + updates
                    for i, sub_p in enumerate(sub_params):
                        suffix = sub_p.name if sub_p.name is not None else str(i)
                        flat_name = f"{parent_name}{self._separator}{suffix}"
                        
                        if flat_name in params:
                            val = params[flat_name]
                            # Handle single-element arrays or scalars
                            if hasattr(val, 'item') and val.size == 1:
                                val = val.item()
                            try:
                                val = float(val)
                            except:
                                raise Exception(f"Value for flat parameter '{flat_name}' must be convertible to float. Got: {val}")
                            new_sub_values.append(val)
                            
                            # Remove the flat key so it doesn't trigger 'unknown parameter' errors
                            del params[flat_name]
                            updates_found = True
                        else:
                            new_sub_values.append(sub_p.value)
                    
                    if updates_found:
                        # Re-assemble the parent parameter
                        new_val_flat = jnp.array(new_sub_values)
                        new_val_shaped = new_val_flat.reshape(parent_param.value.shape)
                        
                        # Update the params dict with the FULL parent object
                        # This ensures it hits the "Case 1" logic in the rest of the function
                        params[parent_name] = dataclasses.replace(parent_param, value=new_val_shaped)            
    
        # Validate the callers's input
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

        # Convert to an array of parameters instead of floats
        if all(is_convertible_to_float(v) for v in params.values()):            
            for name, value in params.items():
                new_params[name] = dataclasses.replace(new_params[name], value=jnp.array(value))
        else:
            new_params.update(params)
        new_flat_params = list(new_params.values())
    
        # Get the current flat parameter object
        params_tree, static = partition(self, include_fixed=True, param_objects=True)
        flat_params, treedef = jax.tree.flatten(params_tree, is_leaf=is_valid_param)
        
        # We allow the caller to pass None for name and then we update the name. Otherwise names should match
        for i, param in enumerate(flat_params):
            if new_flat_params[i].name is None:
                new_flat_params[i] = dataclasses.replace(new_flat_params[i], name=param.name)
        
        # Create the update tree and return
        new_params_tree = jax.tree.unflatten(treedef, new_flat_params)
        return eqx.combine(new_params_tree, static)

    def with_mapped_params(
        self: Self, 
        mapper: Callable[[Parameter], Parameter], 
        param_filter: str | Sequence[str] | Parameter | Sequence[Parameter] | Callable[[str], bool] | None = None, 
        *, 
        map_others: Callable[[Parameter], Parameter] | None = None,
        prefixes: bool = False,
        include_fixed: bool = False,
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
        current_params = self.named_params(include_fixed=include_fixed)
        current_param_names = set(current_params.keys())
        
        # NEW: If no filter is provided, target all parameters in the module
        if param_filter is None:
            resolved_filter = current_param_names
        else:
            if isinstance(param_filter, str):
                param_filter = [param_filter]
            elif isinstance(param_filter, Parameter):
                param_filter = [param_filter]

            if isinstance(param_filter, list) and param_filter and isinstance(param_filter[0], str) and prefixes:
                for prefix in param_filter:
                    if not any(name.startswith(prefix) for name in current_param_names):
                        raise ValueError(f"Specified prefix '{prefix}' does not match any parameters in the module")
                
                valid_prefixes = tuple(param_filter)
                param_filter = lambda p: p.startswith(valid_prefixes)            
                
            if isinstance(param_filter, Callable):
                param_filter = [p for p in current_params.keys() if param_filter(p)]
            elif isinstance(param_filter[0], Parameter):
                param_ids = {id(p) for p in param_filter}
                param_filter = [k for k, v in current_params.items() if id(v) in param_ids]

            resolved_filter = set(param_filter)            
            for param_name in resolved_filter:
                if param_name not in current_param_names:
                    raise ValueError(f"Specified parameter '{param_name}' not found in module")
        
        new_params = current_params.copy()
        for name, param in current_params.items():
            if name in resolved_filter:
                new_params[name] = mapper(param)
            elif map_others is not None:
                new_params[name] = map_others(param)
                
        return self.with_params(new_params)   
        
    def with_fixed_params(self: Self, param_filter: str | Sequence[str] | Parameter | Sequence[Parameter] | Callable[[str], bool], free_others: bool = False, **kwargs) -> Self:
        """Return a module with specified parameters fixed.

        This maps each parameter in the filter, calling :meth:`Parameter.as_fixed` on each.

        See :meth:`.with_mapped_params`.

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

        return self.with_mapped_params(lambda p: p.as_fixed(), param_filter=param_filter, map_others=map_others, **kwargs)
    
    def with_free_params(self: Self, param_filter: str | Sequence[str] | Parameter | Sequence[Parameter] | Callable[[str], bool], *, fix_others: bool = False, **kwargs) -> Self:
        """Free the specified parameters.

        This maps each parameter in the filter, calling :meth:`Parameter.as_free` on each.

        See :meth:`.with_mapped_params`.

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

        return self.with_mapped_params(lambda p: p.as_free(), param_filter=param_filter, map_others=map_others, **kwargs)
    
    def with_free_params_only(self: Self, param_filter: str | list[str] | Callable[[str], bool], **kwargs) -> Self:
        """Returns a module with only the specified parameters freed.
        
        This is an alias for calling :meth:`.with_free_params`
        with `fix_others=True`.

        See :meth:`.`with_free_params``.
        """
        kwargs.setdefault('fix_others', True)
        if kwargs['fix_others'] == False:
            raise Exception("Cannot pass fix_others == False for `with_free_params_only`.")
        return self.with_free_params(param_filter, **kwargs)

    def with_all_params_fixed(self: Self, **kwargs) -> Self:
        """Returns a module with all parameters fixed.
        
        This is an alias for calling :meth:`.with_free_params`
        with `fix_others=True` and no parameters passed.

        See :meth:`.`with_free_params``.
        """
        kwargs.setdefault('fix_others', True)
        if kwargs['fix_others'] == False:
            raise Exception("Cannot pass fix_others == False for `with_all_params_fixed`.")
        return self.with_free_params({}, **kwargs)

    def with_all_params_free(self: Self, **kwargs) -> Self:
        """Returns a module with all parameters free.
        
        This is an alias for calling :meth:`.with_free_params`
        with all parameters passed.

        See :meth:`.`with_free_params``.
        """
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
        
        # 1. Identify all parameter names claimed by the NEW groups
        new_claimed_params = set()
        for group in param_groups:
            # Assumes the field name is 'param_names' per your previous context
            new_claimed_params.update(group.names)

        # 2. Filter OLD groups (Atomic check)
        current_groups = self._param_groups if self._param_groups is not None else []
        kept_existing_groups = []
        
        for group in current_groups:
            # Check for intersection: Does this existing group contain ANY parameter 
            # that is now being redefined in the new groups?
            existing_group_params = set(group.names)
            
            if existing_group_params.isdisjoint(new_claimed_params):
                # No conflict: Keep this group entirely
                kept_existing_groups.append(group)
            else:
                # Conflict found: Discard this group entirely. 
                # Note: Parameters in this group that were NOT in 'new_claimed_params' 
                # are now effectively "released" and will be treated as singletons 
                # by the main param_groups() method.
                pass

        # 3. Combine
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
        # 1. Identify immediate submodules and their prefixes
        submodule_prefixes = {} 
        for f in dataclasses.fields(self):
            if isinstance(getattr(self, f.name), Module):
                prefix = f.name + self._separator
                submodule_prefixes[prefix] = f.name

        # 2. Sort current groups into "keep" (stay here) or "demote" (move to child)
        groups_to_keep = []
        submodule_groups = {name: [] for name in submodule_prefixes.values()}
        
        current_groups = self._param_groups if self._param_groups is not None else []

        for group in current_groups:
            demoted = False
            for prefix, field_name in submodule_prefixes.items():
                # Check if ALL parameters in the group belong to this submodule
                if all(name.startswith(prefix) for name in group.names):
                    # Strip prefix
                    new_names = [name[len(prefix):] for name in group.names]
                    new_group = dataclasses.replace(group, param_names=new_names)
                    submodule_groups[field_name].append(new_group)
                    demoted = True
                    break
            
            if not demoted:
                groups_to_keep.append(group)

        # 3. Apply updates to submodules AND recurse
        new_fields = {}
        
        # We iterate over all submodules (even if they didn't receive new groups from us)
        # because they might have their *own* local groups that need demoting further down.
        for prefix, field_name in submodule_prefixes.items():
            child_module: Module = getattr(self, field_name)
            
            # A. Push: Add the groups we demoted from the current level
            groups_to_push = submodule_groups[field_name]
            if groups_to_push:
                child_module = child_module.with_param_groups(groups_to_push)
            
            # B. Recurse: Ask the child to demote its groups (including the ones we just pushed)
            child_module = child_module.with_demoted_param_groups()
            
            new_fields[field_name] = child_module

        # 4. Return updated module
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
        new_fields = {}  # Removed '_param_groups': [] from the initialization
        
        for f in dataclasses.fields(self):
            # Skip the target field since we handle it at the end
            if f.name == '_param_groups':
                continue
                
            child = getattr(self, f.name)
            
            # 1. Recurse into direct submodules
            if isinstance(child, Module):
                new_fields[f.name] = child.with_no_param_groups()
                
            # 2. Recurse into sequences of submodules (e.g., in composites like Cascade)
            elif isinstance(child, (list, tuple)):
                # Only process the sequence if it actually contains at least one Module
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
        mapper: Callable[[Distribution], Distribution], 
        dist_filter: Callable[[Distribution], bool] | None = None, 
        *, 
        map_others: Callable[[Distribution], Distribution] | None = None,
        param_groups: bool = False
    ) -> Self:
        """Return a module with a function applied to its parameter distributions.

        This method allows for bulk-updates of distributions, such as widening variances 
        or changing distribution types.

        If ``param_groups`` is False, the mapping is applied to the distributions 
        of individual parameters (flattened).

        If ``param_groups`` is True, the mapping is applied to the distributions 
        of :class:`ParameterGroup` objects. This mode is recursive: it will traverse 
        the module tree and apply the mapping to all explicit parameter groups in all submodules.

        Parameters
        ----------
        mapper : Callable[[Distribution], Distribution]
            Function that takes a distribution and returns a new one.
        dist_filter : Callable[[Distribution], bool] | None, default=None
            A predicate function. If provided, the mapping is only applied to 
            distributions where ``dist_filter(dist)`` is True. If None, applies to all.
        map_others : Callable[[Distribution], Distribution] | None, default=None
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
            # 1. Map Local Groups (Current Level)
            current_groups = self._param_groups if self._param_groups is not None else []
            for group in current_groups:
                if dist_filter is None or dist_filter(group.distribution):
                    mapped_module = mapped_module.with_param_groups(group.with_distribution(mapper(group.distribution)))
                elif map_others is not None:
                    mapped_module = mapped_module.with_param_groups(group.with_distribution(map_others(group.distribution)))

            # 2. Recurse into Submodules
            new_submodules = {}
            for f in dataclasses.fields(mapped_module):
                child = getattr(mapped_module, f.name)
                # Check if the field is a direct submodule
                if isinstance(child, Module):
                    # Recursive call
                    updated_child = child.with_mapped_distributions(
                        mapper, 
                        dist_filter, 
                        map_others=map_others, 
                        param_groups=True
                    )
                    new_submodules[f.name] = updated_child
            
            # Apply submodule updates if any
            if new_submodules:
                mapped_module = mapped_module.with_fields(**new_submodules)

        else:
            # 3. Existing logic for individual params (Global via named_params)
            new_params = {}
            for name, param in self.named_params().items():
                if dist_filter is None or dist_filter(param.distribution):
                    new_params[name] = param.with_distribution(mapper(param.distribution))
                elif map_others is not None:
                    new_params[name] = param.with_distribution(map_others(param.distribution))
            
            # Apply all parameter updates at once
            if new_params:
                mapped_module = mapped_module.with_params(new_params)
                    
        return mapped_module
    
    def with_uniform_distributions(self, percentage: float, param_filter: str | Sequence[str] | Parameter | Sequence[Parameter] | Callable[[str], bool] = None, *, respect_bounds=False, remove_param_groups=True, zero_values='keep', **kwargs):
        """Return a module with uniform distributions set centered on current parameter values.

        The distributions are defined with bounds calculated as ``value * (1.0 +/- percentage)``.

        Parameters
        ----------
        percentage : float
            The fractional width of the uniform distribution (e.g. 0.1 = 10%).
        filter: str | Sequence[str] | Callable[[str], bool], default=None
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
        updates = {}
        current_params = self.named_params(param_filter, **kwargs)
        for name, param in current_params.items():
            value = param.value
            if value == 0.0:
                if zero_values == 'keep':
                    continue
                else:
                    raise Exception("Unknown option for 'zero_values'")
            elif value > 0.0:
                new_min = value * (1.0 - percentage)
                new_max = value * (1.0 + percentage)
            elif value < 0.0:
                new_min = value * (1.0 + percentage)
                new_max = value * (1.0 - percentage)
            
            if respect_bounds:
                new_min = max(new_min, param.min)
                new_max = min(new_max, param.max)

            distribution = dist.Uniform(new_min, new_max)
            updates[name] = param.with_distribution(distribution)
            
        new_module = self.with_params(updates)
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
                self.p = p   # underlying partial

            def __call__(self, *call_args, **call_kwargs):
                # 1. Clone ONLY the defaults baked into the wrapper.
                # This ensures each new module gets its own independent blueprint parameters.
                baked_args = deepcopy(self.p.args)
                baked_kwargs = deepcopy(self.p.keywords)
                
                # 2. Merge with explicitly passed arguments.
                # We DO NOT clone call_args or call_kwargs to preserve intentional parameter tying!
                final_args = baked_args + call_args
                final_kwargs = {**baked_kwargs, **call_kwargs}
                
                # 3. Instantiate the module
                return self.p.func(*final_args, **final_kwargs)

            # chaining
            def with_defaults(self, *new_args, **new_kwargs):
                # merge new defaults after existing ones
                merged_args = self.p.args + new_args
                merged_kwargs = {**self.p.keywords, **new_kwargs} if self.p.keywords else new_kwargs
                return DefaultsWrapper(partial(self.p.func, *merged_args, **merged_kwargs))
                
        return DefaultsWrapper(partial(cls, *args, **kwargs))
    
    def with_modules(self: Self, modules: Self | Sequence[Self]) -> Self:
        """Combines this module with free parameters in other modules.
        
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
            combined = combined.with_param_groups(other.param_groups)
        return combined
    
    def with_fields(self: Self, *args, **kwargs) -> Self:
        """
        Return a copy of this module with dataclass-style field replacements.

        Parameters are forwarded to :func:`dataclasses.replace`.
        """
        new_module = dataclasses.replace(self, *args, **kwargs)
        
        # Generically copy all init=False fields so replace() doesn't drop them
        for f in dataclasses.fields(self):
            if not f.init:
                val = getattr(self, f.name)
                # Use deepcopy to prevent shared state mutations, just in case
                object.__setattr__(new_module, f.name, deepcopy(val))
                
        return new_module
    
    def with_name(self: Self, name: str | None) -> Self:
        """
        Return a copy of this module with a different name.
        """
        return self.with_fields(name=name)
    
    def with_submodule_fields(self: Self, submodule: str | Sequence[str], *args, **kwargs) -> Self:
        """
        Return a copy of this module with dataclass-style field replacements on a nested sub-module.

        Parameters are forwarded to :func:`dataclasses.replace`.

        Parameters
        ----------
        submodule : str | Sequence[str]
            The name of the submodule (or sequence of names) to traverse.
            Can be a single string with a path e.g. 'submodule1.submodule2',
            or a list of submodules e.g. ['submodule1', 'submodule2'].
        """
        # Normalize input to a list of strings
        if isinstance(submodule, str) and submodule.find('.'):
            path = submodule.split('.')
        else:
            path = [submodule] if isinstance(submodule, str) else list(submodule)
        
        if not path:
            # If path is empty, apply fields to self (standard with_fields behavior)
            return self.with_fields(*args, **kwargs)

        target_key = path[0]

        if len(path) == 1:
            # Base case: We are at the parent of the final target submodule
            updated_child = getattr(self, target_key).with_fields(*args, **kwargs)
        else:
            # Recursive step: Tell the child to handle the rest of the path
            child = getattr(self, target_key)
            updated_child = child.with_submodule_fields(path[1:], *args, **kwargs)

        # Return a copy of 'self' with the new version of the child
        return self.with_fields(**{target_key: updated_child})    
    
    def with_free_submodules(self: Self, submodules: 'Module' | Sequence['Module'] | str | Sequence[str], include_fixed=False, fix_others=False) -> Self:
        """Free all parameters in the given submodules.

        Submodules parameters are obtained using :meth:`.param_names`.,
        and subsequently freed using :meth:`.`with_free_params``.
        
        Parameters
        ----------
        submodules : Module | Sequence[Module] | str | Sequence[str]
            Submodules whose parameters should be free.
        include_fixed : bool, default=False
            Also free parameters that are currently fixed in the submodules.
        fix_others : bool, default=False
            Fix all other submodules.

        Returns
        -------
        Self
        """        
        module_param_names = self.param_names(include_fixed=include_fixed, submodules=submodules)
        return self.with_free_params(module_param_names, fix_others=fix_others)

    def with_free_submodules_only(self: Self, *args, **kwargs) -> Self:
        """Returns a module with only the specified submodules freed.
        
        This is an alias for calling :meth:`.with_free_submodules`
        with `fix_others=True`.

        See :meth:`.`with_free_params``.
        """     
        kwargs.setdefault('fix_others', True)
        if kwargs['fix_others'] == False:
            raise Exception("Cannot pass fix_others == False for `with_free_submodules_only`.")
        return self.with_free_submodules(*args, **kwargs)
    
    def with_fixed_submodules(self: Self, submodules: 'Module' | Sequence['Module'] | str | Sequence[str]) -> Self:
        """Fix all parameters in the given submodules.

        Submodules parameters are obtained using :meth:`.param_names`.,
        and subsequently fixed using :meth:`.`with_fixed_params``.
        
        Parameters
        ----------
        submodules : Module | Sequence[Module] | str | Sequence[str]
            Submodules whose parameters should be fixed.

        Returns
        -------
        Self
        """        
        module_param_names = self.param_names(submodules=submodules)
        return self.with_fixed_params(module_param_names)