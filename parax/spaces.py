from typing import Any, Literal
import dataclasses

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree
from parax.utils.distribution import hypercube_to_physical, physical_to_hypercube

from parax.parameter import Parameter, is_free_param
from parax.tree import partition
from parax.module import Module

# ------------------------------------------------------------------
# Bounds Extraction (Returns pure jnp.ndarrays)
# ------------------------------------------------------------------

def make_bounds(
    tree: PyTree, 
    lower_override: float | None = None, 
    upper_override: float | None = None
) -> tuple[PyTree, PyTree]:
    """
    Extract the lower and upper bounds of all free parameters in a PyTree.

    This function traverses a PyTree and returns two structurally identical PyTrees
    containing pure `jax.numpy.ndarray` bounds for each `Parameter`.

    Parameters
    ----------
    tree : PyTree
        The input PyTree containing parax Parameters.
    lower_override : float or None, optional
        A global scalar to override all lower bounds.
    upper_override : float or None, optional
        A global scalar to override all upper bounds.

    Returns
    -------
    tuple of (PyTree, PyTree)
        A tuple `(lower_bounds, upper_bounds)` containing PyTrees of the extracted
        boundary arrays. Static/fixed parameters are partitioned out.
    """
    def get_lower(x: Parameter):
        if not is_free_param(x): return x
        if lower_override is not None: return jnp.full_like(x.value, lower_override)
        if x.bounds is not None: return x.bounds[..., 0]
        if x.distribution is not None and hasattr(x.distribution, 'icdf'):
            return x.distribution.icdf(jnp.full_like(x.value, 0.001)) 
        return jnp.full_like(x.value, -jnp.inf)

    def get_upper(x: Parameter):
        if not is_free_param(x): return x
        if upper_override is not None: return jnp.full_like(x.value, upper_override)
        if x.bounds is not None: return x.bounds[..., 1]
        if x.distribution is not None and hasattr(x.distribution, 'icdf'):
            return x.distribution.icdf(jnp.full_like(x.value, 1.0 - 0.001)) 
        return jnp.full_like(x.value, jnp.inf)

    lower_tree = jax.tree.map(get_lower, tree, is_leaf=is_free_param)
    upper_tree = jax.tree.map(get_upper, tree, is_leaf=is_free_param)

    # Partition out anything that isn't a JAX array (removes fixed params, etc.)
    (lower_bounds, upper_bounds), _ = partition((lower_tree, upper_tree))
    return lower_bounds, upper_bounds

def make_bijectors(lower: PyTree, upper: PyTree):
    """
    Constructs a PyTree of bijectors based on the provided lower and upper bounds.
    """
    from distreqx.bijectors import Chain, ScalarAffine, Sigmoid, Softplus
    
    def _get_bijector(low, high):
        has_low = low is not None and not jnp.all(jnp.isneginf(low))
        has_high = high is not None and not jnp.all(jnp.isposinf(high))

        if has_low and has_high:
            return Chain([ScalarAffine(shift=low, scale=(high - low)), Sigmoid()])
        elif has_low:
            return Chain([ScalarAffine(shift=low), Softplus()])
        elif has_high:
            return Chain([ScalarAffine(shift=high, scale=-1.0), Softplus()])
        else:
            return ScalarAffine(shift=0.0, scale=1.0)

    return jax.tree.map(_get_bijector, lower, upper)

def apply_bijectors(bijector_tree, value_tree, forward=True):
    """
    Applies a bijector PyTree to a values PyTree. 
    Supports structural prefixes via custom leaf detection.
    """
    # Instruct JAX to stop traversing the bijector tree once it hits a bijector
    def is_bijector(node):
        return hasattr(node, 'forward') and hasattr(node, 'inverse')

    def _apply(bijector, values):
        # 'values' could be a single leaf array or a deeper sub-PyTree of arrays
        if forward:
            # Map the forward transform over the value (or sub-tree of values)
            # w -> u (Unconstrained to Constrained)
            return jax.tree.map(bijector.forward, values)
        else:
            # u -> w (Constrained to Unconstrained)
            # We clip values before the inverse transform to prevent exact edge
            # cases (like 0 or 1 in Sigmoid) from generating infinity.
            def _safe_inverse(v):
                # Optionally add your icdf_bounds clipping logic here if v is strictly bounded.
                # A generic safe clip for bounded arrays relies on checking the bijector chain.
                return bijector.inverse(v)
                
            return jax.tree.map(_safe_inverse, values)

    return jax.tree.map(
        _apply, 
        bijector_tree, 
        value_tree, 
        is_leaf=is_bijector
    )

# ------------------------------------------------------------------
# Spatial Transformations (Group-aware, separating state from semantics)
# ------------------------------------------------------------------

def get_hypercube_pytree(module: PyTree) -> PyTree:
    """
    Extracts hypercube PyTrees in [0, 1] for all free parameters.
    
    This evaluates the module's parameters and returns a PyTree of pure 
    `jax.numpy.ndarray` objects, leaving the original Module completely untouched.
    It safely handles multivariate distributions by temporarily grouping parameters.

    Parameters
    ----------
    module : PyTree
        The input PyTree or parax Module.

    Returns
    -------
    PyTree
        A PyTree of pure arrays containing the hypercube mapping for free parameters.
    """
    if not isinstance(module, Module):
        # Fallback for simple trees without grouping logic
        def _map_to_u(x: Parameter):
            if not is_free_param(x) or x.distribution is None: return x
            return physical_to_hypercube(x.distribution, x.value)
        u_tree = jax.tree.map(_map_to_u, module, is_leaf=is_free_param)
        u_arrays, _ = partition(u_tree, filter_spec=eqx.is_array)
        return u_arrays

    groups = module.param_groups(include_fixed=False)
    flat_vals = module.named_flat_param_values(include_fixed=False)
    
    u_dict = {}
    for group in groups:
        if group.distribution is None:
            for name in group.param_names:
                u_dict[name] = flat_vals[name]
            continue
            
        arrays = [flat_vals[name] for name in group.param_names]
        x = jnp.stack(arrays)
        if len(arrays) == 1:
            x = jnp.squeeze(x, axis=0)
            
        # Group-aware mapping physical -> [0, 1]
        u = physical_to_hypercube(group.distribution, x)
        
        if len(arrays) == 1:
            u_dict[group.param_names[0]] = u
        else:
            for i, name in enumerate(group.param_names):
                u_dict[name] = u[i]
                
    # Scaffold a new PyTree of pure arrays using the module's structure
    def _inject_u(p: Parameter):
        if not is_free_param(p): return p
        return u_dict[p.name]
        
    u_tree = jax.tree.map(_inject_u, module, is_leaf=is_free_param)
    
    # Strip out the static/fixed parts, returning ONLY the arrays for the optimizer
    u_arrays, _ = partition(u_tree, filter_spec=eqx.is_array)
    return u_arrays


def apply_hypercube_pytree(module: PyTree, u: PyTree) -> PyTree:
    """
    Takes a static module template and a structurally matching tree of pure U-arrays,
    performing group-aware mapping back to a physically valid Module.

    Parameters
    ----------
    module : PyTree
        The static template containing the distributions and bijectors.
    u_arrays : PyTree
        The pure array tree containing values in the [0, 1] unit hypercube.

    Returns
    -------
    PyTree
        A newly constructed Module with valid latent states corresponding to the U-arrays.
    """
    if not isinstance(module, Module):
        def _map_to_x(p: Parameter, u: jnp.ndarray):
            if not is_free_param(p): return p
            if p.distribution is None: return p.with_value(u)
            x = hypercube_to_physical(p.distribution, u)
            return p.with_value(x)
        return jax.tree.map(_map_to_x, module, u, is_leaf=is_free_param)

    # Temporarily map the u-arrays into the leaves so we can flat-extract them by name
    def _swap_leaves(param, u_val):
        return u_val if is_free_param(param) else param
        
    temp_u_module: Module = jax.tree.map(_swap_leaves, module, u, is_leaf=is_free_param)
    flat_u = temp_u_module.named_flat_param_values(include_fixed=False)
    
    groups = module.param_groups(include_fixed=False)
    flat_params = module.named_flat_params(include_fixed=False)
    new_params = {}
    
    for group in groups:
        if group.distribution is None:
            for name in group.param_names:
                new_params[name] = flat_params[name].with_value(flat_u[name])
            continue
            
        arrays = [flat_u[name] for name in group.param_names]
        u = jnp.stack(arrays)
        if len(arrays) == 1:
            u = jnp.squeeze(u, axis=0)
            
        # Map U [0, 1] back to physical X
        x = hypercube_to_physical(group.distribution, u)
        
        # Use .with_value() which safely handles bijector inverses natively
        if len(arrays) == 1:
            new_params[group.param_names[0]] = flat_params[group.param_names[0]].with_value(x)
        else:
            for i, name in enumerate(group.param_names):
                new_params[name] = flat_params[name].with_value(x[i])
                
    return module.with_params(new_params)


# ------------------------------------------------------------------
# Bound Enforcement via Auto-Bijectors
# ------------------------------------------------------------------

def enforce_bounds(module: Module) -> Module:
    """
    Applies auto-generated bounding bijectors to free parameters that 
    have bounds but lack a user-defined transform.
    
    Parameters with existing transforms are bypassed, trusting the user's mapping.
    """
    from distreqx.bijectors import Chain, ScalarAffine, Sigmoid, Softplus
    
    def _apply_bounding_bijectors(x: Parameter):
        if not is_free_param(x): 
            return x
        
        # Trust the user: If a transform is already present, rely on it
        if x.transform is not None:
            return x
        
        lower, upper = -jnp.inf, jnp.inf
        
        # Extract limits
        if x.bounds is not None:
            lower = x.bounds[..., 0]
            upper = x.bounds[..., 1]
        else:
            return x  # No bounds to enforce
        
        has_lower = jnp.any(lower > -jnp.inf)
        has_upper = jnp.any(upper < jnp.inf)
        
        if not has_lower and not has_upper:
            return x
        
        # Construct the correct mapping to the real line
        if has_lower and has_upper:
            scale = upper - lower
            bijector = Chain([ScalarAffine(shift=lower, scale=scale), Sigmoid()])
        elif has_lower and not has_upper:
            bijector = Chain([ScalarAffine(shift=lower), Softplus()])
        elif not has_lower and has_upper:
            bijector = Chain([ScalarAffine(shift=upper, scale=jnp.full_like(upper, -1.0)), Softplus()])
            
        # Re-initialize the parameter with the new transform
        return x.with_transform(bijector)

    return jax.tree.map(_apply_bounding_bijectors, module, is_leaf=is_free_param)