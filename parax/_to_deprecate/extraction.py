"""
Extraction of Parameter metadata into shadow pytrees.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree
from parax.utils.distribution import physical_to_hypercube

from parax.parameter import Parameter, is_free_param
from parax.filters import partition
from parax.deprecate.module import Module

def extract_metadata(tree: PyTree) -> PyTree:
    def _get_metadata(p: Parameter):
        if not is_free_param(p):
            return p
        return p.metadata

    return jax.tree.map(_get_metadata, tree, is_leaf=is_free_param)


def extract_metadata_fields(tree: PyTree, field: str) -> PyTree:
    def _get_field(p: Parameter):
        if not is_free_param(p):
            return p
        return p.metadata[field]

    return jax.tree.map(_get_field, tree, is_leaf=is_free_param)

def extract_names(tree: PyTree) -> PyTree:
    return extract_metadata_fields(tree, 'names')

def extract_distributions(tree: PyTree) -> PyTree:
    return extract_metadata_fields(tree, 'distribution')

def extract_transforms(tree: PyTree) -> PyTree:
    return extract_metadata_fields(tree, 'transforms')

def extract_bounds(tree: PyTree) -> PyTree:
    return extract_metadata_fields(tree, 'bounds')

def extract_scales(tree: PyTree) -> PyTree:
    return extract_metadata_fields(tree, 'scales')




def extract_bounds(
    tree: PyTree, 
    lower_override: float | None = None, 
    upper_override: float | None = None
) -> tuple[PyTree, PyTree]:
    """
    Extract the lower and upper bounds of all free parameters in a PyTree.

    This function traverses a PyTree and returns two structurally identical PyTrees
    containing pure `jax.numpy.ndarray` bounds in place of each `Parameter`.

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
        if lower_override is not None: return jnp.full_like(x.physical_value, lower_override)
        if x.bounds is not None: return x.bounds[..., 0]
        if x.distribution is not None and hasattr(x.distribution, 'icdf'):
            return x.distribution.icdf(jnp.full_like(x.physical_value, 0.001)) 
        return jnp.full_like(x.physical_value, -jnp.inf)

    def get_upper(x: Parameter):
        if not is_free_param(x): return x
        if upper_override is not None: return jnp.full_like(x.physical_value, upper_override)
        if x.bounds is not None: return x.bounds[..., 1]
        if x.distribution is not None and hasattr(x.distribution, 'icdf'):
            return x.distribution.icdf(jnp.full_like(x.physical_value, 1.0 - 0.001)) 
        return jnp.full_like(x.physical_value, jnp.inf)

    lower_tree = jax.tree.map(get_lower, tree, is_leaf=is_free_param)
    upper_tree = jax.tree.map(get_upper, tree, is_leaf=is_free_param)

    # Partition out anything that isn't a JAX array (removes fixed params, etc.)
    (lower_bounds, upper_bounds), _ = partition((lower_tree, upper_tree))
    return lower_bounds, upper_bounds


def extract_group_ids(module: PyTree) -> PyTree:
    """
    Extracts string-based param_groups into a PyTree of integer group IDs.
    Aware of parax.Module and parax.Parameter.
    """
    # 1. Fetch the legacy grouping information
    groups = module.param_groups(include_fixed=False)
    flat_params = module.named_flat_params(include_fixed=False)
    
    # 2. Map the physical memory ID of each Parameter to a group index
    param_id_to_group_idx = {}
    for group_idx, group in enumerate(groups):
        for name in group.param_names:
            param = flat_params[name]
            param_id_to_group_idx[id(param)] = group_idx
            
    # 3. Build the shadow tree
    def _get_group_id(p):
        if not is_free_param(p): return None
        # Return the assigned group index, or fallback to the parameter's own id if ungrouped
        return param_id_to_group_idx.get(id(p), id(p))
        
    return jax.tree.map(_get_group_id, module, is_leaf=is_free_param)


def extract_hypercube_values(module: PyTree) -> PyTree:
    """
    Extracts hypercube values in [0, 1] for all free parameters.
    
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
            return physical_to_hypercube(x.distribution, x.physical_value)
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