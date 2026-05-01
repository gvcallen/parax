from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Array
import equinox as eqx

from parax.filters import is_param
from parax.tree import unwrap


def extract_state(dynamic_model: PyTree, bounded: bool) -> PyTree:
    """
    Extracts the pure math state arrays for the solvers.
    Assumes the tree has already been partitioned to contain only free parameters.
    
    Args:
        dynamic_model: The dynamic half of the parax model.
        bounded: If True, extracts the O(1) unscaled physical space for bounded solvers. 
                 If False, extracts the infinite latent space for unconstrained solvers.
                 
    Returns:
        A PyTree of pure JAX arrays matching the structure of `dynamic_model`.
    """
    space = "unscaled" if bounded else "raw"
    return unwrap(dynamic_model, space=space)


def inject_state(
    dynamic_model: PyTree, 
    state_tree: PyTree, 
    bounded: bool,
) -> PyTree:
    """
    Injects a solver's active state arrays back into the dynamic skeleton of a parax model.

    Args:
        dynamic_model: The original dynamic parax PyTree containing `Parameter` objects.
        state_tree: The PyTree of updated arrays provided by the solver.
        bounded: Whether the state arrays are bounded (unscaled) or infinite (raw).
        
    Returns:
        A new dynamic parax model PyTree. (Must be eqx.combine'd with the static model).
    """
    def _inject(param: Any, state_val: Any) -> Any:
        if is_param(param):
            if bounded:
                # Push the bounded physical value backwards to sync the latent space
                if param.constraint is not None and param.constraint.bijector is not None:
                    new_raw = param.constraint.bijector.inverse(state_val)
                else:
                    new_raw = state_val
            else:
                # Unbounded state maps 1:1 with the latent space
                new_raw = state_val
                
            # Perform a functional Equinox update
            return eqx.tree_at(lambda p: p.raw_value, param, new_raw)
        return param

    # Map the solver's arrays over the dynamic skeleton
    return jax.tree_util.tree_map(_inject, dynamic_model, state_tree, is_leaf=is_param)


def build_objective_fn(
    fn: Callable, 
    dynamic_model: PyTree, 
    static_model: PyTree, 
    bounded: bool,
) -> Callable:
    """
    Wraps a physics loss function for use with SciPy or Optimistix.
    
    Args:
        fn: User function with signature `fn(model, *args, **kwargs)`.
        dynamic_model: The dynamic half of the parax model.
        static_model: The static half of the parax model.
        bounded: If True, handles inverse bijector mapping for bounded solvers.

    Returns:
        Callable `objective(state_tree, *args, **kwargs)` returning a scalar loss.
    """
    def objective(state_tree: PyTree, *args, **kwargs) -> Array:
        # 1. Inject math arrays back into physical parameters
        updated_dynamic = inject_state(dynamic_model, state_tree, bounded=bounded)
        
        # 2. Recombine with static graph (The Equinox Way)
        full_model = eqx.combine(updated_dynamic, static_model)
        
        # 3. Evaluate physics
        return fn(full_model, *args, **kwargs)
        
    return objective


def build_logprior_fn(dynamic_model: PyTree) -> Callable:
    """
    Builds a function that calculates the total log-prior probability.
    Automatically applies Jacobian Change-of-Variables for constraints.
    
    Args:
        dynamic_model: The dynamic half of the parax model.

    Returns:
        Callable `log_prior_fn(state_tree)` evaluating purely on latent states.
    """
    def logprior_fn(state_tree: PyTree) -> Array:
        def get_leaf_log_p(param: Any, state_val: Any) -> Array:
            if is_param(param):
                lp = jnp.array(0.0)
                bijector = param.constraint.bijector if param.constraint is not None else None
                
                # 1. Add Bijector Jacobian (Change of Variables)
                if bijector is not None:
                    lp += jnp.sum(bijector.forward_log_det_jacobian(state_val))
                    unscaled_val = bijector.forward(state_val)
                else:
                    unscaled_val = state_val
                    
                # 2. Add Distribution Log Prob
                if param.distribution is not None:
                    lp += jnp.sum(param.distribution.log_prob(unscaled_val))
                    
                return lp
            return jnp.array(0.0)
            
        log_p_tree = jax.tree_util.tree_map(
            get_leaf_log_p, dynamic_model, state_tree, is_leaf=is_param
        )
        
        leaves, _ = jax.tree_util.tree_flatten(log_p_tree)
        return jnp.sum(jnp.array(leaves)) if leaves else jnp.array(0.0)
        
    return logprior_fn


def build_logposterior_fn(
    loglikelihood_fn: Callable, 
    dynamic_model: PyTree,
    static_model: PyTree
) -> Callable:
    """
    Wraps a physics log-likelihood function for use with BlackJAX or Numpyro.

    Args:
        loglikelihood_fn: User function `loglikelihood_fn(model, *args, **kwargs)`.
        dynamic_model: The dynamic half of the parax model.
        static_model: The static half of the parax model.

    Returns:
        Callable `log_posterior(state_tree, *args, **kwargs)` targeting the latent space.
    """
    log_prior_fn = build_logprior_fn(dynamic_model)
    
    def logposterior_fn(state_tree: PyTree, *args, **kwargs) -> Array:
        # Prior is evaluated on the math arrays
        lp = log_prior_fn(state_tree)
        
        # Inject to physical space (bounded=False because MCMC uses latent space)
        updated_dynamic = inject_state(dynamic_model, state_tree, bounded=False)
        full_model = eqx.combine(updated_dynamic, static_model)
        
        # Evaluate physics
        ll = loglikelihood_fn(full_model, *args, **kwargs)
        
        return lp + ll
        
    return logposterior_fn


def build_prior_transform_fn(dynamic_model: PyTree) -> Callable:
    """Builds a function mapping a unit hypercube to the unconstrained latent space."""
    def prior_transform_fn(hypercube_state: PyTree) -> PyTree:
        def _transform(param: Any, u: Any) -> Any:
            if is_param(param):
                if param.distribution is None:
                    raise ValueError(
                        f"Hypercube samplers require a proper prior distribution. "
                        f"Missing on parameter: {param.name}"
                    )
                # 1. Hypercube -> Unscaled Physical (via Inverse CDF / Quantile)
                unscaled_val = param.distribution.icdf(u)
                
                # 2. Unscaled Physical -> Latent Raw Space (via Inverse Bijector)
                if param.constraint is not None and param.constraint.bijector is not None:
                    return param.constraint.bijector.inverse(unscaled_val)
                return unscaled_val
            return param
            
        return jax.tree_util.tree_map(
            _transform, dynamic_model, hypercube_state, is_leaf=is_param
        )
    return prior_transform_fn