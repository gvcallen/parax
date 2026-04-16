import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Key, PyTree

from parax.module import Module
from parax.parameter_group import ParameterGroup

from distreqx.distributions import (
    AbstractSampleLogProbDistribution,
    AbstractProbDistribution,
    AbstractCDFDistribution,
    AbstractSTDDistribution,
    AbstractSurvivalDistribution
)

class ModuleDistribution(
    AbstractSampleLogProbDistribution,
    AbstractProbDistribution,
    AbstractCDFDistribution,
    AbstractSTDDistribution,
    AbstractSurvivalDistribution,
    strict=True
):
    """
    (experimental) A joint distribution over a Parax Module's free parameters.
    
    Returned from [`parax.Module.distribution`][].
    
    This acts as a Facade. It treats the complex parameter group DAG 
    as a flat dictionary PyTree (dict[str, jax.Array]), fully satisfying 
    both the distreqx contract and the JAX ecosystem.
    """
    module_template: Module
    groups: list[ParameterGroup]

    def __init__(self, module: Module):
        self.module_template = module
        self.groups = module.param_groups(include_fixed=False)

    # ---- Internal Routing Helpers -------------------------------------------

    def _to_val_dict(self, value: PyTree | Module) -> dict[str, Array]:
        """Ensures the incoming value is a flat dictionary of JAX arrays."""
        if isinstance(value, type(self.module_template)):
            return value.named_flat_param_values()
        elif isinstance(value, dict):
            return value
        else:
            raise ValueError("Expected a Parax Module or a flat dictionary of parameter arrays.")

    def _prep_group_args(self, group: ParameterGroup, val_dict: dict[str, Array]) -> Array:
        """Extracts and formats arguments for a specific parameter group."""
        group_vals = jnp.stack([val_dict[name] for name in group.param_names])
        # If the group only has 1 parameter, squeeze it so univariate dists receive shape ()
        if len(group.param_names) == 1:
            group_vals = jnp.squeeze(group_vals, axis=0)
        return group_vals

    def _aggregate_scalar(self, method_name: str, value: PyTree | Module = None) -> Array:
        """Routes a method to all groups and safely sums the results into a scalar."""
        total = jnp.array(0.0)
        val_dict = self._to_val_dict(value) if value is not None else None
            
        for group in self.groups:
            method = getattr(group.distribution, method_name)
            
            if val_dict is not None:
                args = self._prep_group_args(group, val_dict)
                res = method(args)
            else:
                res = method()
                
            # Defensively sum over any batch/event dimensions returned by the group's distribution
            total += jnp.sum(res)
            
        return total

    def _build_tree(self, method_name: str, key: Key = None, value: PyTree | Module = None) -> PyTree:
        """Routes a method to all groups and returns a flat dictionary PyTree."""
        new_vals = {}
        val_dict = self._to_val_dict(value) if value is not None else None

        # Safely split keys if sampling
        keys = jax.random.split(key, len(self.groups)) if key is not None else [None] * len(self.groups)
        
        for k, group in zip(keys, self.groups):
            method = getattr(group.distribution, method_name)
            kwargs = {'key': k} if key is not None else {}
            
            if val_dict is not None:
                args = self._prep_group_args(group, val_dict)
                res = method(args, **kwargs)
            else:
                res = method(**kwargs)
            
            # Unpack the results (whether scalar or vector) back into named parameters
            res_array = jnp.atleast_1d(res)
            for i, name in enumerate(group.param_names):
                new_vals[name] = res_array[i]
                
        return new_vals

    # ---- distreqx.AbstractDistribution Implementation -----------------------

    @property
    def event_shape(self) -> PyTree:
        # The PyTree shape is a dictionary matching the flat parameter names
        val_dict = self.module_template.named_flat_param_values()
        return {k: jnp.shape(v) for k, v in val_dict.items()}

    def log_prob(self, value: PyTree | Module) -> Array:
        return self._aggregate_scalar('log_prob', value)

    def sample(self, key: Key[Array, ""]) -> PyTree:
        return self._build_tree('sample', key=key)

    def entropy(self) -> Array:
        return self._aggregate_scalar('entropy')

    def log_cdf(self, value: PyTree | Module) -> Array:
        return self._aggregate_scalar('log_cdf', value)

    def icdf(self, value: PyTree | Module) -> PyTree:
        return self._build_tree('icdf', value=value)

    def mean(self) -> PyTree:
        return self._build_tree('mean')

    def median(self) -> PyTree:
        return self._build_tree('median')

    def mode(self) -> PyTree:
        return self._build_tree('mode')

    def variance(self) -> PyTree:
        return self._build_tree('variance')

    def kl_divergence(self, other_dist: 'ModuleDistribution', **kwargs) -> Array:
        if not isinstance(other_dist, ModuleDistribution):
            raise TypeError("KL divergence is only supported between two ModuleDistributions.")
            
        total_kl = jnp.array(0.0)
        for self_group, other_group in zip(self.groups, other_dist.groups):
            kl_res = self_group.distribution.kl_divergence(other_group.distribution, **kwargs)
            total_kl += jnp.sum(kl_res)
            
        return total_kl