from typing import Any, Callable, Optional
import abc

import jax
from jaxtyping import PyTree, Array, Scalar
import equinox as eqx

from parax.infer.results import InferenceResults
from parax.filters import is_free_array, where_free_param
from parax._to_deprecate.bridge import (
    extract_state, 
    inject_state, 
    build_logprior_fn, 
    build_objective_fn, 
    build_prior_transform_fn, 
    build_logposterior_fn
)


class SamplePayload(eqx.Module):
    """The core mathematical payload of a sampling run."""
    #: Stacked latent/unscaled arrays (the samples)
    samples: PyTree[Array]
    
    #: Stacked log-likelihoods or log-posteriors
    fn_values: Array
    
    #: Statistical weights (mostly for Nested/Importance sampling)
    weights: Array | None = None
    
    #: Stacked auxiliary data
    auxes: PyTree[Array] | None = None


class AbstractSampler(eqx.Module):
    """Base marker class for all parax samplers."""
    @abc.abstractmethod
    def sample(
        self,
        fn: Callable[[PyTree, Any], Any],
        *args,
        **kwargs,
    ) -> tuple[SamplePayload, PyTree]:
        raise NotImplementedError


class AbstractJointSampler(AbstractSampler):
    """Interface for samplers exploring the joint log-posterior (e.g. MCMC-based NUTS or HMC)."""
    @abc.abstractmethod
    def sample(
        self,
        logposterior_fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        key: Array,
        args: PyTree[Any] = None,
        init_samples: Optional[PyTree] = None,
        max_steps: int | None = None,
        has_aux: bool = False,
        **kwargs,
    ) -> tuple[SamplePayload, PyTree]:
        raise NotImplementedError


class AbstractSplitSampler(AbstractSampler):
    """Interface for samplers needing separate likelihood and prior densities (e.g., modern Nested Sampling)."""
    @abc.abstractmethod
    def sample(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Any],
        logprior_fn: Callable[[PyTree], Scalar],
        y0: PyTree,
        key: Array,
        args: PyTree[Any] = None,
        init_samples: Optional[PyTree] = None,
        max_steps: int | None = None,
        has_aux: bool = False,
        **kwargs,
    ) -> tuple[SamplePayload, PyTree]:
        raise NotImplementedError


class AbstractHypercubeSampler(AbstractSampler):
    """Interface for samplers operating in a unit hypercube (e.g., classical Nested Sampling)."""
    @abc.abstractmethod
    def sample(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Any],
        prior_transform_fn: Callable[[PyTree], PyTree],
        y0: PyTree,
        key: Array,
        args: PyTree[Any] = None,
        init_samples: Optional[PyTree] = None,
        max_steps: int | None = None,
        has_aux: bool = False,
        **kwargs,
    ) -> tuple[SamplePayload, PyTree]:
        raise NotImplementedError


def sample(
    fn: Callable[[PyTree, Any], Any],
    solver: AbstractSampler,
    y0: PyTree,
    key: Array,
    args: PyTree[Any] = None,
    init_samples: Optional[PyTree] = None,
    max_steps: int | None = None,
    has_aux: bool = False,
    filter_spec: PyTree | None = None,
    **kwargs
) -> InferenceResults:
    """
    High-level Bayesian sampling API for parax models.
    
    Automatically handles static/dynamic tree partitioning, topological 
    bijector mappings, and prior injection.
    """
    # 1. Structural Partitioning
    filter_spec = filter_spec if filter_spec is not None else where_free_param(y0)
    dynamic_model, static_model = eqx.partition(y0, filter_spec, is_leaf=is_free_array)

    # 2. Build the Math Wrappers
    wrapped_ll = build_objective_fn(fn, dynamic_model, static_model, bounded=False)
    wrapped_prior = build_logprior_fn(dynamic_model)
    wrapped_posterior = build_logposterior_fn(fn, dynamic_model, static_model)
    
    math_y0 = extract_state(dynamic_model, bounded=False)
    
    # 3. Route to the correct interface contract
    if isinstance(solver, AbstractJointSampler):
        kwargs['logposterior_fn'] = wrapped_posterior
    elif isinstance(solver, AbstractSplitSampler):
        kwargs['loglikelihood_fn'] = wrapped_ll
        kwargs['logprior_fn'] = wrapped_prior
    elif isinstance(solver, AbstractHypercubeSampler):
        kwargs['loglikelihood_fn'] = wrapped_ll
        kwargs['prior_transform_fn'] = build_prior_transform_fn(dynamic_model)
    else:
        raise TypeError(f"Unknown solver type in `parax.sample`: {type(solver)}")
    
    payload, metrics = solver.sample(
        y0=math_y0,
        key=key,
        args=args,
        init_samples=init_samples,
        max_steps=max_steps,
        has_aux=has_aux,
        **kwargs
    )    

    # 4. Reconstruct the Batched Physical Model
    # We vmap over the leading dimension (0) of the latent samples payload
    batched_dynamic = jax.vmap(inject_state, in_axes=(None, 0, None))(
        dynamic_model, payload.samples, False
    )
    batched_model = eqx.combine(batched_dynamic, static_model)

    # 5. Return High-Level Results
    return InferenceResults(
        model=batched_model, 
        fn_values=payload.fn_values,
        weights=payload.weights,
        auxes=payload.auxes,
        metrics=metrics, 
    )