"""
Parameter transform sub-classes with pre-defined functionality.
"""
from typing import Sequence

import jax.numpy as jnp
import jax.scipy.special as jss
import jax.nn as jnn

from parax.core.parameter import Parameter
from parax.core.transform import Transform, ParameterTransform

class IdentityTransform(Transform):
    def __call__(self, x):
        return x
    
    def inv(self, x):
        return x
    
class LowerPercentile(ParameterTransform):
    percentile: float = 0.999
    
    def forward(self, param: Parameter):
        if param.distribution is None:
            return param.with_value(jnp.full_like(param.latent_value, -jnp.inf))
        return param.with_value(param.distribution.icdf(1.0 - self.percentile))

class UpperPercentile(ParameterTransform):
    percentile: float = 0.999

    def forward(self, param: Parameter):
        if param.distribution is None:
            return param.with_value(jnp.full_like(param.latent_value, jnp.inf))
        return param.with_value(param.distribution.icdf(self.percentile))
    
class HypercubeTransform(ParameterTransform):
    def forward(self, param: Parameter):
        if param.distribution is None:
            return param
        
        # Clip on the forward pass just to be perfectly safe
        eps = jnp.finfo(param.latent_value.dtype).eps
        cdf_val = param.distribution.cdf(param.latent_value)
        safe_cdf = jnp.clip(cdf_val, eps, 1.0 - eps)
        
        return param.with_value(safe_cdf)

    def inverse(self, param: Parameter):
        if param.distribution is None:
            return param
            
        # The crucial fix: Prevent SciPy from passing exactly 0.0 or 1.0 to the ICDF
        eps = jnp.finfo(param.latent_value.dtype).eps
        safe_val = jnp.clip(param.latent_value, eps, 1.0 - eps)
        
        return param.with_value(param.distribution.icdf(safe_val))
    
class LogitTransform(ParameterTransform):
    """Maps values from [0, 1] to the unbounded real line (-inf, inf)."""
    def forward(self, param: Parameter):
        eps = jnp.finfo(param.latent_value.dtype).eps
        # Assume input value is already bounded in [0, 1]
        u = param.latent_value
        return param.with_value(jss.logit(jnp.clip(u, eps, 1.0 - eps)))

    def inverse(self, param: Parameter):
        eps = jnp.finfo(param.latent_value.dtype).eps
        # Map unbounded real line back to [0, 1]
        u = jnp.clip(jnn.sigmoid(param.latent_value), eps, 1.0 - eps)
        return param.with_value(u)
    
class ComposeTransform(ParameterTransform):
    """Composes multiple ParameterTransforms together."""
    
    transforms: Sequence[ParameterTransform] 

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    def inv(self, x):
        for transform in reversed(self.transforms):
            param = transform.inv(param)  # <-- Use .inv() to respect the pipeline
        return param
    
def HypercubeLogitTransform() -> ComposeTransform:
    return ComposeTransform([HypercubeTransform(), LogitTransform()])