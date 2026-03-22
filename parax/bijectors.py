"""
Additional bijectors not present in distreqx.
"""
from typing import Sequence

import jax.numpy as jnp
import jax.scipy.special as jss
import jax.nn as jnn
from jaxtyping import PyTree

from parax.core.parameter import Parameter
from parax.core.transform import Transform, ParameterTransform

import jax.numpy as jnp
import jax.scipy as jsp
import distreqx.bijectors as bij
from typing import Tuple


class Exponential(
    bij.AbstractFowardInverseBijector,
    bij.AbstractInvLogDetJacBijector,
    bij.AbstractFwdLogDetJacBijector,
    strict=True,
):
    _is_constant_jacobian: bool = False
    _is_constant_log_det: bool = False

    def forward_and_log_det(self, x: PyTree):
        y = jnp.exp(x)
        return y, x

    def inverse_and_log_det(self, y: PyTree):
        x = jnp.log(y)
        return x, -jnp.log(y)

    def same_as(self, other) -> bool:
        return isinstance(other, Exponential)

# class NormalToUniform(distreqx.bijectors.AbstractBijector):
#     """
#     A bijection mapping a Normal(loc, scale) distribution to a Uniform(0, 1) distribution 
#     using the Probability Integral Transform.
#     """
#     loc: jnp.ndarray
#     scale: jnp.ndarray

#     def __init__(self, loc: float = 0.0, scale: float = 1.0):
#         self.loc = jnp.asarray(loc)
#         self.scale = jnp.asarray(scale)

#     def forward(self, x: jnp.ndarray) -> jnp.ndarray:
#         """Maps x ~ Normal to y ~ Uniform(0, 1)."""
#         z = (x - self.loc) / self.scale
#         return jsp.special.ndtr(z)

#     def inverse(self, y: jnp.ndarray) -> jnp.ndarray:
#         """Maps y ~ Uniform(0, 1) to x ~ Normal."""
#         # Clip to prevent evaluating ndtri at exact 0 or 1, which causes NaNs/Infs
#         eps = jnp.finfo(y.dtype).eps
#         y_clipped = jnp.clip(y, a_min=eps, a_max=1.0 - eps)
#         z = jsp.special.ndtri(y_clipped)
#         return z * self.scale + self.loc

#     def forward_and_log_det(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
#         """Returns the forward transformation and its log-determinant Jacobian."""
#         y = self.forward(x)
#         # The derivative of the CDF is exactly the PDF. 
#         # log(det(Jacobian)) = log(PDF(x))
#         log_det = jsp.stats.norm.logpdf(x, loc=self.loc, scale=self.scale)
#         return y, log_det

#     def inverse_and_log_det(self, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
#         """Returns the inverse transformation and its log-determinant Jacobian."""
#         x = self.inverse(y)
#         # By the inverse function theorem, the log det of the inverse is the 
#         # negative log det of the forward function evaluated at x.
#         log_det = -jsp.stats.norm.logpdf(x, loc=self.loc, scale=self.scale)
#         return x, log_det
    
# class UnitRescaler(distreqx.bijectors.AbstractBijector):
#     """
#     Standardizes a Uniform(low, high) distribution to a Uniform(0, 1) distribution.
#     Also known as Min-Max scaling.
#     """
#     low: jnp.ndarray
#     high: jnp.ndarray

#     def __init__(self, low: float = 0.0, high: float = 1.0):
#         self.low = jnp.asarray(low)
#         self.high = jnp.asarray(high)
        
#         # Ensure the range is positive to avoid division by zero
#         self._range = self.high - self.low

#     def forward(self, x: jnp.ndarray) -> jnp.ndarray:
#         """Maps x ~ [low, high] -> y ~ [0, 1]"""
#         return (x - self.low) / self._range

#     def inverse(self, y: jnp.ndarray) -> jnp.ndarray:
#         """Maps y ~ [0, 1] -> x ~ [low, high]"""
#         return y * self._range + self.low

#     def forward_and_log_det(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
#         """Returns the scaled value and the constant log-determinant."""
#         y = self.forward(x)
#         # log(1 / range) = -log(range)
#         log_det = -jnp.log(jnp.abs(self._range))
#         return y, jnp.broadcast_to(log_det, x.shape)

#     def inverse_and_log_det(self, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
#         """Returns the inverse value and the inverse log-determinant."""
#         x = self.inverse(y)
#         log_det = jnp.log(jnp.abs(self._range))
#         return x, jnp.broadcast_to(log_det, y.shape)

class IdentityTransform(Transform):
    def forward(self, x):
        return x
    
    def inverse(self, x):
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

    def forward(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    def inverse(self, x):
        for transform in reversed(self.transforms):
            param = transform.inverse(param)  # <-- Use .inv() to respect the pipeline
        return param
    
def HypercubeLogitTransform() -> ComposeTransform:
    return ComposeTransform([HypercubeTransform(), LogitTransform()])