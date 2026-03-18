"""
Additional transforms not defined in NumPyro.
"""

import jax.numpy as jnp
import jax.scipy.special as jss
import jax.nn as jnn

from parax.core.parameter import Parameter
from parax.core.transform import ParameterTransform

class HypercubeTransform(ParameterTransform):
    def __call__(self, param: Parameter):
        if param.distribution is None:
            return param
        return param.with_value(param.distribution.cdf(param.value))

    def inv(self, param: Parameter):
        if param.distribution is None:
            return param
        return param.with_value(param.distribution.icdf(param.value))

class UnboundedTransform(ParameterTransform):
    def __call__(self, param: Parameter):
        if param.distribution is None:
            return param
        eps = jnp.finfo(param.value.dtype).eps
        u = param.distribution.cdf(param.value)
        return param.with_value(jss.logit(jnp.clip(u, eps, 1.0 - eps)))

    def inv(self, param: Parameter):
        if param.distribution is None:
            return param
        eps = jnp.finfo(param.value.dtype).eps    
        u = jnp.clip(jnn.sigmoid(param.value), eps, 1.0 - eps)
        return param.with_value(param.distribution.icdf(u))