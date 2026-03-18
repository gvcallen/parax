"""
Additional transforms not defined in NumPyro.
"""

import equinox as eqx

from parax.parameter import Parameter

class ParameterTransform(eqx.Module):
    """Base class for all Parax parameter transformations."""
    def __call__(self, param: Parameter) -> Parameter:
        raise NotImplementedError

    def inv(self, param: Parameter) -> Parameter:
        raise NotImplementedError