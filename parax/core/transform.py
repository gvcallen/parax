"""
Additional utility transforms.
"""
import equinox as eqx
from parax.core.parameter import Parameter

class Transform(eqx.Module):
    def __call__(self, x):
        raise NotImplementedError
    
    def inv(self, x):
        raise NotImplementedError
    
class ParameterTransform(Transform):
    def __call__(self, x):
        if isinstance(x, Parameter):
            return self.forward(x)
        return x
    
    def inv(self, x):
        if isinstance(x, Parameter):
            return self.inverse(x)
        return x
    
    def forward(self, param: Parameter) -> Parameter:
        raise NotImplementedError
    
    def inverse(self, param: Parameter) -> Parameter:
        raise NotImplementedError
