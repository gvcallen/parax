"""
Additional utility transforms.
"""
import equinox as eqx
from parax.core.parameter import Parameter

class Transform(eqx.Module):
    def forward(self, x):
        raise NotImplementedError
    
    def inverse(self, x):
        raise NotImplementedError
    
class ParameterTransform(Transform):
    def forward(self, x):
        if isinstance(x, Parameter):
            return self.forward(x)
        return x
    
    def inverse(self, x):
        if isinstance(x, Parameter):
            return self.inverse(x)
        return x
    
    def forward(self, param: Parameter) -> Parameter:
        raise NotImplementedError
    
    def inverse(self, param: Parameter) -> Parameter:
        raise NotImplementedError
