from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

from parax.field import field
from parax.io import load, save
from parax.module import Module
from parax.parameter import Parameter
from parax.parameter_group import ParameterGroup
from parax.partition import partition

from parax.parameters import *

import parax.parameters as parameters
import parax.distributions as distributions

__all__ = [
    "field",
    "load",
    "save",
    "Module",
    "Parameter",
    "ParameterGroup",
    "partition",
    "distributions",
]
__all__.extend(parameters.__all__)