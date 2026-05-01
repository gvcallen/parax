from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

from parax.parameter import (
    Parameter as Parameter,
    asparam as asparam,
    field as field,
)

from parax.filters import (
    is_param as is_param,
    is_free_param as is_free_param,
    is_fixed_param as is_fixed_param,
    is_constraint as is_constraint,
    is_distribution as is_distribution,
    where_free_raw_value as where_free_raw_value,
    when_free_raw_value as when_free_raw_value,
)

from parax import tree as tree