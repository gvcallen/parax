from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

from parax.parameter import (
    Parameter as Parameter,
    asparam as asparam,
    param as param,
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

from parax.paramtree import (
    freeze as freeze,
    unfreeze as unfreeze,
    merge_update as merge_update,
    partition as partition,
    combine as combine,
)

from parax import optimize as optimize

from parax.replace import tree_replace as tree_replace
from parax import paramtree as paramtree
from parax import experimental as experimental