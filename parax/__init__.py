from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

from parax.replace import tree_replace as tree_replace

from parax.constant import (
    AbstractConstant as AbstractConstant,
    as_free as as_free,
)

from parax.metadata import (
    AbstractHasMetadata as AbstractHasMetadata,
)

from parax.unwrappables import (
    AbstractUnwrappable as AbstractUnwrappable,
    unwrap as unwrap,
    Parameterized as Parameterized,
    Computed as Computed,
    Frozen as Frozen,
    as_frozen as as_frozen,
)

from parax.variables import (
    AbstractVariable as AbstractVariable,
    AbstractConstrained as AbstractConstrained,
    Param as Param,
    Derived as Derived,
    Constrained as Constrained,
    Physical as Physical,
    Fixed as Fixed,
    Variable as Variable,
    param as param,
    derived as derived,
    constrained as constrained,
    physical as physical,
    as_param as as_param,
    as_fixed as as_fixed,
    map_variables as map_variables,
    map_variables_with_path as map_variables_with_path,
)

from parax.constraints import (
    AbstractConstraint as AbstractConstraint,
    RealLine as RealLine,
    GreaterThan as GreaterThan,
    LessThan as LessThan,
    Interval as Interval,
    Positive as Positive,
    Negative as Negative,
    TransformedConstraint as TransformedConstraint,
    TreeConstraint as TreeConstraint,
    CustomConstraint as CustomConstraint,
)

from parax.filters import (
    is_constant as is_constant,
    is_not_constant as is_not_constant,
    is_unwrappable as is_unwrappable,
    is_variable as is_variable,
    is_constrained as is_constrained,
    is_constraint as is_constraint,
    is_distribution as is_distribution,
    is_bijector as is_bijector,
)

# from parax import optimize as optimize
# from parax import experimental as experimental