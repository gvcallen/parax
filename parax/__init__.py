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
    MetadataContainer as MetadataContainer,
)

from parax.bounded import (
    AbstractBounded as AbstractBounded,
    tree_bounded_base as tree_bounded_base,
    tree_bounded_lower as tree_bounded_lower,
    tree_bounded_upper as tree_bounded_upper,
    tree_bounded_bounds as tree_bounded_bounds,
    tree_bounded_convert as tree_bounded_convert,
    tree_bounded_replace as tree_bounded_replace,
)

from parax.unwrappables import (
    AbstractUnwrappable as AbstractUnwrappable,
    unwrap as unwrap,
    Frozen as Frozen,
    as_frozen as as_frozen,
    Parameterized as Parameterized,
    Computed as Computed,
)

from parax.variables import (
    AbstractVariable as AbstractVariable,
    Param as Param,
    param as param,
    as_param as as_param,
    Fixed as Fixed,
    as_fixed as as_fixed,
    Derived as Derived,
    derived as derived,
    Constrained as Constrained,
    constrained as constrained,
    Physical as Physical,
    physical as physical,
    ParamLike as ParamLike,
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
    is_variable as is_variable,
    is_param_like as is_param_like,
    is_unwrappable as is_unwrappable,
    is_bounded as is_bounded,
    is_constraint as is_constraint,
    is_distribution as is_distribution,
    is_bijector as is_bijector,
)

from parax import optimize as optimize
