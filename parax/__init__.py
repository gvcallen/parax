from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

from parax.replace import tree_replace as tree_replace

from parax.unwrappables import (
    AbstractUnwrappable as AbstractUnwrappable,
    Frozen as Frozen,
    Parameterized as Parameterized,
    Computed as Computed,
    unwrap as unwrap,
)

from parax.variables import (
    AbstractVariable as AbstractVariable,
    ParamLike as ParamLike,
    Param as Param,
    Fixed as Fixed,
    Derived as Derived,
    Constrained as Constrained,
    Physical as Physical,
    param as param,
    derived as derived,
    constrained as constrained,
    physical as physical,
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
    is_tagged as is_tagged,
    is_variable as is_variable,
    is_param_like as is_param_like,
    is_unwrappable as is_unwrappable,
    is_bounded as is_bounded,
    is_probabilistic as is_probabilistic,
    is_constraint as is_constraint,
    is_distribution as is_distribution,
    is_bijector as is_bijector,
)

from parax.converters import (
    as_free as as_free,
    as_fixed as as_fixed,
    as_frozen as as_frozen,
    as_param as as_param,
)


from parax import (
    constant as constant,
    tagged as tagged,
    bounded as bounded,
    probabilistic as probabilistic,
    unwrappables as unwrappables,
    variables as variables,
    constraints as constraints,
    filters as filters,
    converters as converters,
)