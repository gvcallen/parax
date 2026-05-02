from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

from parax.replace import tree_replace as tree_replace

from parax.unwrappable import (
    AbstractUnwrappable as AbstractUnwrappable,
    unwrap as unwrap,
    DerivedTree as DerivedTree,
    FixedTree as FixedTree,
)

from parax.variables import (
    AbstractVariable as AbstractVariable,
    AbstractMetadataVariable as AbstractMetadataVariable,
    AbstractFreeVariable as AbstractFreeVariable,
    Param as Param,
    Constrained as Constrained,
    Derived as Derived,
    Physical as Physical,
    Fixed as Fixed,
    Variable as Variable,
    map_variables as map_variables,
    map_variables_with_path as map_variables_with_path,
    param as param,
    constrained as constrained,
    derived as derived,
    physical as physical,
    asparam as asparam,
    asfree as asfree,
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
    is_variable as is_variable,
    is_free_variable as is_free_variable,
    is_constraint as is_constraint,
    is_distribution as is_distribution,
    is_bijector as is_bijector,
    where_free_array as where_free_array,
    when_free_array as when_free_array,
)

# from parax import optimize as optimize
# from parax import experimental as experimental