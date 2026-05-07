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
    Static as Static,
    unwrap as unwrap,
)

from parax.variables import (
    AbstractVariable as AbstractVariable,
    Param as Param,
    Tagged as Tagged,
    Fixed as Fixed,
    Derived as Derived,
    Constrained as Constrained,
    Random as Random,
    tagged as tagged,
    derived as derived,
    constrained as constrained,
    random as random,
)

from parax.filters import (
    is_constant as is_constant,
    is_annotated as is_annotated,
    is_variable as is_variable,
    is_param as is_param,
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
    annotated as annotated,
    constant as constant,
    constraints as constraints,
    bounded as bounded,
    experimental as experimental,
    probabilistic as probabilistic,
    unwrappables as unwrappables,
    variables as variables,
    constraints as constraints,
    filters as filters,
    converters as converters,
)