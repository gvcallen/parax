from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

from parax.replace import tree_replace as tree_replace

from parax.constants import (
    is_constant as is_constant,
    as_free as as_free,
)
from parax.annotation import is_annotated as is_annotated
from parax.bounds import is_bounded as is_bounded
from parax.probability import is_probabilistic as is_probabilistic
from parax.constraints import is_constraint as is_constraint, is_constrained as is_constrained, is_constrainable as is_constrainable

from parax.wrappers import (
    AbstractUnwrappable as AbstractUnwrappable,
    AbstractWrappable as AbstractWrappable,
    wrap as wrap,
    unwrap as unwrap,
    unwrap_self as unwrap_self,
    is_unwrappable as is_unwrappable,
    is_wrappable as is_wrappable,
    as_unwrapped as as_unwrapped,
    Freeze as Freeze,
    Parameterize as Parameterize,
    Apply as Apply,
    Static as Static,
    Tie as Tie,
    Bound as Bound,
    Constrain as Constrain,
    Probabilize as Probabilize,
    as_opaque as as_opaque,
)

from parax.variables import (
    AbstractVariable as AbstractVariable,
    Param as Param,
    Real as Real,
    Tagged as Tagged,
    Fixed as Fixed,
    Derived as Derived,
    Bounded as Bounded,
    Constrained as Constrained,
    Random as Random,
    tagged as tagged,
    derived as derived,
    constrained as constrained,
    bounded as bounded,
    random as random,
    is_variable as is_variable,
    is_param as is_param,
    as_param as as_param,
    as_variable as as_variable,
    as_fixed as as_fixed,
)

from parax.filters import (
    is_distribution as is_distribution,
    is_bijector as is_bijector,
    remove as remove,
)

from parax import (
    annotation as annotation,
    bounds as bounds,
    constants as constants,
    constraints as constraints,
    experimental as experimental,
    probability as probability,
    variables as variables,
    wrappers as wrappers,
)