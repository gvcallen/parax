from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

from parax.replace import tree_replace as tree_replace

from parax.constant import (
    is_constant as is_constant,
    as_free as as_free,
)
from parax.annotated import is_annotated as is_annotated
from parax.bounded import is_bounded as is_bounded
from parax.probabilistic import is_probabilistic as is_probabilistic
from parax.constraints import is_constraint as is_constraint
from parax.constrainable import is_constrainable as is_constrainable
from parax.transforms import is_transform as is_transform

from parax.unwrappable import (
    AbstractUnwrappable as AbstractUnwrappable,
    unwrap as unwrap,
    is_unwrappable as is_unwrappable,
    as_unwrapped as as_unwrapped,
)

from parax.wrappable import (
    AbstractWrappable as AbstractWrappable,
    wrap as wrap,
    is_wrappable as is_wrappable,
)

from parax.wrappers import (
    Frozen as Frozen,
    Parameterized as Parameterized,
    Computed as Computed,
    Static as Static,
    Tied as Tied,
    as_frozen as as_frozen,
    as_static as as_static,
    as_frozen_or_static as as_frozen_or_static,
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
    is_variable as is_variable,
    is_param as is_param,
    as_param as as_param,
    as_fixed as as_fixed,
)

from parax.filters import (
    is_distribution as is_distribution,
    is_bijector as is_bijector,
    remove as remove,
)


from parax import (
    annotated as annotated,
    constant as constant,
    constraints as constraints,
    constrainable as constrainable,
    bounded as bounded,
    experimental as experimental,
    probabilistic as probabilistic,
    variables as variables,
    transforms as transforms,
    wrappers as wrappers,
)