"""
Bijectors, filled in from distreqx where available.

parax supports any distreqx: the latest release, the unreleased upstream main,
or gvcallen's fork. Names the installed distreqx provides are re-exported from
it; the rest fall back to parax's own copies, so importing from here always
works regardless of which distreqx is installed.

`Sigmoid` is always parax's own. It is not a missing name but a fixed one:
distreqx's version produces NaN gradients where the input saturates, and a
capability check cannot detect that.
"""

from distreqx.bijectors import (
    AbstractBijector as AbstractBijector,
    AbstractForwardInverseBijector as AbstractForwardInverseBijector,
    AbstractFwdLogDetJacBijector as AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector as AbstractInvLogDetJacBijector,
    AbstractLinearBijector as AbstractLinearBijector,
    Block as Block,
    Chain as Chain,
    DiagLinear as DiagLinear,
    ScalarAffine as ScalarAffine,
    Shift as Shift,
    Tanh as Tanh,
    TriangularLinear as TriangularLinear,
    UnconstrainedAffine as UnconstrainedAffine,
)

from parax._vendor._sigmoid import Sigmoid as Sigmoid
from parax._bijectors import Elementwise as Elementwise

try:
    from distreqx.bijectors import Identity as Identity
except ImportError:
    from parax._bijectors import Identity as Identity

try:
    from distreqx.bijectors import Inverse as Inverse
except ImportError:
    from parax._bijectors import Inverse as Inverse

try:
    from distreqx.bijectors import Softplus as Softplus
except ImportError:
    from parax._bijectors import Softplus as Softplus

try:
    from distreqx.bijectors import Leafwise as Leafwise
except ImportError:
    from parax._bijectors import Leafwise as Leafwise

try:
    from distreqx.bijectors import Exp as Exp
except ImportError:
    from parax._vendor._exp import Exp as Exp

try:
    from distreqx.bijectors import Permute as Permute
except ImportError:
    from parax._vendor._permute import Permute as Permute

try:
    from distreqx.bijectors import R2ToComplex as R2ToComplex
except ImportError:
    from parax._vendor._r2_to_complex import R2ToComplex as R2ToComplex

try:
    from distreqx.bijectors import Transpose as Transpose
except ImportError:
    from parax._vendor._transpose import Transpose as Transpose

try:
    from distreqx.bijectors import Reshape as Reshape
except ImportError:
    from parax._vendor._reshape import Reshape as Reshape

try:
    from distreqx.bijectors import Restructure as Restructure
except ImportError:
    from parax._vendor._restructure import Restructure as Restructure

try:
    from distreqx.bijectors import Split as Split
except ImportError:
    from parax._vendor._split import Split as Split


__all__ = [
    "AbstractBijector",
    "AbstractForwardInverseBijector",
    "AbstractFwdLogDetJacBijector",
    "AbstractInvLogDetJacBijector",
    "AbstractLinearBijector",
    "Block",
    "Chain",
    "DiagLinear",
    "Elementwise",
    "Exp",
    "Identity",
    "Inverse",
    "Leafwise",
    "Permute",
    "R2ToComplex",
    "Reshape",
    "Restructure",
    "ScalarAffine",
    "Shift",
    "Sigmoid",
    "Softplus",
    "Split",
    "Tanh",
    "Transpose",
    "TriangularLinear",
    "UnconstrainedAffine",
]
