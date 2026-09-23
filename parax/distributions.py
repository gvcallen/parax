"""
Distributions, filled in from distreqx where available.

parax supports any distreqx: the latest release, the unreleased upstream main,
or gvcallen's fork. Names the installed distreqx provides are re-exported from
it; the rest fall back to parax's own copies, so importing from here always
works regardless of which distreqx is installed.
"""

from distreqx.distributions import (
    AbstractCDFDistribution as AbstractCDFDistribution,
    AbstractDistribution as AbstractDistribution,
    AbstractProbDistribution as AbstractProbDistribution,
    AbstractSampleLogProbDistribution as AbstractSampleLogProbDistribution,
    AbstractSTDDistribution as AbstractSTDDistribution,
    AbstractSurvivalDistribution as AbstractSurvivalDistribution,
    AbstractTransformed as AbstractTransformed,
    Bernoulli as Bernoulli,
    Beta as Beta,
    Categorical as Categorical,
    Gamma as Gamma,
    Independent as Independent,
    Logistic as Logistic,
    MixtureSameFamily as MixtureSameFamily,
    MultivariateNormalDiag as MultivariateNormalDiag,
    MultivariateNormalFullCovariance as MultivariateNormalFullCovariance,
    MultivariateNormalTri as MultivariateNormalTri,
    Normal as Normal,
    OneHotCategorical as OneHotCategorical,
    Transformed as Transformed,
    Uniform as Uniform,
)

try:
    from distreqx.distributions import ImproperUniform as ImproperUniform
except ImportError:
    from parax._vendor._improper_uniform import ImproperUniform as ImproperUniform

try:
    from distreqx.distributions import Joint as Joint
except ImportError:
    from parax._vendor._joint import Joint as Joint

try:
    from distreqx.distributions import LogNormal as LogNormal
except ImportError:
    from parax._vendor._lognormal import LogNormal as LogNormal

try:
    from distreqx.distributions import TruncatedNormal as TruncatedNormal
except ImportError:
    from parax._vendor._truncated_normal import TruncatedNormal as TruncatedNormal


__all__ = [
    "AbstractCDFDistribution",
    "AbstractDistribution",
    "AbstractProbDistribution",
    "AbstractSTDDistribution",
    "AbstractSampleLogProbDistribution",
    "AbstractSurvivalDistribution",
    "AbstractTransformed",
    "Bernoulli",
    "Beta",
    "Categorical",
    "Gamma",
    "ImproperUniform",
    "Independent",
    "Joint",
    "LogNormal",
    "Logistic",
    "MixtureSameFamily",
    "MultivariateNormalDiag",
    "MultivariateNormalFullCovariance",
    "MultivariateNormalTri",
    "Normal",
    "OneHotCategorical",
    "Transformed",
    "TruncatedNormal",
    "Uniform",
]
