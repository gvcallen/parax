from typing import Any

from jaxtyping import PyTree, Scalar
import equinox as eqx


class OptimizeResults(eqx.Module):
    """
    The standardized output of a parax optimization run.
    """
    #: The fully reconstructed parax model with optimized parameters.
    model: PyTree

    #: The final fn value
    final_value: Scalar

    #: Auxiliary data returned by the objective function on the final evaluation step.
    #: Will be None if `has_aux=False`.
    aux: Any = None

    #: The raw backend-specific solution/metrics object (e.g., optx.Solution).
    metrics: PyTree | None = None