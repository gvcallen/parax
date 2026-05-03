"""
Interfaces for attaching metadata to Equinox modules.
"""

from typing import Any

import equinox as eqx

class AbstractHasMetadata(eqx.Module):
    """
    An abstract interface for an Equinox module that carries arbitrary metadata.

    In Parax, metadata is typically used to store units, descriptions, tags, 
    or optimization hints alongside the underlying JAX array.
    """
    #: Returns the underlying metadata dictionary.
    metadata: eqx.AbstractVar[dict[Any, Any]]