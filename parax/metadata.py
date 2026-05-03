from typing import Any

import equinox as eqx

class AbstractHasMetadata(eqx.Module):
    """An abstract interface for an Equinox module with metadata."""
    #: Returns the underlying metadata.
    metadata: eqx.AbstractVar[dict[Any, Any]]