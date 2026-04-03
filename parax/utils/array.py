from typing import Any

import jax.numpy as jnp

def serialize_array(arr: jnp.ndarray | None) -> Any:
    """Helper to safely serialize real or complex JAX arrays."""
    if arr is None:
        return None
    if jnp.iscomplexobj(arr):
        return {
            "__is_complex__": True, 
            "real": jnp.real(arr).tolist(), 
            "imag": jnp.imag(arr).tolist()
        }
    return arr.tolist()

def deserialize_array(data: Any) -> Any:
    """Helper to reconstruct real or complex JAX arrays from JSON data."""
    if data is None:
        return None
    if isinstance(data, dict) and data.get("__is_complex__"):
        return jnp.array(data["real"]) + 1j * jnp.array(data["imag"])
    return data # jnp.asarray handles standard lists natively

def format_array(val: float | jnp.ndarray, sig_figs: int = 4) -> str:
    """Safely extract and format a scalar or array value for clean printing."""
    if jnp.iscomplexobj(val):
        if hasattr(val, "ndim") and val.ndim == 0:
            return f"{complex(val):.{sig_figs}g}"
        elif hasattr(val, "size") and val.size == 1:
            return f"{complex(val.item()):.{sig_figs}g}"
        else:
            # Fallback for multi-dimensional arrays
            return f"f64{list(val.shape)}" if hasattr(val, "shape") else repr(val)        
    else:
        if hasattr(val, "ndim") and val.ndim == 0:
            return f"{float(val):.{sig_figs}g}"
        elif hasattr(val, "size") and val.size == 1:
            return f"{float(val.item()):.{sig_figs}g}"
        else:
            # Fallback for multi-dimensional arrays
            return f"f64{list(val.shape)}" if hasattr(val, "shape") else repr(val)