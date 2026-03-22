import jax.numpy as jnp
import numpyro.distributions as dist
from parax import Module, Parameter, ParameterGroup, Fixed

class Affine(Module):
    """A simple leaf module."""
    loc: Parameter = 50.0
    scale: Parameter = 0.0

class Quadratic(Module):
    """A nested module with its own parameter group."""
    a: Parameter = 1.2
    b: Parameter = 0.8
    c: Parameter = Fixed(0.0)

    def __post_init__(self):
        # Assign a dummy joint distribution to test group extraction
        group = ParameterGroup(
            param_names=['a', 'b'], 
            distribution=dist.Normal(jnp.zeros(2), jnp.ones(2))
        )
        self._param_groups = [group]

class MathModel(Module):
    """The root model containing submodules and a vectorized parameter."""
    affine: Affine
    quadratic: Quadratic
    # Vectorized parameter (e.g., vector at different frequency points) to test array logic
    vector: Parameter = jnp.array([10.0, 0.0, -10.0])