import jax.numpy as jnp
import numpyro.distributions as dist
from parax import Module, Parameter, ParameterGroup

class MatchingNetwork(Module):
    """A simple leaf module."""
    resistance: Parameter = 50.0
    reactance: Parameter = 0.0

class Resonator(Module):
    """A nested module with its own parameter group."""
    inductance: Parameter = 1.2
    capacitance: Parameter = 0.8

    def __post_init__(self):
        # Assign a dummy joint distribution to test group extraction
        group = ParameterGroup(
            param_names=['inductance', 'capacitance'], 
            distribution=dist.Normal(jnp.zeros(2), jnp.ones(2))
        )
        object.__setattr__(self, '_param_groups', [group])

class CircuitModel(Module):
    """The root model containing submodules and a vectorized parameter."""
    input_match: MatchingNetwork
    resonator: Resonator
    # Vectorized parameter (e.g., gain at different frequency points) to test array logic
    gain: Parameter = jnp.array([10.0, 0.0, -10.0])