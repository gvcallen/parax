import pytest
import jax.numpy as jnp
import numpyro.distributions as dist
from tests.dummy_model import CircuitModel

def test_with_uniform_distributions_mixed_array(base_model: CircuitModel):
    """Tests that uniform distributions handle positive, zero, and negative vectorized bounds natively."""
    updated_model = base_model.with_uniform_distributions(percentage=0.1, zero_values='keep')
    
    gain_param = updated_model.gain
    assert isinstance(gain_param.distribution, dist.Uniform)
    
    # Original gain: [10.0, 0.0, -10.0]
    # Expected low:  [9.0, 0.0, -11.0]
    # Expected high: [11.0, 0.0, -9.0]
    assert jnp.allclose(gain_param.distribution.low, jnp.array([9.0, 0.0, -11.0]))
    assert jnp.allclose(gain_param.distribution.high, jnp.array([11.0, 0.0, -9.0]))

def test_with_mapped_distributions(base_model: CircuitModel):
    """Tests custom distribution mapping across the tree."""
    def to_normal(d):
        return dist.Normal(0.0, 1.0)
        
    updated_model = base_model.with_mapped_distributions(to_normal)
    
    assert isinstance(updated_model.input_match.resistance.distribution, dist.Normal)