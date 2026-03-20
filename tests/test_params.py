import pytest
import jax.numpy as jnp
from tests.dummy_model import CircuitModel

def test_with_params_array(base_model: CircuitModel):
    """Tests the high-efficiency eqx.partition array unflattening path."""
    flat_vals = base_model.flat_param_values()
    new_vals = flat_vals + 1.5
    updated_model = base_model.with_params(new_vals)
    
    # Direct access inspection
    assert jnp.allclose(updated_model.input_match.resistance.value, 51.5)
    assert jnp.allclose(updated_model.resonator.inductance.value, 2.7)
    assert jnp.allclose(updated_model.gain.value, jnp.array([11.5, 1.5, -8.5]))

def test_with_params_dict(base_model: CircuitModel):
    """Tests updating parameters via dictionary mapping."""
    updates = {'input_match_resistance': base_model.input_match.resistance.with_value(100.0)}
    updated_model = base_model.with_params(updates)
    assert jnp.allclose(updated_model.input_match.resistance.value, 100.0)

def test_with_params_size_mismatch(base_model: CircuitModel):
    """Ensures the array path catches shape mismatches."""
    bad_array = jnp.array([1.0, 2.0])
    with pytest.raises(Exception, match="Array size mismatch"):
        base_model.with_params(bad_array)

def test_with_mapped_params(base_model: CircuitModel):
    """Tests native JAX tree mapping over parameters."""
    updated_model = base_model.with_mapped_params(
        mapper=lambda p: p.with_value(p.value * 2.0), 
        param_filter=['input_match'], 
        prefixes=True
    )
    
    assert jnp.allclose(updated_model.input_match.resistance.value, 100.0)
    assert jnp.allclose(updated_model.input_match.reactance.value, 0.0)
    assert jnp.allclose(updated_model.resonator.inductance.value, 1.2)

def test_fixed_and_free_params(base_model: CircuitModel):
    """Tests fixing and freeing parameter utilities via direct attribute inspection."""
    # Fix a specific parameter
    fixed_model = base_model.with_fixed_params('input_match_resistance')
    assert fixed_model.input_match.resistance.fixed is True
    assert fixed_model.resonator.inductance.fixed is False

    # Free parameters only (should fix the others)
    freed_only_model = fixed_model.with_free_params_only(['input_match_resistance'])
    assert freed_only_model.input_match.resistance.fixed is False
    assert freed_only_model.resonator.inductance.fixed is True

    # Fix all
    all_fixed = base_model.with_all_params_fixed()
    assert all_fixed.input_match.resistance.fixed is True
    assert all_fixed.resonator.inductance.fixed is True
    
    # Free all
    all_free = all_fixed.with_all_params_free()
    assert all_free.input_match.resistance.fixed is False
    assert all_free.resonator.inductance.fixed is False