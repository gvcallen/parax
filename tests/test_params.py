import pytest
import jax.numpy as jnp
from tests.dummy_model import MathModel

def test_with_params_array(base_model: MathModel):
    """Tests the high-efficiency eqx.partition array unflattening path."""
    flat_vals = base_model.flat_param_values()
    new_vals = flat_vals + 1.5
    updated_model = base_model.with_params(new_vals)
    
    # Direct access inspection
    assert jnp.allclose(updated_model.affine.loc.latent_value, 51.5)
    assert jnp.allclose(updated_model.quadratic.a.latent_value, 2.7)
    assert jnp.allclose(updated_model.vector.latent_value, jnp.array([11.5, 1.5, -8.5]))

def test_with_params_dict(base_model: MathModel):
    """Tests updating parameters via dictionary mapping."""
    updates = {'affine_loc': base_model.affine.loc.with_value(100.0)}
    updated_model = base_model.with_params(updates)
    assert jnp.allclose(updated_model.affine.loc.latent_value, 100.0)

def test_with_params_size_mismatch(base_model: MathModel):
    """Ensures the array path catches shape mismatches."""
    bad_array = jnp.array([1.0, 2.0])
    with pytest.raises(Exception, match="Array size mismatch"):
        base_model.with_params(bad_array)

def test_with_mapped_params(base_model: MathModel):
    """Tests native JAX tree mapping over parameters."""
    updated_model = base_model.with_mapped_params(
        mapper=lambda p: p.with_value(p.latent_value * 2.0), 
        param_filter=['affine'], 
        prefixes=True
    )
    
    assert jnp.allclose(updated_model.affine.loc.latent_value, 100.0)
    assert jnp.allclose(updated_model.affine.scale.latent_value, 0.0)
    assert jnp.allclose(updated_model.quadratic.a.latent_value, 1.2)

def test_fixed_and_free_params(base_model: MathModel):
    """Tests fixing and freeing parameter utilities via direct attribute inspection."""
    # Fix a specific parameter
    fixed_model = base_model.with_fixed_params('affine_loc')
    assert fixed_model.affine.loc.fixed is True
    assert fixed_model.quadratic.a.fixed is False

    # Free parameters only (should fix the others)
    freed_only_model = fixed_model.with_free_params_only(['affine_loc'])
    assert freed_only_model.affine.loc.fixed is False
    assert freed_only_model.quadratic.a.fixed is True

    # Fix all
    all_fixed = base_model.with_all_params_fixed()
    assert all_fixed.affine.loc.fixed is True
    assert all_fixed.quadratic.a.fixed is True
    
    # Free all
    all_free = all_fixed.with_all_params_free()
    assert all_free.affine.loc.fixed is False
    assert all_free.quadratic.a.fixed is False