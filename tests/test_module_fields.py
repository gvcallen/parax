import pytest
import jax.numpy as jnp
from tests.dummy_model import MathModel, Affine, Quadratic

def test_with_modules(base_model: MathModel):
    """Tests the combining of free parameters from a different module instance."""
    alt_match = Affine(loc=75.0, scale=10.0)
    alt_res = Quadratic()
    alt_model = MathModel(affine=alt_match, quadratic=alt_res)
    
    combined_model = base_model.with_modules(alt_model)
    assert jnp.allclose(combined_model.affine.loc.value, 75.0)

def test_with_fields_and_name(base_model: MathModel):
    """Tests standard dataclass replacement utilities."""
    named_model = base_model.with_name("MyMathModel")
    assert named_model.name == "MyMathModel"
    
    alt_match = Affine(loc=99.0)
    replaced_model = base_model.with_fields(affine=alt_match)
    assert jnp.allclose(replaced_model.affine.loc.value, 99.0)

def test_with_submodule_fields(base_model: MathModel):
    """Tests nested field replacement using dot notation."""
    # Create the new parameter object we want to swap in
    new_param = base_model.affine.loc.with_value(42.0)
    
    # Target 'affine' and swap its 'loc' keyword argument
    updated = base_model.with_submodule_fields('affine', loc=new_param)
    
    assert jnp.allclose(updated.affine.loc.value, 42.0)

def test_submodule_fixing_and_freeing(base_model: MathModel):
    """Tests bulk submodule fixing via direct access."""
    fixed_match = base_model.with_fixed_submodules('affine')
    
    assert fixed_match.affine.loc.fixed is True
    assert fixed_match.affine.scale.fixed is True
    assert fixed_match.quadratic.a.fixed is False # unaffected

    freed_only = fixed_match.with_free_submodules_only('affine', include_fixed=True)
    assert freed_only.affine.loc.fixed is False
    assert freed_only.quadratic.a.fixed is True # 'fix_others' kicked in

def test_submodule_isolation_workflow(base_model: MathModel):
    """
    Tests the 'isolation' workflow where a user wants to fix the rest of the model 
    but preserve the exact internal free/fixed configuration of a target submodule.
    """
    # 1. Set up a mixed internal state in the input match
    # loc is free (default), scale is fixed
    mixed_model = base_model.with_fixed_params('affine_scale')
    
    # Verify initial mixed state
    assert mixed_model.affine.loc.fixed is False
    assert mixed_model.affine.scale.fixed is True
    
    # Quadratic parameters are both currently free
    assert mixed_model.quadratic.a.fixed is False
    assert mixed_model.quadratic.b.fixed is False

    # 2. Isolate the affine using include_fixed=False
    # This should leave affine exactly as it is, and fix everything else.
    isolated_model = mixed_model.with_free_submodules_only(
        'affine', 
        include_fixed=False
    )
    
    # 3. Verify target submodule's internal state was perfectly preserved
    assert isolated_model.affine.loc.fixed is False
    assert isolated_model.affine.scale.fixed is True
    
    # 4. Verify the rest of the model was successfully locked down
    assert isolated_model.quadratic.a.fixed is True
    assert isolated_model.quadratic.b.fixed is True
    assert isolated_model.vector.fixed is True