import pytest
import jax.numpy as jnp
from tests.dummy_model import CircuitModel, MatchingNetwork, Resonator

def test_with_modules(base_model: CircuitModel):
    """Tests the combining of free parameters from a different module instance."""
    alt_match = MatchingNetwork(resistance=75.0, reactance=10.0)
    alt_res = Resonator()
    alt_model = CircuitModel(input_match=alt_match, resonator=alt_res)
    
    combined_model = base_model.with_modules(alt_model)
    assert jnp.allclose(combined_model.input_match.resistance.value, 75.0)

def test_with_fields_and_name(base_model: CircuitModel):
    """Tests standard dataclass replacement utilities."""
    named_model = base_model.with_name("MyCircuit")
    assert named_model.name == "MyCircuit"
    
    alt_match = MatchingNetwork(resistance=99.0)
    replaced_model = base_model.with_fields(input_match=alt_match)
    assert jnp.allclose(replaced_model.input_match.resistance.value, 99.0)

def test_with_submodule_fields(base_model: CircuitModel):
    """Tests nested field replacement using dot notation."""
    # Create the new parameter object we want to swap in
    new_param = base_model.input_match.resistance.with_value(42.0)
    
    # Target 'input_match' and swap its 'resistance' keyword argument
    updated = base_model.with_submodule_fields('input_match', resistance=new_param)
    
    assert jnp.allclose(updated.input_match.resistance.value, 42.0)

def test_submodule_fixing_and_freeing(base_model: CircuitModel):
    """Tests bulk submodule fixing via direct access."""
    fixed_match = base_model.with_fixed_submodules('input_match')
    
    assert fixed_match.input_match.resistance.fixed is True
    assert fixed_match.input_match.reactance.fixed is True
    assert fixed_match.resonator.inductance.fixed is False # unaffected

    freed_only = fixed_match.with_free_submodules_only('input_match', include_fixed=True)
    assert freed_only.input_match.resistance.fixed is False
    assert freed_only.resonator.inductance.fixed is True # 'fix_others' kicked in

def test_submodule_isolation_workflow(base_model: CircuitModel):
    """
    Tests the 'isolation' workflow where a user wants to fix the rest of the model 
    but preserve the exact internal free/fixed configuration of a target submodule.
    """
    # 1. Set up a mixed internal state in the input match
    # Resistance is free (default), Reactance is fixed
    mixed_model = base_model.with_fixed_params('input_match_reactance')
    
    # Verify initial mixed state
    assert mixed_model.input_match.resistance.fixed is False
    assert mixed_model.input_match.reactance.fixed is True
    
    # Resonator parameters are both currently free
    assert mixed_model.resonator.inductance.fixed is False
    assert mixed_model.resonator.capacitance.fixed is False

    # 2. Isolate the input_match using include_fixed=False
    # This should leave input_match exactly as it is, and fix everything else.
    isolated_model = mixed_model.with_free_submodules_only(
        'input_match', 
        include_fixed=False
    )
    
    # 3. Verify target submodule's internal state was perfectly preserved
    assert isolated_model.input_match.resistance.fixed is False
    assert isolated_model.input_match.reactance.fixed is True
    
    # 4. Verify the rest of the circuit was successfully locked down
    assert isolated_model.resonator.inductance.fixed is True
    assert isolated_model.resonator.capacitance.fixed is True
    assert isolated_model.gain.fixed is True