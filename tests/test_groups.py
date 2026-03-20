import pytest
import numpyro.distributions as dist
from tests.dummy_model import CircuitModel

def test_param_groups_recursive_extraction(base_model: CircuitModel):
    """Ensures param_groups correctly lifts nested groups to the root level."""
    groups = base_model.param_groups()
    resonator_group = next((g for g in groups if 'resonator_inductance' in g.param_names), None)
    
    assert resonator_group is not None
    assert 'resonator_capacitance' in resonator_group.param_names
    assert isinstance(resonator_group.distribution, dist.Normal)

def test_with_no_param_groups(base_model: CircuitModel):
    """Verifies that param groups can be recursively wiped out."""
    wiped_model = base_model.with_no_param_groups()
    
    # Should only return singletons for each free parameter (no grouped distributions)
    groups = wiped_model.param_groups()
    for g in groups:
        assert len(g.param_names) == 1