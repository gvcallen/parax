import pytest
from distreqx.distributions import Normal
from tests.dummy_model import MathModel

def test_param_groups_recursive_extraction(base_model: MathModel):
    """Ensures param_groups correctly lifts nested groups to the root level."""
    groups = base_model.param_groups()
    quadratic_group = next((g for g in groups if 'quadratic_a' in g.param_names), None)
    
    assert quadratic_group is not None
    assert 'quadratic_b' in quadratic_group.param_names
    assert isinstance(quadratic_group.distribution, Normal)

def test_with_no_param_groups(base_model: MathModel):
    """Verifies that param groups can be recursively wiped out."""
    wiped_model = base_model.with_no_param_groups()
    
    # Should only return singletons for each free parameter (no grouped distributions)
    groups = wiped_model.param_groups()
    for g in groups:
        assert len(g.param_names) == 1
        
    groups_explicit = wiped_model.param_groups(explicit_only=True)
    assert len(groups_explicit) == 0