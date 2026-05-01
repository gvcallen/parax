# test_tree.py

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from unittest.mock import MagicMock, patch

# Assuming your modules are inside a package named `parax`
from parax.parameter import Parameter
import parax.tree as ptree

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def mock_bijector():
    """A simple mock bijector that doubles the input for testing purposes."""
    mock = MagicMock()
    mock.forward.side_effect = lambda x: x * 2.0
    mock.inverse.side_effect = lambda x: x / 2.0
    return mock

@pytest.fixture
def mock_constraint(mock_bijector):
    mock = MagicMock()
    mock.bijector = mock_bijector
    return mock

@pytest.fixture
def param_tree(mock_constraint):
    """
    Creates a PyTree of Parameter objects with mixed configurations.
    """
    p1 = Parameter(
        base_value=2.0, 
        scale=3.0, 
        name="p1", 
        metadata={"layer": "dense1"}
    )
    
    p2 = Parameter(
        base_value=4.0, 
        scale=1.0, 
        fixed=True, 
        name="p2", 
        constraint=mock_constraint
    )
    
    p3 = Parameter(
        raw_value=-1.0, 
        scale=10.0, 
        name=["p3", "alias"], 
        metadata={"layer": "dense2"}
    )
    
    return {
        "params_a": p1,
        "params_b": [p2, p3]
    }

# -------------------------------------------------------------------------
# Tests: Value Extractors
# -------------------------------------------------------------------------

def test_raw_values(param_tree):
    raws = ptree.raw_values(param_tree)
    
    assert jnp.allclose(raws["params_a"], 2.0)
    # p2 uses a constraint bijector that halves the base_value (4.0 / 2 = 2.0)
    assert jnp.allclose(raws["params_b"][0], 2.0)
    assert jnp.allclose(raws["params_b"][1], -1.0)

def test_base_values(param_tree):
    bases = ptree.base_values(param_tree)
    
    assert jnp.allclose(bases["params_a"], 2.0)
    assert jnp.allclose(bases["params_b"][0], 4.0)
    # p3 raw is -1.0, no constraint, so base is -1.0
    assert jnp.allclose(bases["params_b"][1], -1.0)

def test_physical_values(param_tree):
    physicals = ptree.physical_values(param_tree)
    
    # physical = base * scale
    assert jnp.allclose(physicals["params_a"], 6.0)   # 2.0 * 3.0
    assert jnp.allclose(physicals["params_b"][0], 4.0)  # 4.0 * 1.0
    assert jnp.allclose(physicals["params_b"][1], -10.0) # -1.0 * 10.0

# -------------------------------------------------------------------------
# Tests: Metadata Extractors
# -------------------------------------------------------------------------

def test_scales(param_tree):
    scales_tree = ptree.scales(param_tree)
    
    assert jnp.allclose(scales_tree["params_a"], 3.0)
    assert jnp.allclose(scales_tree["params_b"][0], 1.0)
    assert jnp.allclose(scales_tree["params_b"][1], 10.0)

def test_fixed(param_tree):
    fixed_tree = ptree.fixed(param_tree)
    
    assert not fixed_tree["params_a"]
    assert fixed_tree["params_b"][0]
    assert not fixed_tree["params_b"][1]

def test_names(param_tree):
    names_tree = ptree.names(param_tree)
    
    assert names_tree["params_a"] == "p1"
    assert names_tree["params_b"][0] == "p2"
    assert names_tree["params_b"][1] == ["p3", "alias"]

def test_metadata(param_tree):
    meta_tree = ptree.metadata(param_tree)
    
    assert meta_tree["params_a"] == {"layer": "dense1"}
    assert meta_tree["params_b"][0] is None
    assert meta_tree["params_b"][1] == {"layer": "dense2"}

# -------------------------------------------------------------------------
# Tests: Tree Setters / Mutators
# -------------------------------------------------------------------------

def test_replace_raw_values(param_tree):
    # Create a new PyTree of matching structure with updated raw arrays
    new_raws = {
        "params_a": jnp.array(99.0),
        "params_b": [jnp.array(88.0), jnp.array(77.0)]
    }
    
    new_tree = ptree.replace_raw_values(param_tree, new_raws)
    
    # Check that raw values updated
    assert jnp.allclose(new_tree["params_a"].raw_value, 99.0)
    assert jnp.allclose(new_tree["params_b"][0].raw_value, 88.0)
    assert jnp.allclose(new_tree["params_b"][1].raw_value, 77.0)
    
    # Check that metadata remained intact
    assert new_tree["params_a"].name == "p1"
    assert new_tree["params_b"][0].fixed == True
    assert jnp.allclose(new_tree["params_b"][1].scale, 10.0)

# -------------------------------------------------------------------------
# Tests: Bijectors & Complex Classes
# -------------------------------------------------------------------------

@patch("parax.tree.Joint")
def test_distribution_extraction(joint_mock, param_tree):
    """Test that `distribution` correctly wraps the tree in a Joint distribution."""
    _ = ptree.distribution(param_tree)
    
    # Joint should be instantiated with a PyTree of distributions
    joint_mock.assert_called_once()
    called_tree = joint_mock.call_args[0][0]
    assert called_tree["params_a"] is None  # We didn't set a dist in the fixture


@patch("parax.tree.TreeConstraint")
def test_constraint_extraction(tree_constraint_mock, param_tree):
    """Test that `constraint` correctly wraps the tree in a TreeConstraint."""
    _ = ptree.constraint(param_tree)
    
    tree_constraint_mock.assert_called_once()
    called_tree = tree_constraint_mock.call_args[0][0]
    assert called_tree["params_a"] is None
    assert called_tree["params_b"][0] is not None # The mock constraint


@patch("parax.tree.TreeMap")
def test_bijector_mappings(tree_map_mock, param_tree):
    """Tests that the combined bijectors are mapped correctly."""
    ptree.raw_to_base_bijector(param_tree)
    tree_map_mock.assert_called()
    
    ptree.raw_to_physical_bijector(param_tree)
    assert tree_map_mock.call_count == 2
    
    ptree.base_to_physical_bijector(param_tree)
    assert tree_map_mock.call_count == 3