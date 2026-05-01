# test_tree.py

import pytest
import jax.numpy as jnp
from unittest.mock import MagicMock, patch

from parax import Parameter
import parax as prx
from parax.constraints import Unconstrained, AbstractConstraint
from distreqx.distributions import ImproperUniform
from distreqx.bijectors import Scale

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

class MockConstraint(AbstractConstraint):
    @property
    def bounds(self):
        return (-jnp.inf, jnp.inf)
    @property
    def bijector(self):
        return Scale(2.0)


@pytest.fixture
def mock_constraint():
    return MockConstraint()

@pytest.fixture
def paramtree(mock_constraint):
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

def test_raw_values(paramtree):
    raws = prx.paramtree.raw_values(paramtree)
    
    assert jnp.allclose(raws["params_a"], 2.0)
    # p2 uses a constraint bijector that halves the base_value (4.0 / 2 = 2.0)
    assert jnp.allclose(raws["params_b"][0], 2.0)
    assert jnp.allclose(raws["params_b"][1], -1.0)

def test_base_values(paramtree):
    bases = prx.paramtree.base_values(paramtree)
    
    assert jnp.allclose(bases["params_a"], 2.0)
    assert jnp.allclose(bases["params_b"][0], 4.0)
    # p3 raw is -1.0, no constraint, so base is -1.0
    assert jnp.allclose(bases["params_b"][1], -1.0)

def test_physical_values(paramtree):
    physicals = prx.paramtree.physical_values(paramtree)
    
    # physical = base * scale
    assert jnp.allclose(physicals["params_a"], 6.0)   # 2.0 * 3.0
    assert jnp.allclose(physicals["params_b"][0], 4.0)  # 4.0 * 1.0
    assert jnp.allclose(physicals["params_b"][1], -10.0) # -1.0 * 10.0

# -------------------------------------------------------------------------
# Tests: Metadata Extractors
# -------------------------------------------------------------------------

def test_scales(paramtree):
    scales_tree = prx.paramtree.scales(paramtree)
    
    assert jnp.allclose(scales_tree["params_a"], 3.0)
    assert jnp.allclose(scales_tree["params_b"][0], 1.0)
    assert jnp.allclose(scales_tree["params_b"][1], 10.0)

def test_fixed(paramtree):
    fixed_tree = prx.paramtree.fixed(paramtree)
    
    assert not fixed_tree["params_a"]
    assert fixed_tree["params_b"][0]
    assert not fixed_tree["params_b"][1]

def test_names(paramtree):
    names_tree = prx.paramtree.names(paramtree)
    
    assert names_tree["params_a"] == "p1"
    assert names_tree["params_b"][0] == "p2"
    assert names_tree["params_b"][1] == ["p3", "alias"]

def test_metadata(paramtree):
    meta_tree = prx.paramtree.metadata(paramtree)
    
    assert meta_tree["params_a"] == {"layer": "dense1"}
    assert meta_tree["params_b"][0] == {}
    assert meta_tree["params_b"][1] == {"layer": "dense2"}

# -------------------------------------------------------------------------
# Tests: Bijectors & Complex Classes
# -------------------------------------------------------------------------

@patch("parax.paramtree.Joint")
def test_distribution_extraction(joint_mock, paramtree):
    """Test that `distribution` correctly wraps the tree in a Joint distribution."""
    _ = prx.paramtree.distribution(paramtree)
    
    # Joint should be instantiated with a PyTree of distributions
    joint_mock.assert_called_once()
    called_tree = joint_mock.call_args[0][0]
    assert isinstance(called_tree["params_a"], ImproperUniform)


@patch("parax.paramtree.TreeConstraint")
def test_constraint_extraction(tree_constraint_mock, paramtree):
    """Test that `constraint` correctly wraps the tree in a TreeConstraint."""
    _ = prx.paramtree.constraint(paramtree)
    
    tree_constraint_mock.assert_called_once()
    called_tree = tree_constraint_mock.call_args[0][0]
    assert isinstance(called_tree["params_a"], Unconstrained)
    assert called_tree["params_b"][0] is not None


@patch("parax.paramtree.TreeMap")
def test_bijector_mappings(tree_map_mock, paramtree):
    """Tests that the combined bijectors are mapped correctly."""
    prx.paramtree.raw_to_base_bijector(paramtree)
    tree_map_mock.assert_called()
    
    prx.paramtree.raw_to_physical_bijector(paramtree)
    assert tree_map_mock.call_count == 2
    
    prx.paramtree.base_to_physical_bijector(paramtree)
    assert tree_map_mock.call_count == 3