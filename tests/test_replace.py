import pytest
import dataclasses
from parax import tree_replace

@dataclasses.dataclass
class DummyNode:
    val: int
    flag: bool

def test_tree_replace_no_kwargs():
    """Test that passing no kwargs returns the original tree unharmed."""
    tree = {"a": DummyNode(1, False)}
    res = tree_replace(tree)
    assert res["a"] is tree["a"]

def test_tree_replace_global_broadcast():
    """Test broadcasting a single scalar replacement across all leaves in the PyTree."""
    tree = {
        "leaf1": DummyNode(1, False),
        "leaf2": DummyNode(2, False)
    }
    
    res = tree_replace(tree, val=10)
    
    assert res["leaf1"].val == 10
    assert res["leaf2"].val == 10
    # Flags should remain untouched
    assert res["leaf1"].flag is False
    assert res["leaf2"].flag is False

def test_tree_replace_parallel_tree():
    """Test replacing fields using a parallel tree structure."""
    tree = [DummyNode(1, False), DummyNode(2, True)]
    
    # We pass a list matching the exact structure of the parent tree
    res = tree_replace(tree, val=[100, 200])
    
    assert res[0].val == 100
    assert res[1].val == 200

def test_tree_replace_multiple_kwargs():
    """Test replacing multiple properties simultaneously."""
    tree = DummyNode(1, False)
    
    res = tree_replace(tree, val=99, flag=True)
    
    assert res.val == 99
    assert res.flag is True

def test_tree_replace_is_leaf_none_error():
    """
    Test that if is_leaf=None, JAX maps into standard primitives 
    and raises the expected TypeError caught by the function.
    """
    tree = [1, 2] # Raw primitives, not dataclasses

    with pytest.raises(TypeError, match="Expected a dataclass leaf"):
        tree_replace(tree, is_leaf=None, val=10)

def test_tree_replace_custom_is_leaf():
    """Test using a custom is_leaf function safely bypasses non-matching nodes."""
    @dataclasses.dataclass
    class IgnoreMe:
        val: int
        
    tree = [DummyNode(1, False), IgnoreMe(5)]
    
    # Only target DummyNode for replacement
    res = tree_replace(
        tree, 
        is_leaf=lambda x: isinstance(x, DummyNode), 
        val=10
    )
    
    assert res[0].val == 10
    assert res[1].val == 5  # Ignored node remains untouched