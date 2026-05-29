import jax
import jax.numpy as jnp

from parax import (
    unwrap,
    Parameterize,
    Apply,
    Freeze,
)

def test_standard_pytree_unwrap():
    """Ensure unwrap acts as a safe no-op for standard PyTrees."""
    tree = {"a": 1, "b": jnp.array([1.0, 2.0]), "c": "metadata"}
    unwrapped = unwrap(tree)
    
    assert unwrapped["a"] == 1
    assert jnp.allclose(unwrapped["b"], jnp.array([1.0, 2.0]))
    assert unwrapped["c"] == "metadata"

def test_parameterized():
    """Test that Parameterized executes its callable with args and kwargs."""
    def dummy_fn(x, y, multiplier=1.0):
        return (x + y) * multiplier

    param_node = Parameterize(dummy_fn, 2.0, 3.0, multiplier=2.0)
    result = unwrap(param_node)
    
    assert result == 10.0

def test_computed_bypasses_non_arrays():
    """Test that Computed maps over arrays but leaves strings/metadata alone."""
    tree = {
        "x": jnp.array(2.0),
        "y": jnp.array(3.0),
        "meta": "should_not_change",
        "flag": False
    }
    
    # Apply a square function
    computed_node = Apply(lambda val: val ** 2 if not isinstance(val, bool) else True, tree)
    result = unwrap(computed_node)
    
    assert jnp.allclose(result["x"], 4.0)
    assert jnp.allclose(result["y"], 9.0)
    assert result["meta"] == "should_not_change"
    assert result["flag"] is True

def test_nested_unwrappables():
    """Test the recursive inside-out resolution of nested Unwrappables."""
    # Inner node: generates a base array
    inner = Parameterize(lambda: jnp.array(5.0))
    
    # Outer node: computes the square of whatever the inner node produces
    outer = Apply(lambda x: x ** 2, {"val": inner})
    
    result = unwrap(outer)
    
    # 5.0 gets generated, then squared
    assert jnp.allclose(result["val"], 25.0)

def test_freeze_stops_gradients():
    """Test that Freeze actually applies jax.lax.stop_gradient."""
    def loss_fn(x):
        # We wrap x in Freeze. The unwrapped value should be detached from the graph.
        freeze_x = Freeze(x)
        val = unwrap(freeze_x)
        return val ** 2  # Normally, d(x^2)/dx = 2x. 

    # Gradient of a detached graph should be exactly 0.0 in JAX
    grad_fn = jax.grad(loss_fn)
    gradient = grad_fn(jnp.array(3.0))
    
    assert jnp.allclose(gradient, 0.0)

def test_freeze_double_wrap_prevention():
    """Test the init safeguard against Freeze(Freeze(x))."""
    base_array = jnp.array([1.0, 2.0])
    f1 = Freeze(base_array)
    f2 = Freeze(f1)
    
    # f2 should absorb f1 and point directly to the base array
    assert f2.tree is base_array
    assert not isinstance(f2.tree, Freeze)

def test_freeze_as_free():
    """Test that as_free returns the underlying tree."""
    tree = {"a": jnp.array(1.0)}
    f = Freeze(tree)
    
    freed = f.free()
    assert freed is tree


def test_unwrap_cascade_true_matches_outer():
    """Test that cascade=True unwraps the matched node and all of its descendants."""
    inner = Parameterize(lambda: 42.0)
    # The outer node is just an identity function that returns whatever is inside it
    outer = Parameterize(lambda x: x, inner)
    
    # Condition: Match ONLY the outer node
    condition = lambda node: node is outer
    
    # Because cascade=True (default), hitting the outer node forces the 
    # inner node to unwrap as well.
    result = unwrap(outer, only_if=condition, cascade=True)
    
    # Both unwrapped: outer(inner()) -> 42.0
    assert result == 42.0

def test_unwrap_cascade_false_matches_outer():
    """Test that cascade=False strictly unwraps ONLY the matched node."""
    inner = Parameterize(lambda: 42.0)
    outer = Parameterize(lambda x: x, inner)
    
    # Condition: Match ONLY the outer node
    condition = lambda node: node is outer
    
    # Because cascade=False, the inner node is spared from unwrapping.
    # The outer node unwraps and evaluates its lambda, returning the still-wrapped inner node.
    result = unwrap(outer, only_if=condition, cascade=False)
    
    assert isinstance(result, Parameterize)
    
    # We can verify it still holds its latent value
    assert result.unwrap() == 42.0

def test_unwrap_cascade_false_matches_inner():
    """Test bottom-up reconstruction when an inner node matches but an outer node does not."""
    inner = Parameterize(lambda: 42.0)
    outer = Parameterize(lambda x: x, inner)
    
    # Condition: Match ONLY the inner node
    condition = lambda node: node is inner
    
    # cascade=False means the tree is traversed bottom-up. 
    # The inner node unwraps into 42.0, and the outer node stays wrapped.
    result = unwrap(outer, only_if=condition, cascade=False)
    
    # The outer node itself is still wrapped
    assert isinstance(result, Parameterize)
    assert result is not outer  # It's a newly reconstructed PyTree node
    
    # Because JAX reconstructs the PyTree bottom-up, the new outer node 
    # now contains the fully unwrapped `42.0` instead of the `inner` node.
    # Calling unwrap() on this new outer node should yield the raw number.
    assert result.unwrap() == 42.0