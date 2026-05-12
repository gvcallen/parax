import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from parax.unwrappable import unwrap
from parax.wrappers import Tied, _TiedPlaceholder

# ==========================================
# FIXTURES & DUMMY MODELS
# ==========================================

class DummyModel(eqx.Module):
    a: jax.Array
    b: jax.Array
    c: jax.Array

@pytest.fixture
def base_model():
    return DummyModel(
        a=jnp.array(2.0),
        b=jnp.array(3.0),
        c=jnp.array(4.0)
    )

# ==========================================
# TESTS
# ==========================================

def test_tied_initialization(base_model):
    """Tests that the target is successfully stripped and replaced with a placeholder."""
    tied_model = Tied(
        tree=base_model,
        target=lambda m: m.b,
        source=lambda m: m.a
    )
    
    # The target in the internal tree should be the placeholder
    assert isinstance(tied_model.tree.b, _TiedPlaceholder)
    # The source should remain untouched
    assert tied_model.tree.a == jnp.array(2.0)
    # The tie metadata should be registered
    assert len(tied_model.ties) == 1


def test_tied_unwrap_basic(base_model):
    """Tests that unwrap correctly computes the tied value on the fly."""
    tied_model = Tied(
        tree=base_model,
        target=lambda m: m.b,
        source=lambda m: m.a,
        tie_fn=lambda a: a * 10.0
    )
    
    active_model = unwrap(tied_model)
    
    # 'a' is 2.0, so 'b' should be 20.0
    assert active_model.b == jnp.array(20.0)
    # 'a' and 'c' should be unchanged
    assert active_model.a == jnp.array(2.0)
    assert active_model.c == jnp.array(4.0)


def test_tied_chaining(base_model):
    """Tests that multiple Tied wrappers can be chained sequentially."""
    # b = a * 2 (b becomes 4.0)
    tied_1 = Tied(
        tree=base_model,
        target=lambda m: m.b,
        source=lambda m: m.a,
        tie_fn=lambda a: a * 2.0
    )
    
    # c = b + 10 (c becomes 14.0)
    tied_2 = Tied(
        tree=tied_1,
        target=lambda m: m.c,
        source=lambda m: m.b,
        tie_fn=lambda b: b + 10.0
    )
    
    active_model = unwrap(tied_2)
    
    assert active_model.a == jnp.array(2.0)
    assert active_model.b == jnp.array(4.0)
    assert active_model.c == jnp.array(14.0)


def test_optimizer_invisibility(base_model):
    """Tests that equinox correctly hides the tied parameter from the optimizer."""
    tied_model = Tied(
        tree=base_model,
        target=lambda m: m.b,
        source=lambda m: m.a
    )
    
    # Filter for trainable arrays
    trainable_params, _ = eqx.partition(tied_model, eqx.is_inexact_array)
    
    # Tree leaves of trainable params should only contain 'a' and 'c'
    leaves = jax.tree_util.tree_leaves(trainable_params)
    assert len(leaves) == 2  
    
    assert isinstance(trainable_params.tree.b, _TiedPlaceholder)


def test_gradient_flow_and_updates(base_model):
    """Tests that gradients flow properly and the tied parameter updates dynamically."""
    tied_model = Tied(
        tree=base_model,
        target=lambda m: m.b,
        source=lambda m: m.a,
        tie_fn=lambda a: a * 3.0
    )
    
    optimizer = optax.sgd(learning_rate=0.5)
    opt_state = optimizer.init(eqx.filter(tied_model, eqx.is_inexact_array))
    
    @eqx.filter_jit
    def step(model, state):
        def loss_fn(m):
            active_m = unwrap(m)
            # Loss = (b - 12.0)^2. 
            # Since b = a * 3, we want b to move to 12, which means 'a' should move to 4.
            return (active_m.b - 12.0) ** 2
            
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, state = optimizer.update(grads, state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, state
        
    updated_model, opt_state = step(tied_model, opt_state)
    active_updated_model = unwrap(updated_model)
    
    # Initial 'a' was 2.0. Initial 'b' was 6.0. 
    # Loss pulls 'b' up towards 12.0, so 'a' should be strictly > 2.0
    assert active_updated_model.a > 2.0
    # 'b' must remain exactly 3x 'a'
    assert jnp.allclose(active_updated_model.b, active_updated_model.a * 3.0)