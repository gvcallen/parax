import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from parax.wrappers import Tie, _TiePlaceholder, unwrap

# ==========================================
# FIXTURES & DUMMY MODELS
# ==========================================

class DummyModel(eqx.Module):
    a: jax.Array
    b: jax.Array
    c: jax.Array
    d: jax.Array

@pytest.fixture
def base_model():
    return DummyModel(
        a=jnp.array(2.0),
        b=jnp.array(3.0),
        c=jnp.array(4.0),
        d=jnp.array(5.0),
    )

# ==========================================
# TESTS
# ==========================================

def test_tied_initialization(base_model):
    """Tests that the target is successfully stripped and replaced with a placeholder."""
    tied_model = Tie(
        tree=base_model,
        target=lambda m: m.b,
        source=lambda m: m.a
    )
    
    # The target in the internal tree should be the placeholder
    assert isinstance(tied_model.tree.b, _TiePlaceholder)
    # The source should remain untouched
    assert tied_model.tree.a == jnp.array(2.0)
    # The tie metadata should be registered
    assert len(tied_model.ties) == 1


def test_tied_unwrap_basic(base_model):
    """Tests that unwrap correctly computes the tied value on the fly."""
    tied_model = Tie(
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
    
    
def test_tied_multiple(base_model):
    """Tests that we can tie multiple values at once."""
    # Tie b to a and d to c
    tied_model = Tie(
        tree=base_model,
        target=lambda m: (m.b, m.d),
        source=lambda m: (m.a, m.c),
    )
    
    active_model = unwrap(tied_model)
    
    assert active_model.b == active_model.a
    assert active_model.d == active_model.c


def test_tied_chaining(base_model):
    """Tests that multiple Tied wrappers can be chained sequentially."""
    # b = a * 2 (b becomes 4.0)
    tied_1 = Tie(
        tree=base_model,
        target=lambda m: m.b,
        source=lambda m: m.a,
        tie_fn=lambda a: a * 2.0
    )
    
    # c = b + 10 (c becomes 14.0)
    tied_2 = Tie(
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
    tied_model = Tie(
        tree=base_model,
        target=lambda m: m.b,
        source=lambda m: m.a
    )
    
    # Filter for trainable arrays
    trainable_params, _ = eqx.partition(tied_model, eqx.is_inexact_array)
    
    # Tree leaves of trainable params should only contain 'a', 'c' and 'd'
    leaves = jax.tree_util.tree_leaves(trainable_params)
    assert len(leaves) == 3
    
    assert isinstance(trainable_params.tree.b, _TiePlaceholder)


def test_gradient_flow_and_updates(base_model):
    """Tests that gradients flow properly and the tied parameter updates dynamically."""
    tied_model = Tie(
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

# ==========================================
# UNWRAPPING THE TIED VALUE
# ==========================================
#
# `unwrap` must return a tree with nothing left to unwrap. Copying a source across
# as-is breaks that whenever the source is itself unwrappable, and leaves the tied
# position looking like a second independent node rather than a derived value.


def test_tied_position_holds_a_value_not_the_source_node():
    """A tied position receives the source's value, not whatever produced it."""
    from parax.variables import Real

    class Model(eqx.Module):
        a: object
        b: object

    model = Model(a=Real(jnp.array(2.0)), b=jnp.array(3.0))
    tied = Tie(tree=model, target=lambda m: m.b, source=lambda m: m.a)

    resolved = tied.unwrap()
    assert not isinstance(resolved.b, Real)
    assert jnp.allclose(jnp.asarray(resolved.b), 2.0)


def test_tied_position_is_resolved_before_the_source_is():
    """
    The tied position holds a value after a single step, while the source is left for
    the surrounding recursion to resolve as it would anywhere else.
    """
    from parax.variables import Real
    from parax.wrappers import is_unwrappable

    class Model(eqx.Module):
        a: object
        b: object

    model = Model(a=Real(jnp.array(2.0)), b=jnp.array(3.0))
    tied = Tie(tree=model, target=lambda m: m.b, source=lambda m: m.a)

    stepped = tied.unwrap()
    assert not is_unwrappable(stepped.b)
    assert is_unwrappable(stepped.a)

    fully = unwrap(tied)
    assert not is_unwrappable(fully.a)
    assert not is_unwrappable(fully.b)
    assert jnp.allclose(jnp.asarray(fully.a), jnp.asarray(fully.b))


def test_tied_node_is_not_counted_twice():
    """
    A tie means one node appearing in two places, so anything walking the unwrapped
    tree must see one source of truth and one derived value -- not two equals.
    """
    from parax.variables import Real

    class Model(eqx.Module):
        a: object
        b: object

    untied = Model(a=Real(jnp.array(2.0)), b=Real(jnp.array(2.0)))
    tied = Tie(
        tree=Model(a=Real(jnp.array(2.0)), b=jnp.array(3.0)),
        target=lambda m: m.b,
        source=lambda m: m.a,
    ).unwrap()

    count = lambda tree: sum(
        isinstance(leaf, Real)
        for leaf in jax.tree_util.tree_leaves(tree, is_leaf=lambda x: isinstance(x, Real))
    )
    assert count(untied) == 2
    assert count(tied) == 1


def test_plain_pytree_ties_are_unchanged(base_model):
    """Values are untouched where the source has nothing to unwrap."""
    tied = Tie(tree=base_model, target=lambda m: m.b, source=lambda m: m.a)
    resolved = unwrap(tied)

    assert jnp.allclose(resolved.a, 2.0)
    assert jnp.allclose(resolved.b, 2.0)
    assert jnp.allclose(resolved.c, 4.0)
    assert jnp.allclose(resolved.d, 5.0)


def test_tie_fn_receives_the_unwrapped_value():
    """`tie_fn` operates on the value, so ordinary arithmetic works on any source."""
    from parax.variables import Real

    class Model(eqx.Module):
        a: object
        b: object

    model = Model(a=Real(jnp.array(2.0)), b=jnp.array(0.0))
    tied = Tie(
        tree=model,
        target=lambda m: m.b,
        source=lambda m: m.a,
        tie_fn=lambda x: x * 10.0,
    )

    assert jnp.allclose(jnp.asarray(tied.unwrap().b), 20.0)
