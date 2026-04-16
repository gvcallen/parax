import pytest
import jax
import jax.numpy as jnp

from distreqx.distributions import Normal
from parax.distributions import ModuleDistribution
from tests.dummy_model import MathModel

@pytest.fixture
def prob_model(base_model: MathModel) -> MathModel:
    """Populates missing distributions so the module forms a valid joint space."""
    def add_prior(p):
        if getattr(p, "distribution", None) is None:
            return p.with_distribution(Normal(0.0, 1.0))
        return p
    return base_model.with_mapped_params(add_prior)

def test_distribution_initialization_and_shape(prob_model: MathModel):
    dist = ModuleDistribution(prob_model)
    event_shape = dist.event_shape
    
    # Event shape should be a dictionary mapping names to shapes
    assert isinstance(event_shape, dict)
    assert 'affine_loc' in event_shape
    assert event_shape['affine_loc'] == ()

def test_distribution_sample(prob_model: MathModel):
    dist = ModuleDistribution(prob_model)
    key = jax.random.key(42)
    
    sample_dict = dist.sample(key)
    
    # 1. The keys must match exactly
    assert set(sample_dict.keys()) == set(dist.event_shape.keys())
    
    # 2. The shapes of the sampled arrays must match the expected event shapes
    for name, array in sample_dict.items():
        assert array.shape == dist.event_shape[name]
    
    # Verify we can seamlessly pass this dictionary back to Parax
    sampled_mod = prob_model.with_params(sample_dict)
    assert isinstance(sampled_mod, type(prob_model))
    assert not jnp.allclose(prob_model.flat_param_values(), sampled_mod.flat_param_values())

def test_distribution_log_prob(prob_model: MathModel):
    dist = ModuleDistribution(prob_model)
    
    # Facade accepts the Module directly
    lp = dist.log_prob(prob_model)
    assert isinstance(lp, jax.Array)
    assert lp.shape == ()
    
    # Facade accepts the raw dictionary PyTree
    dynamic_dict = prob_model.named_flat_param_values()
    lp_from_dict = dist.log_prob(dynamic_dict)
    assert jnp.allclose(lp, lp_from_dict)

def test_distribution_sample_and_log_prob(prob_model: MathModel):
    dist = ModuleDistribution(prob_model)
    key = jax.random.key(123)
    
    sample_dict, log_prob = dist.sample_and_log_prob(key)
    
    assert isinstance(sample_dict, dict)
    assert log_prob.shape == ()
    assert jnp.allclose(log_prob, dist.log_prob(sample_dict))

def test_distribution_tree_metrics(prob_model: MathModel):
    dist = ModuleDistribution(prob_model)
    
    mean_tree = dist.mean()
    var_tree = dist.variance()
    
    # They should all be flat dictionaries
    assert isinstance(mean_tree, dict)
    assert isinstance(var_tree, dict)
    assert 'affine_loc' in mean_tree
    
    # Variance must be strictly positive
    for val in jax.tree_util.tree_leaves(var_tree):
        assert jnp.all(val >= 0.0)

def test_distribution_scalar_metrics(prob_model: MathModel):
    dist = ModuleDistribution(prob_model)
    entropy_val = dist.entropy()
    
    assert isinstance(entropy_val, jax.Array)
    assert entropy_val.shape == ()

def test_distribution_kl_divergence(prob_model: MathModel):
    dist1 = ModuleDistribution(prob_model)
    
    # Shift only Normal distributions safely
    def shift_normal(d):
        if isinstance(d, Normal):
            return Normal(loc=d.loc + 1.0, scale=d.scale)
        return d
        
    shifted_model = prob_model.with_mapped_distributions(shift_normal, param_groups=True)
    dist2 = ModuleDistribution(shifted_model)
    
    kl_identical = dist1.kl_divergence(dist1)
    assert jnp.allclose(kl_identical, 0.0, atol=1e-5)
    
    kl_different = dist1.kl_divergence(dist2)
    assert kl_different > 0.0
    assert kl_different.shape == ()