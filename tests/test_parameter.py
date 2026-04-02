import pytest
import json
import dataclasses
import jax.numpy as jnp
import distreqx.distributions as dist
from distreqx.bijectors import Block

# Assuming your library structure looks something like this:
from parax.parameter import Parameter
from parax.parameter_metadata import ParameterMetadata
from parax.parameters import Uniform, Normal, Fixed, Free, RelativeUniform

@pytest.fixture
def dummy_distribution():
    return dist.Normal(loc=0.0, scale=1.0)

class TestParameterInitialization:
    def test_basic_initialization(self):
        param = Parameter(value=5.0)
        assert param.value == 5.0
        assert param.latent_value == 5.0
        assert not param.fixed
        assert param.scale == 1.0

    def test_initialization_with_latent(self):
        param = Parameter(latent_value=jnp.array(2.0))
        assert param.latent_value == 2.0
        assert param.value == 2.0

    def test_initialization_fails_without_value(self):
        with pytest.raises(Exception, match="Must pass one of either"):
            Parameter()

    def test_metadata_routing(self, dummy_distribution):
        param = Parameter(
            value=10.0, 
            name="amplifier_gain", 
            distribution=dummy_distribution,
            custom_info_flag=True
        )
        assert param.name == "amplifier_gain"
        assert param.distribution is dummy_distribution
        assert param.info["custom_info_flag"] is True

    def test_dataclasses_replace_metadata_override(self):
        """Tests the upgraded __init__ logic for dataclasses.replace"""
        param = Parameter(value=1.0, name="filter_cap", initial_flag=False)
        
        # Replace the name and add a new custom info flag
        new_param = dataclasses.replace(param, name="n_plexer_cap", new_flag=True)
        
        assert new_param.name == "n_plexer_cap"
        assert new_param.info["initial_flag"] is False
        assert new_param.info["new_flag"] is True
        assert new_param.value == 1.0

class TestParameterMathAndArrays:
    def test_math_operations(self):
        p1 = Parameter(value=10.0)
        p2 = Parameter(value=2.0)
        
        assert jnp.allclose(p1 + p2, 12.0)
        assert jnp.allclose(p1 - p2, 8.0)
        assert jnp.allclose(p1 * p2, 20.0)
        assert jnp.allclose(p1 / p2, 5.0)
        
        # Broadcasting with scalars
        assert jnp.allclose(p1 * 2.0, 20.0)
        assert jnp.allclose(2.0 * p1, 20.0)

    def test_scale_applied_in_array_conversion(self):
        param = Parameter(value=5.0, scale=10.0)
        # The latent/value is 5.0, but when cast to array it should multiply by scale
        assert jnp.allclose(jnp.array(param), 50.0)

# class TestParameterTransformations:
#     def test_bijector_initialization_inverts_value(self):
#         # Use your custom Exponential bijector directly
#         exp_bij = Exp()
#         param = Parameter(value=1.0, bijector=exp_bij)
        
#         # log(1.0) = 0.0
#         assert jnp.allclose(param.latent_value, 0.0)
#         assert jnp.allclose(param.value, 1.0)

#     def test_transformed_chains_bijectors(self):
#         param = Parameter(value=0.0)
#         exp_bij = Exp()
        
#         transformed_param = param.transformed(exp_bij)
        
#         # Latent remains unchanged
#         assert jnp.allclose(transformed_param.latent_value, 0.0)
#         # Value is mapped through the new bijector forward pass: exp(0) = 1.0
#         assert jnp.allclose(transformed_param.value, 1.0)

#     def test_transformed_updates_bounds(self):
#         param = Parameter(value=2.0, bounds=jnp.array([1.0, 3.0]))
        
#         # Use our new custom Lambda bijector for a simple shift
#         shift_bij = LambdaBijector(
#             fn_forward=lambda x: x + 10.0,
#             fn_inverse=lambda y: y - 10.0,
#             fn_forward_log_det=lambda x: jnp.zeros_like(x),
#             fn_inverse_log_det=lambda y: jnp.zeros_like(y)
#         )
        
#         transformed_param = param.transformed(shift_bij)
#         assert jnp.allclose(transformed_param.bounds, jnp.array([11.0, 13.0]))

class TestParameterFactories:
    def test_uniform_factory(self):
        param = Uniform(low=0.0, high=10.0)
        assert param.value == 5.0  # Midpoint default
        assert isinstance(param.distribution, dist.Uniform)

    def test_uniform_factory_vectorized(self):
        param = Uniform(low=0.0, high=10.0, n=3)
        assert param.shape == (3,)
        assert jnp.allclose(param.value, jnp.array([5.0, 5.0, 5.0]))

    def test_relative_uniform(self):
        param = RelativeUniform(mean=100.0, deviation_fraction=0.1)
        # Bounds should be 90 to 110
        assert param.value == 100.0
        # Check distribution bounds
        assert jnp.allclose(param.distribution.low, 90.0)
        assert jnp.allclose(param.distribution.high, 110.0)

    def test_normal_factory(self):
        param = Normal(mean=50.0, std=5.0)
        assert param.value == 50.0
        assert isinstance(param.distribution, dist.Normal)

    def test_fixed_and_free_factories(self):
        fixed_p = Fixed(value=42.0)
        assert fixed_p.fixed is True
        
        free_p = Free(value=42.0)
        assert free_p.fixed is False

class TestParameterFlattening:
    def test_flattened_scalar(self):
        param = Parameter(value=3.14, name="scalar_param")
        flat = param.flattened()
        assert len(flat) == 1
        assert flat[0] is param

    def test_flattened_array(self):
        param = Parameter(value=jnp.array([1.0, 2.0]), name="vector")
        flat = param.flattened()
        
        assert len(flat) == 2
        assert flat[0].value == 1.0
        assert flat[0].name == "vector_0"
        assert flat[1].value == 2.0
        assert flat[1].name == "vector_1"

    def test_flattened_with_explicit_names(self):
        param = Parameter(value=jnp.array([1.0, 2.0]), name=["alpha", "beta"])
        flat = param.flattened()
        
        assert len(flat) == 2
        assert flat[0].name == "alpha"
        assert flat[1].name == "beta"

class TestParameterSerialization:
    def test_to_and_from_json(self):
        param = Parameter(
            value=3.14, 
            fixed=True, 
            scale=1e-3, 
            name="microstrip_width",
            custom_tolerance=0.05
        )
        
        json_str = param.to_json()
        reconstructed = Parameter.from_json(json_str)
        
        assert reconstructed.value == 3.14
        assert reconstructed.fixed is True
        assert reconstructed.scale == 1e-3
        assert reconstructed.name == "microstrip_width"
        assert reconstructed.info["custom_tolerance"] == 0.05