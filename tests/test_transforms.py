"""
Tests for Parax transforms.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy.testing as npt
from distreqx.bijectors import Shift as DistreqxShift

# Adjust the import path to match your project structure
from parax.transforms import (
    Affine,
    Shift,
    Scale,
    Clip,
    LogSoftmax,
    Round,
    Reshape,
    Clip,
    Softmax,
    Normalize,
    Chain,
    Bijective,
    Leafwise,
    Custom,
)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_array(key):
    return jax.random.normal(key, (5, 5))


def test_affine():
    """Tests the Affine transformation."""
    x = jnp.array([1.0, 2.0, 3.0])
    transform = Affine(shift=1.0, scale=2.0)
    
    out = transform(x)
    expected = jnp.array([3.0, 5.0, 7.0])
    
    npt.assert_allclose(out, expected)


def test_shift():
    """Tests the pure Shift transformation."""
    x = jnp.array([1.0, 2.0])
    transform = Shift(shift=5.0)
    
    out = transform(x)
    expected = jnp.array([6.0, 7.0])
    
    npt.assert_allclose(out, expected)


def test_scale():
    """Tests the pure Scale transformation."""
    x = jnp.array([1.0, -2.0])
    transform = Scale(scale=3.0)
    
    out = transform(x)
    expected = jnp.array([3.0, -6.0])
    
    npt.assert_allclose(out, expected)


def test_clip():
    """Tests the Clip transformation bounds."""
    x = jnp.array([-10.0, 0.0, 10.0])
    transform = Clip(lower=-5.0, upper=5.0)
    
    out = transform(x)
    expected = jnp.array([-5.0, 0.0, 5.0])
    
    npt.assert_allclose(out, expected)


def test_softmax(sample_array):
    """Tests the Softmax transformation over a specific axis."""
    transform = Softmax(axis=-1)
    out = transform(sample_array)
    
    # Softmax probabilities should sum to 1.0 along the specified axis
    sums = jnp.sum(out, axis=-1)
    expected_sums = jnp.ones(sample_array.shape[0])
    
    npt.assert_allclose(sums, expected_sums, rtol=1e-5)
    # Output must be strictly positive
    assert jnp.all(out >= 0.0)


def test_reshape():
    """Tests structural reshaping of arrays."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    transform = Reshape(shape=(2, 2))
    
    out = transform(x)
    expected = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    
    assert out.shape == (2, 2)
    npt.assert_allclose(out, expected)

def test_round():
    """Tests quantization and rounding."""
    x = jnp.array([1.1, 2.5, 3.8, -1.2])
    
    # Test integer rounding
    transform_int = Round(decimals=0)
    npt.assert_allclose(transform_int(x), jnp.array([1.0, 2.0, 4.0, -1.0]))
    
    # Test decimal rounding
    x_dec = jnp.array([1.123, 2.567])
    transform_dec = Round(decimals=1)
    npt.assert_allclose(transform_dec(x_dec), jnp.array([1.1, 2.6]))

def test_log_softmax(sample_array):
    """Tests the numerically stable LogSoftmax."""
    transform_log = LogSoftmax(axis=-1)
    transform_soft = Softmax(axis=-1)
    
    out_log = transform_log(sample_array)
    out_soft = transform_soft(sample_array)
    
    # exp(log_softmax) should equal softmax
    npt.assert_allclose(jnp.exp(out_log), out_soft, rtol=1e-5)


def test_normalize(sample_array):
    """Tests the Normalize transformation for zero mean and unit variance."""
    transform = Normalize(axis=None, epsilon=0.0)  # Epsilon 0 for exact checks
    out = transform(sample_array)
    
    mean = jnp.mean(out)
    var = jnp.var(out)
    
    npt.assert_allclose(mean, 0.0, atol=1e-6)
    npt.assert_allclose(var, 1.0, rtol=1e-5)


def test_chain():
    """Tests sequential composition of transformations in mathematical order."""
    x = jnp.array([2.0])
    
    # (x * 2) + 3 = 7
    transform = Chain([Shift(3.0), Scale(2.0)])
    out = transform(x)
    npt.assert_allclose(out, jnp.array([7.0]))
    
    # (x + 3) * 2 = 10
    transform_reverse = Chain([Scale(2.0), Shift(3.0)])
    out_reverse = transform_reverse(x)
    npt.assert_allclose(out_reverse, jnp.array([10.0]))


def test_bijector_transform():
    """Tests integration with a distreqx bijector."""
    x = jnp.array([1.0, 2.0])
    # distreqx Shift bijector
    bijector = DistreqxShift(shift=jnp.array(5.0))
    transform = Bijective(bijector=bijector)
    
    out = transform(x)
    npt.assert_allclose(out, jnp.array([6.0, 7.0]))


def test_tree_transform():
    """Tests mapping transformations over a PyTree."""
    # Note: This test assumes `parax.filters.is_transform` correctly 
    # identifies your subclasses.
    inputs = {
        "a": jnp.array(1.0),
        "b": [jnp.array(2.0), jnp.array(3.0)]
    }
    
    transforms = {
        "a": Shift(2.0),
        "b": [Scale(2.0), Shift(-1.0)]
    }
    
    tree_transform = Leafwise(transforms)
    out = tree_transform(inputs)
    
    npt.assert_allclose(out["a"], jnp.array(3.0))
    npt.assert_allclose(out["b"][0], jnp.array(4.0))
    npt.assert_allclose(out["b"][1], jnp.array(2.0))


def test_custom_transform():
    """Tests a lambda function wrapped in Custom."""
    x = jnp.array([1.0, 2.0, 3.0])
    
    # Square the input
    transform = Custom(fn=lambda x: x ** 2)
    out = transform(x)
    
    npt.assert_allclose(out, jnp.array([1.0, 4.0, 9.0]))


def test_transforms_are_jittable():
    """
    Because transforms inherit from equinox.Module, they should be 
    safely jittable without being marked as static arguments.
    """
    x = jnp.array([1.0, 2.0])
    transform = Affine(shift=1.0, scale=2.0)
    
    @jax.jit
    def apply_transform(t, inputs):
        return t(inputs)
        
    out = apply_transform(transform, x)
    npt.assert_allclose(out, jnp.array([3.0, 5.0]))