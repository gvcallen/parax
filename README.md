![Parax Logo](https://raw.githubusercontent.com/gvcallen/parax/main/assets/logo.png)

**Parax** is a mini-framework designed for parametric/scientific modeling in [JAX](https://github.com/jax-ml/jax).

It uses [Equinox](https://github.com/patrick-kidger/equinox) to provide `parax.Parameter` - a custom PyTree class representing a model parameter with metadata. Further, **Parax** provides useful tools and wrappers for optimization, inference, and model inspection/manipulation.

| **Parax** |  |
|-------------|-------|
| **Author**  | Gary Allen |
| **Homepage** | [github.com/parax/parax](https://github.com/parax/parax) |
| **Docs** | [gvcallen.github.io/parax](https://gvcallen.github.io/parax) |

## Features

- **Parameters with Metadata**: `parax.Parameter` is a JAX PyTree providing common physical metadata, such as `fixed`, `scale`, `constraint` and `distribution` (via [distreqx](https://github.com/lockwo/distreqx)), as well as arbitrary metadata support. `parax.param` provides a matching field specifier.
- **Unit Support**: Support for units in the `scale` field (via [unxt](https://github.com/GalacticDynamics/unxt)).
- **Optimization and Inference Wrappers**: Out-of-the-box support for both optimization ((via [optimistix](https://github.com/patrick-kidger/optimistix) and `scipy.optimize.minimize`)) and Bayesian inference (via [BlackJAX](https://github.com/blackjax-devs/blackjax)).
- **ParamTree Manipulation**: Easy manipulation of PyTree's containing `parax.Parameter` leaf-nodes ("ParamTrees") via built-in filters and mapping utilities including `parax.partition`, `parax.combine`, `parax.is_free_param`, and advanced extractors in `parax.paramtree`.

## Installation
Parax can be installed using pip:

``
pip install parax
``

You likely also need a custom `distreqx` branch:

``
pip install git+https://github.com/gvcallen/distreqx.git
``

## Overview

In classical/physical modeling, you care about raw arrays alone, but rather focus on concept Parameters: values that have physical constraints, scales, units, and prior distributions. In JAX-land, the common way to supply such metadata is to work with "shadow" PyTrees - separate PyTrees with a structure that "shadows" your original model structure but only contains each separate pieces of metadata.

Using the above approach directly, however, can be very tedious in some applications, where you commonly want to define metadata right during "model creation", and also potentially manipulate this metadata during "model preparation". Parax aims to make this workflow possible by providing a `Parameter` class alongside tree utilities to unpack and manipulate any resultant "ParamTrees". This allows the above workflow to be compatible with common JAX transformations.

Further, to allow for experimentation with models without manual unwrapping (e.g. in a Jupyter notebook), Parax overides the (experimental) `__jax_array__` protocol, allowing parameters to behave just like JAX arrays for simple applications.

## Example 1: Parameters Constraints

This example demonstrate defining a parameter with constraints, and then evaluating it interactively without unwrapping.

```python
import jax.numpy as jnp
import parax as prx
from parax.constraints import Interval

# Define a parameter bounded between 0 and 10 with a starting physical value of 5.0
p = prx.Parameter(8.0, constraint=Interval(0.0, 10.0), name="transmission_rate")

# Use the parameter directly in math! 
result = jnp.sin(p) + (p * 2.0)

print(f"Physical Result: {result}") 
print(f"Raw (unconstrained) value: {p.raw_value}")
```

## Example 2: Optimizing an Model using Optimistix

In this example, we define a simple quadratic model ($y = ax^2 + bx + c$) using `equinox.Module` and `parax.Parameter`. We provide a default for the first parameter, fix the y-intercept, and use `parax.optimize.minimize` with `optimistix` to fit the model to some noisy data.

```python
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx

import parax as prx

# 1. Define the Parametric Model
class Quadratic(eqx.Module):
    """A generic quadratic curve: y = a*x^2 + b*x + c"""
    
    a: prx.Parameter = prx.param(1.5)
    b: prx.Parameter = prx.param(0.0)
    c: prx.Parameter = prx.param(0.0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.a * (x ** 2) + self.b * x + self.c
    
# We pass in free/fixed parameters without metadata using factories.
model = Quadratic(b=prx.Parameter(0.5), c=prx.Parameter(10.0, fixed=True))

# 2. Generate some dummy "ground truth" data with noise
x_true = jnp.linspace(-5.0, 5.0, 100)
y_true = 3.0 * (x_true ** 2) - 2.0 * x_true + 10.0 # True a=3.0, b=-2.0
y_true = y_true + jax.random.normal(jax.random.key(0), x_true.shape)

# 4. Define the loss Function
def loss_fn(model, args=None):
    y_pred = model(x_true)
    return jnp.mean((y_pred - y_true)**2)

# 5. Run the BFGS optimizer
solver = optx.LBFGS(rtol=1e-6, atol=1e-6)
results = prx.optimize.minimize(
    fn=loss_fn,
    solver=solver,
    y0=model,
)

fitted_model = results.model

print(f"Fitted 'a': {jnp.array(fitted_model.a):.8f} (Expected ~3.0)")
print(f"Fitted 'b': {jnp.array(fitted_model.b):.8f} (Expected ~-2.0)")
print(f"Fixed 'c':  {jnp.array(fitted_model.c):.8f} (Remained 10.0)")
print(f'Final loss: {results.final_value}')
```
