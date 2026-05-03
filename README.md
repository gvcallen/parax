![Parax Logo](https://raw.githubusercontent.com/gvcallen/parax/main/assets/logo.png)

| **Parax** |  |
|-------------|-------|
| **Author**  | Gary Allen |
| **Homepage** | [github.com/parax/parax](https://github.com/parax/parax) |
| **Docs** | [gvcallen.github.io/parax](https://gvcallen.github.io/parax) |

**Parax** is a library for parametric modeling in [JAX](https://github.com/jax-ml/jax).

## Features

- Derived/fixed parameters
- PyTree parameterizations and freezing
- Built-in constraints and metadata support
- Filtering and PyTree manipulation tools.

## Installation
Parax can be installed using pip:

``
pip install parax
``

You may need a custom `distreqx` branch for some constraints:

``
pip install git+https://github.com/gvcallen/distreqx.git
``

## Overview

**Parax** aims to provide a foundation for "parametric modeling", with focus on libraries such as [Equinox](https://github.com/patrick-kidger/equinox).

Parametric modeling here means modeling with a focus on the concept of a model parameter as a *derived array with metadata*. This means supporting parameterizations, constraints, units, arbitrary metadata, etc.

The design was motivated by several others libraries, including [Flax](https://github.com/google/flax), [paramax](https://github.com/danielward27/paramax), and [PyTorch](https://github.com/pytorch/pytorch).

## Example 1: Constrained Parameters

`parax.Param` represents a simple JAX array with metadata, while `parax.Constrained` also caters for built-in *constraints*. Both classes override `parax.AbstractVariable`, providing array-like behaviour via `__jax_array__`.

The example below demonstrates defining a `parax.Constrained` parameter and then using it in a JAX expression.

```python
import jax.numpy as jnp
import parax as prx

# Define a parameter bounded between 0 and 10
p = prx.Constrained(8.0, prx.Interval(0.0, 10.0))

# We can print any constraint's bounds
print(f"Bounds: {p.bounds}")

# We can use the parameter directly in an equation and print the result
result = jnp.sin(p) + (p * 2.0)
print(f"Result: {result}") 
print(f"Raw (unconstrained) value: {p.raw_value}")

# We could have also unwrapped directly
assert jnp.allclose(prx.unwrap(p), 8.0)
```

## Example 2: Optimizing a Model using Optimistix

In this example, we define a damped pendulum model using `equinox.Module`. We set the first parameter as unconstrained, the second as only positive with a scale of "mm", and the third as a fixed variable.

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optimistix as optx
from dataclasses import replace
import parax as prx 

class DampedPendulum(eqx.Module):
    # Unconstrained variable (creates a prx.Param).
    friction: prx.Variable = prx.param(0.1) 
    
    # Constrained variable (creates a prx.Physical)
    length: prx.Variable = prx.physical(9.81, scale='mm', constraint=prx.Positive())

    # Dummy variable (to be fixed)
    k: prx.Variable = prx.param(1.0)

    def __call__(self, state):
        return self.k * state * self.friction / self.length
    
# Create our model and fix the multiplier
initial_model = DampedPendulum()
initial_model = replace(initial_model, k=prx.Fixed(initial_model.k))

# Partition the model, stopping at any constants e.g. `prx.Fixed` variables and `prx.Frozen` layers.
# Then, define the loss function.
params, static = eqx.partition(initial_model, eqx.is_inexact_array, is_leaf=prx.is_constant)
def loss_fn(params, args):
    model = prx.unwrap(eqx.combine(params, static))
    x, y = args
    predictions = jax.vmap(model)(x)
    return jnp.sum((predictions - y)**2)

# Generate some dummy data with friction/length ratio of 0.25
x_data = jnp.linspace(0, 10, 100)
noise = jr.normal(jr.key(42), x_data.shape) * 0.1
y_data = x_data * 0.25 + noise

# Run the optimization
solver = optx.LBFGS(rtol=1e-5, atol=1e-5)
results = optx.minimise(
    fn=loss_fn,
    solver=solver, 
    y0=params, 
    args=(x_data, y_data),
)

# Print the results
final_model = prx.unwrap(eqx.combine(results.value, static))
print(f"Optimized Friction: {final_model.friction:.4f}")
print(f"Optimized Length: {final_model.length:.4f}")
print(f"Optimized Ratio: {final_model.friction / final_model.length:.4f}")
```
