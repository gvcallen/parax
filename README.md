![Parax Logo](https://raw.githubusercontent.com/gvcallen/parax/main/assets/logo.png)

| **Parax** |  |
|-------------|-------|
| **Author**  | Gary Allen |
| **Homepage** | [github.com/parax/parax](https://github.com/parax/parax) |
| **Docs** | [gvcallen.github.io/parax](https://gvcallen.github.io/parax) |

**Parax** is a mini-framework for parametric modeling in [JAX](https://github.com/jax-ml/jax).

## Features

The library provides:

- Parameters with metadata and constraints
- Computable/derived parameters
- PyTree parameterizations
- Filtering and PyTree manipulation tools

## Installation
Parax can be installed using pip:

``
pip install parax
``

You may need a custom `distreqx` branch for some constraints:

``
pip install git+https://github.com/gvcallen/distreqx.git
``


## Motivation

The goal of **Parax** is to provide a foundation for parameter-driven modeling with libraries like [Equinox](https://github.com/patrick-kidger/equinox), to be used directly or as part of other frameworks.

Although Equinox has a strong filtering system, it lacks the ability to easily attach metadata and apply constraints/parameterizations directly to model parameters. This is heavily desired in both scientific modeling and modern machine learning (especially for bounded optimization or probabilistic modeling).

This library therefore implements the above features in the hopes to make such approaches more accessible and straight-forward while still following Equinox's core principles. The features are motivated by several libraries such as [Flax](https://github.com/google/flax), [paramax](https://github.com/danielward27/paramax) and [PyTorch](https://github.com/pytorch/pytorch).

``

## Example 1: Parameters Constraints

While `parax.Param` represents a simple wrapper around a JAX array, `parax.Constrained` allows for arbitrary parameter constrained for bounded and unbounded optimizers. Both classes implement `parax.AbstractFreeVariable` and therefore `parax.AbstractVariable`, which overrides the experimental `__jax_array__` interface and implements several dunder methods.

This example demonstrates defining a parameter with an interval constraint, as well as evaluating it interactively (i.e. without explicit unwrapping).

```python
import jax.numpy as jnp
import parax as prx

# Define a parameter bounded between 0 and 10
p = prx.Constrained(8.0, prx.Interval(0.0, 10.0))

# Use the parameter directly in an equation and print the result
result = jnp.sin(p) + (p * 2.0)
print(f"Result: {result}") 
print(f"Raw (unconstrained) value: {p.raw_value}")

# We could have also unwrapped directly
assert jnp.allclose(prx.unwrap(p), 8.0)
```

## Example 2: Optimizing a Model using Optimistix

In this example, we define a damped pendulum model using `equinox.Module`. We set the first parameter is unconstrained, the second as only positive with a scale of mm, and the third is a fixed variable.

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optimistix as optx
from dataclasses import replace
import parax as prx 

class DampedPendulum(eqx.Module):
    # Unconstrained "ML" parameter
    friction: prx.Variable = prx.param(0.1) 
    
    # Constrained "SciML" parameter
    length: prx.Variable = prx.physical(9.81, scale='mm', constraint=prx.Positive())

    # Dummy multiplier
    k: prx.Variable = prx.param(1.0)

    def __call__(self, state):
        return self.k * state * self.friction / self.length
    
# Create our model and then fix the multiplier
initial_model = DampedPendulum()
initial_model = replace(initial_model, k=prx.Fixed(initial_model.k))

# Partition the model and define the loss function
params, static = eqx.partition(initial_model, prx.is_free_variable, is_leaf=prx.is_variable)
def loss_fn(params, args):
    model = prx.unwrap(eqx.combine(params, static, is_leaf=prx.is_variable))
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
final_model = prx.unwrap(results.value)
print(f"Optimized Friction: {final_model.friction:.4f}")
print(f"Optimized Length: {final_model.length:.4f}")
print(f"Optimized Ratio: {final_model.friction / final_model.length:.4f}")
```
