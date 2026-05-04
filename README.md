![Parax Logo](https://raw.githubusercontent.com/gvcallen/parax/main/assets/logo.png)

| **Parax** |  |
|-------------|-------|
| **Author**  | Gary Allen |
| **Homepage** | [github.com/parax/parax](https://github.com/parax/parax) |
| **Docs** | [gvcallen.github.io/parax](https://gvcallen.github.io/parax) |

**Parax** is a library for parametric modeling in [JAX](https://github.com/jax-ml/jax).

## Features

- Derived/constrained parameters with metadata
- Computed PyTrees and callable parameterizations
- Interfaces for PyTree/parameter fixing, bounds and priors
- Filtering and manipulation tools
- Built-in wrapper for SciPy bounded minimization

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

**Parax** aims to provide a foundation for "parametric modeling", i.e. modeling with a focus on the concept of a parameter as a *derived array with metadata*. This means supporting parameterizations, constraints, bounds, priors, units, and arbitrary metadata, which are needed in both machine learning and scientific modeling.

**Parax** accomplishes the above in a general manner by providing a common set of *abstract interfaces* along with filters and tree utilities that use these interfaces. The goal is then to provide a range of tools and concrete classes to minimize boilerplate for users, while still keeping the library extendable and opt-in.

Although **Parax** can be used in any JAX code, it places emphasis on interoperatibility with [Equinox](https://github.com/patrick-kidger/equinox). For example, `parax.AbstractConstant` and `parax.is_constant` allow easy partitioning of model parameters using `eqx.partition`, with `parax.Fixed` and `parax.Frozen` providing concrete implementations.

The library's design was inspired by several others who deserve mention, including [Flax](https://github.com/google/flax), [paramax](https://github.com/danielward27/paramax), and [PyTorch](https://github.com/pytorch/pytorch).

## Example 1: Constrained Parameters

`parax.Param` represents a simple JAX array with metadata, while `parax.Constrained` also caters for built-in *constraints*. Both classes override `parax.AbstractVariable`, providing array-like behaviour via `__jax_array__`. We call variables and/or arrays "param-like".

The example below demonstrates defining a `parax.Constrained` parameter and then using it in a JAX expression.

```python
import jax.numpy as jnp
import parax as prx

# Define a parameter bounded between 0 and 10
p = prx.Constrained(8.0, prx.Interval(0.0, 10.0))

p.constraint.bounds
# (Array(0., dtype=float32), Array(10., dtype=float32))

# We can use the parameter directly in an equation
jnp.sin(p) + (p * 2.0)
# Array(16.989359, dtype=float32)

# The raw (unconstrained) value used by optimizers under the hood
p.raw_value
# Array(1.3862944, dtype=float32)

# We can also unwrap it explicitly
prx.unwrap(p)
# Array(8., dtype=float32)
```

## Example 2: PyTree Parameterizations

While the above approach caters for array constraints, it is sometimes useful to apply computations over an entire PyTree. To accomplish this, Parax uses *unwrapping*.

In the following example, we apply `jnp.exp` to a simple PyTree using `parax.Computed` and `parax.unwrap`.

```python
import jax.numpy as jnp
import parax as prx

# Define a PyTree using a dictionary
pytree = {'a': 1.0, 'b': {'x': 10.0, 'y': 20.0}}

# Wrap the PyTree in `parax.Computed`.
wrapped = prx.Computed(pytree, jnp.exp)

# Unwrap the Pytree, applying the computation
prx.unwrap(wrapped)
# {'a': Array(2.7182817, dtype=float32),
#  'b': {'x': Array(22026.465, dtype=float32), 
#        'y': Array(4.851652e+08, dtype=float32)}}
```

## Example 3: Optimizing an eqx.Model using Optimistix

In this example, we define a damped pendulum model using `equinox.Module` and optimize it using `optimistix`. The first parameter is initialized with a standard JAX array which we then fix. The second parameter is initialized with an unconstrained `prx.Param` with dummy metadata. The final parameter is given a default physical scale and constraint during model definition, which we then initialize using a simple float value later. Note that for bounded optimization, you can use the built-in wrapper at `parax.optimize.minimize_scipy`.

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optimistix as optx
import dataclasses
import parax as prx

class DampedPendulum(eqx.Module):
    # Non-default parameters
    k: prx.ParamLike
    friction: prx.ParamLike
    
    # Default physical parameters (creates a prx.Physical)
    length: prx.ParamLike = prx.physical(scale='mm', constraint=prx.Positive())

    def __call__(self, state):
        return self.k * state * self.friction / self.length
    
# Create our model and fix the multiplier
initial_model = DampedPendulum(
    k=jnp.array(1.0),
    friction=prx.Param(0.1, metadata={'hello': 'world'}),
    length=9.81,
)
initial_model = dataclasses.replace(initial_model, k=prx.Fixed(initial_model.k))

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

# Reconstruct the optimized model
final_model = prx.unwrap(eqx.combine(results.value, static))
final_model.friction # Array(2.4575772, dtype=float32)
final_model.length # Array(9.834487, dtype=float32)

# The optimizer found our ratio of 0.25
final_model.friction / final_model.length 
# Array(0.24989378, dtype=float32)
```