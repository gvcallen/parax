![Parax Logo](https://raw.githubusercontent.com/gvcallen/parax/main/assets/logo.png)

**Parax** is a mini-framework designed for parametric/scientific modeling in [JAX](https://github.com/jax-ml/jax) and [Equinox](https://github.com/patrick-kidger/equinox).

It provides `parax.Parameter` with common (and custom) metadata, as well as useful tools and wrappers for model unwrapping, inspection, manipulation, optimization, and inference.

| **Parax** |  |
|-------------|-------|
| **Author**  | Gary Allen |
| **Homepage** | [github.com/parax/parax](https://github.com/parax/parax) |
| **Docs** | [gvcallen.github.io/parax](https://gvcallen.github.io/parax) |

## Features

- **parax.Parameter**: Equinox module common physical metadata such as `fixed`, `scale`, `constraint` and `distribution` (via [distreqx](https://github.com/lockwo/distreqx)).
- **Unit support**: Support for units in `scale` via [unxt](https://github.com/GalacticDynamics/unxt).
- **Optimization and inference wrappers**: Out-of-the-box support for optimization ((via [optimistix](https://github.com/patrick-kidger/optimistix), [optax](https://github.com/google-deepmind/optax) and `scipy.optimize.minimize`)) and Bayesian inference (via [BlackJAX](https://github.com/blackjax-devs/blackjax)).
- **Lower-level tree utilities**: Built-in filters and tree-mapping utilities, such as `where_free_raw_value`, `is_free_param`, `tree.raw_value`, `tree.constraint`, `tree.distribution` etc.

- **Model manipulation**: Provides `prx.tree_at`, catering for surgical, parameter-driven model manipulation.
- (experimental) **Composable PyTree operations**: Composable operations over arbitrary PyTree arguments in `parax.Operator`.
- (experimental) **Model saving and loading**. Direct model saving (pickling) via `parax.load` and `parax.save`.

## Installation
Parax can be installed using pip directly:

``
pip install parax
``

## Overview

The `Parameter` class is designed to be used as if it was a JAX array. The raw `value` inside a parameter is therefore stored in "latent" space i.e. *untransformed* and *unscaled*. However, parameters eagerly cast to JAX arrays, at which point any transformation/bijection and scaling is applied. This completely abstracts the underlying latent value (to be used in optimization) from the user, bypassing the need to explicitly apply the transform.

To make optimization easy, `Parax` also comes with a built-in `parax.partition` function, which partitions a model into trainable parameters. If a model is built purely using `Parameter`'s, this removes the need for any conditional logical that would usually be done manually during `eqx.partition`.

Further, `Parax` also provides an extended version of Equinox's `Module` in `parax.Module`. This allows for parameter-aware module inspection and manipulation. For example, parameters can easily be flattened, updated using a single string assigned using the hierarchy, and mapped in batches.

The library is mainly intended for use in domain-specific scientific modeling, but can easily be applied to broader applications.

## Example 1: Enforcing bounds

The following example creates a `parax.Parameter` that is strictly bounded between 0.0 and 1.0, and whose physical value follows a normal distribution.

```python
import jax.numpy as jnp
import parax as prx
from distreqx.bijectors import Sigmoid

normal_param = prx.Normal(0.5, 0.1, transform=Sigmoid())
assert normal_param.latent_value == 0.0
assert normal_param.value == 0.5
assert jnp.array(normal_param) == 0.5
```

## Example 2: Manipulating parameters

`parax.Module` is designed as a lightweight add-on to `equinox.Module` with added parameter manipulation, mapping and inspection routines.

The following example defines a nested module, and then fixes one of the parameters, and adds a probability distribution to another.

Notice how parameters can be initialized with floats or default parameters - `parax.Module` automatically applies the `as_param` converter, and also deep-copies any mutatable parameter objects to avoid the Python "mutable default" trap.

```python
import jax.numpy as jnp
import parax as prx

class Quadratic(prx.Module):
    a: prx.Parameter = 0.0
    b: prx.Parameter = 0.0
    c: prx.Parameter = prx.Fixed(0.0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.a * (x ** 2) + self.b * x + self.c

class NthRootOfModule(prx.Module):
    # Create a module that takes the n'th root of another module
    nth_root: prx.Parameter

    # Define the module we are wrapping and mark its parameters
    # as "transparent" in the hierachy (so we dont have an "of_" prefix)
    of: prx.Module = prx.field(transparent=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.power(self.of(x), 1.0/self.n)

# Create a module and print its parameter names
cube_root_quadratic = NthRootOfModule(3, Quadratic(2.0, -3.0))
print(cube_root_quadratic.param_names()) # prints ['nth_root', 'a', 'b']

# Updated 'a' to have a normal distribution and 'nth_root' to be fixed to 2.5.
# prx.Normal is a factory that creates a `distreqx.distribution.Normal`
statistical_2p5_root_quadratic = cube_root_quadratic.with_params(
    nth_root=prx.Fixed(2.5),
    a=prx.Normal(3.0, 1.0),
)
```

## Example 3: Optimizing a model

In this example, we define a simple quadratic model ($y = ax^2 + bx + c$) and derive directly from `eqx.Module`. We fix the y-intercept, leave the other coefficients free, partition using `prx.partition`, and use `optimistix` to fit the model to some noisy data.

```python

import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx

import parax as prx
from parax.parameters import Free, Fixed

# 1. Define the Parametric Model
class Quadratic(eqx.Module):
    """A generic quadratic curve: y = a*x^2 + b*x + c"""
    
    a: prx.Parameter
    b: prx.Parameter
    c: prx.Parameter

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.a * (x ** 2) + self.b * x + self.c
    
# We pass in free/fixed parameters without metadata using factories.
# Note that `parax.Module` would allow us to simply pass `a=1.5` for free parameters.
model = Quadratic(a=Free(1.5), b=Free(0.5), c=Fixed(10.0))

# 2. Generate some dummy "ground truth" data with noise
x_true = jnp.linspace(-5.0, 5.0, 100)
y_true = 3.0 * (x_true ** 2) - 2.0 * x_true + 10.0 # True a=3.0, b=-2.0
y_true = y_true + jax.random.normal(jax.random.key(0), x_true.shape)

# 3. Partition the model into free and fixed parameters
params, static = prx.partition(model)

# 4. Define the loss Function
def loss_fn(params, args=None):
    model = eqx.combine(params, static)
    y_pred = model(x_true)
    return jnp.mean((y_pred - y_true)**2)

# 5. Run the BFGS optimizer
solver = optx.LBFGS(rtol=1e-6, atol=1e-6)
solution = optx.minimise(
    fn=loss_fn,
    y0=params,
    solver=solver,
    args=(x_true, y_true, static),
)

# 6. Recombine to get the final fitted model
fitted_model = eqx.combine(solution.value, static)

print(f"Fitted 'a': {jnp.array(fitted_model.a):.8f} (Expected ~3.0)")
print(f"Fitted 'b': {jnp.array(fitted_model.b):.8f} (Expected ~-2.0)")
print(f"Fixed 'c':  {jnp.array(fitted_model.c):.8f} (Remained 10.0)")
print(f'Final loss: {loss_fn(fitted_model)}')
print(solution.result)
```
