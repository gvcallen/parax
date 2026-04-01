![Parax Logo](https://raw.githubusercontent.com/gvcallen/parax/main/assets/logo.png)

**Parax** is a parametric modelling library built on top of [JAX](https://github.com/jax-ml/jax) and [Equinox](https://github.com/patrick-kidger/equinox).

At its core, the library provides a `Parameter` class which inherits from `eqx.Module` and wraps a JAX array. A parameter can be marked as `fixed` for training, as well as assigned arbitrary metadata.

However, the library also provides additional helpers, including `parax.partition`, `parax.Module` and `parax.Operator`. `partition` caters for easy partitioning of module's that contain `Parameter` objects. `Parameter` and `Module` together provide an experience similar to PyTorch with its `torch.nn.Parameter`, but in a more Equinox/JAX-friendly style. `Operator` caters for composable operations over arbitrary arguments (such as `Module`'s), allowing for easy PyTree feature extraction.

| **Parax** |  |
|-------------|-------|
| **Author**  | Gary Allen |
| **Homepage** | [github.com/parax/parax](https://github.com/parax/parax) |
| **Docs** | [gvcallen.github.io/parax](https://gvcallen.github.io/parax) |

## Features

- **Easy parameter fixing**: Parameters can be marked as `fixed` and the resultant modules partitioned using `parax.partition`.
- **Encapsulated constraints and scaling**: Optional scaling and transformations are abstracted away by applying them when the parameter object is cast to a JAX array. This can be used, for example, to enforce positivity or arbitrary constraints during optimization.
- **Parameter transforms**: Arbitrary transforms can be applied to parameters using `myparam.transformed(bij)`. This applies a transform to both the parameter and its underlying distribution (if any).
- **Arbitrary metadata support**: While **Parax** natively caters for common metadata such as distributions, bijectors, scaling, bounds and a name, arbitrary metadata can also be attached for more complex modelling purposes (for example, in the scientific domain it is common to want to attach units to a parameter).
- **Extended Equinox module**: **Parax** provides `parax.Module`, which extends `eqx.Module` to allow for easy inspection, updating, fixing, freeing, or mapping of parameters and their metadata deep within complex models using simple string paths and bulk `with_*` methods. For example, `parax.Module.named_params()` returns a dictionary of parameters with names based on string paths.
- (experimental) **Composable PyTree operations**: **Parax** provides `parax.Operator`, which caters for composable, parametric operations over arbitrary arguments. This can be very useful in manipulating domain-specific `parax.Module` objects in a parameter-aware manner.
- (experimental) **Model saving and loading**. By employing methods to serialize `distreqx` distributions and bijections, **Parax** provides (experimental) support to directly save (pickle) models using `parax.load` and `parax.save`, as long as they align to certain rules.

## Installation
Parax can be installed using pip directly:

``
pip install parax
``

## Overview

The `Parameter` class is designed to be used as if it was a JAX array. The raw `value` inside a parameter is therefore stored in "latent" space i.e. *untransformed* and *unscaled*. However, parameters eagerly cast to JAX arrays, at which point the bijection and scaling is applied. This completely abstracts the underlying latent value (to be used in optimization) from the user, bypassing the need to explicitly apply the transform.

To make optimization easy, `Parax` also comes with a built-in `parax.partition` function, which partitions a model into trainable parameters. If a model is built purely using `Parameter`'s, this removes the need for any conditional logical that would usually be done manually during `eqx.partition`.

Further, `Parax` also provides an extended version of Equinox's `Module` in `parax.Module`. This allows for parameter-aware module inspection and manipulation. For example, parameters can easily be flattened, updated using a single string assigned using the hierarchy, and mapped in batches.

The library is mainly intended for use in domain-specific scientific modeling, but can easily be applied to broader applications.

## Example 1: Enforcing bounds

The following example creates a `parax.Parameter` that is strictly bounded between 0.0 and 1.0, and whose physical value follows a normal distribution.

```python
import parax as prx
from distreqx.bijectors import Sigmoid

normal_param = prx.Normal(0.5, 0.1, bijector=Sigmoid())
print(normal_param.latent_value) # prints 0.0
print(normal_param.value) # prints 0.5
```

## Example 2: Optimizing a model

In this example, we define a simple quadratic model ($y = ax^2 + bx + c$). We fix the y-intercept, leave the other coefficients free, and use JAX and `optimistix` to fit the model to some noisy data.

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
