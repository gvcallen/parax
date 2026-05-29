# Overview

In this example, we optimize an exponential decay model while specifying constraints using `optimistix`.

## 1. Defining the model

First, we define a simple exponential decay model $y = A e^{-kt} + C$:

```python
import jax.numpy as jnp
import parax as prx
from parax.constraints import Positive

model = {
    'amplitude': prx.Constrained(Positive(), value=5.0),
    'rate': prx.Tagged(0.5, metadata={'desc': 'Decay constant'}),
    'baseline': prx.Fixed(jnp.array(1.2)),
}

def predict(model, t):
    return model['amplitude'] * jnp.exp(-model['rate'] * t) + model['baseline']
```

## 2. Setting up the loss

Although `prx.Fixed` does implement stopping gradients, we can also explicitly split our model into free and fixed parameters:

<!-- pytest-codeblocks:cont -->
```python
import jax

params = {k: v for k, v in model.items() if not prx.is_constant(v)}
fixed = {k: v for k, v in model.items() if prx.is_constant(v)}

def loss_fn(params, args):
    model = params | fixed

    t, y_true = args
    unwrapped_model = prx.unwrap(model)
    y_pred = jax.vmap(predict, in_axes=(None, 0))(unwrapped_model, t)
    return jnp.mean((y_pred - y_true)**2)
```

`parax.unwrap()` recursively resolves any derived variables/PyTrees from the bottom up.

## 3. Running the optimizer

Finally, we generate some dummy data with `amplitude=2.0` and `rate=1.0` and let `optimistix` find the underlying parameters.

<!-- pytest-codeblocks:cont -->
```python
import optimistix as optx

t_data = jnp.linspace(0, 5, 100)
y_data = 2.0 * jnp.exp(-1.0 * t_data) + 1.2

solver = optx.BFGS(rtol=1e-5, atol=1e-5)
results = optx.minimise(loss_fn, solver, y0=params, args=(t_data, y_data))
final_model = prx.unwrap(results.value | fixed)
```

Our optimized model matches our initial parameters:

<!-- pytest-codeblocks:cont -->
```python
final_model["amplitude"]
# 2.000002

final_model["rate"]
# 1.0000012
```