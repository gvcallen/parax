# 1. Defining the model

Lets define a simple exponential decay model: $y = A e^{-kt} + C$ using `equinox` and `parax`!

```python
import jax.numpy as jnp
import equinox as eqx
import parax as prx

class DecayCurve(eqx.Module):
    rate: prx.ParamLike
    baseline: prx.ParamLike
    amplitude: prx.ParamLike = prx.constrained(prx.Positive())

    def __call__(self, t):
        return self.amplitude * jnp.exp(-self.rate * t) + self.baseline

model = DecayCurve(
    amplitude=5.0, 
    rate=prx.Param(0.5, metadata={'desc': 'Decay constant'}), 
    baseline=1.2
)
```

Note that `5.0` is automatically converted to `prx.Constrained` by the dataclass field.

For demonstration purposes, we fix the baseline in this example:

<!-- pytest-codeblocks:cont -->
```python
import dataclasses
model = dataclasses.replace(model, baseline=prx.Fixed(model.baseline))
```

# 2. Setting up the loss

Optimization libraries like `optimistix` expect standard JAX arrays. We need to split our model into trainable parameters and static metadata, and then re-combine during our forward pass.

By passing `is_leaf=prx.is_constant` to `eqx.partition`, we can also separate out all `prx.Fixed` variables (and nested `prx.Frozen` models) into the static half of the tree.

<!-- pytest-codeblocks:cont -->
```python
import jax

params, static = eqx.partition(model, eqx.is_inexact_array, is_leaf=prx.is_constant)
def loss_fn(params, args):
    t, y_true = args
    current_model = prx.unwrap(eqx.combine(params, static))
    y_pred = jax.vmap(current_model)(t)
    return jnp.mean((y_pred - y_true)**2)
```

`parax.unwrap()` recursively resolves any derived variables/PyTrees from the bottom up.

# 3. Running the optimizer

Finally, we generate some dummy data with `amplitude=2.0` and `rate=1.0` and let `optimistix` find the underlying parameters.

<!-- pytest-codeblocks:cont -->
```python
import optimistix as optx

t_data = jnp.linspace(0, 5, 100)
y_data = 2.0 * jnp.exp(-1.0 * t_data) + 1.2

solver = optx.BFGS(rtol=1e-5, atol=1e-5)
results = optx.minimise(loss_fn, solver, y0=params, args=(t_data, y_data))
final_model = prx.unwrap(eqx.combine(results.value, static))
```

Our optimized model matches our initial parameters:

<!-- pytest-codeblocks:cont -->
```python
final_model.amplitude
# 2.000002

final_model.rate
# 1.0000012
```