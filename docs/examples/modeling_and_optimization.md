# 1. Defining the model

Lets define a simple exponential decay model: $y = A e^{-kt} + C$ using `equinox` and `parax`!

```python
import jax.numpy as jnp
import equinox as eqx
import parax as prx
import dataclasses

# Define the model
class DecayCurve(eqx.Module):
    rate: prx.ParamLike
    baseline: prx.ParamLike
    
    # Enforce a strictly positive constraint on the amplitude.
    amplitude: prx.ParamLike = prx.constrained(prx.Positive())

    def __call__(self, t):
        # Math operations instantly unwrap Parax variables to JAX arrays
        return self.amplitude * jnp.exp(-self.rate * t) + self.baseline

# Initialize the model using floats/parameters.
# 5.0 is automatically converted to prx.Constrained by the dataclass field.
model = DecayCurve(
    amplitude=5.0, 
    rate=prx.Param(0.5, metadata={'desc': 'Decay constant'}), 
    baseline=1.2
)

# Fix the baseline so it is ignored during optimization
model = dataclasses.replace(model, baseline=prx.Fixed(model.baseline))
```

# 2. Setting up the loss

Optimization libraries like `optimistix` expect standard JAX arrays. We need to split our model into trainable parameters and static metadata.

By passing `is_leaf=prx.is_constant` to Equinox, all `prx.Fixed` variables (and nested `prx.Frozen` models) are grouped in the static half of the tree.

<!-- pytest-codeblocks:cont -->
```python
import jax

# Partition the model
params, static = eqx.partition(model, eqx.is_inexact_array, is_leaf=prx.is_constant)

def loss_fn(params, args):
    t, y_true = args
    
    # 1. Recombine the tree
    # 2. unwrap() recursively resolves Parax variables and constraints into standard JAX arrays
    current_model = prx.unwrap(eqx.combine(params, static))
    
    y_pred = jax.vmap(current_model)(t)
    return jnp.mean((y_pred - y_true)**2)
```

# 3. Running the optimizer

Next, we generate some dummy data and let `optimistix` find the underlying parameters.

<!-- pytest-codeblocks:cont -->
```python
import optimistix as optx

# Generate dummy data (amplitude=2.0, rate=1.0)
t_data = jnp.linspace(0, 5, 100)
y_data = 2.0 * jnp.exp(-1.0 * t_data) + 1.2

# Run the optimization
solver = optx.BFGS(rtol=1e-5, atol=1e-5)
results = optx.minimise(loss_fn, solver, y0=params, args=(t_data, y_data))

# Reconstruct the optimized model
final_model = prx.unwrap(eqx.combine(results.value, static))

print(final_model.amplitude) 
# 2.000002

print(final_model.rate) 
# 1.0000012
```