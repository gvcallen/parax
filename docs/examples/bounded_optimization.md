# Bounded Optimization (JAXopt)

Parax's `parax.bounded` module caters for easy extraction of bounds from models containing `parax.AbstractBounded` PyTrees (such as `parax.Constrained` and `parax.Physical` variables), while also handling raw arrays alongside them. This is done projecting values to a *base* space using `parax.bounded.tree_base`, and then updating the original model after optimization using `parax.bounded.tree_update`.

Below is a minimal example demonstrating bounded optimization using `jaxopt.ScipyBoundedMinimize`. Note that, thanks to Parax's constraint system, the model is compatible with both unconstrained and bounded optimizers - it is up to use to decided if we want to optimize in *base space*, or in *unconstrained space* with unwrapping.

```python
import jax.numpy as jnp
import equinox as eqx
import jaxopt
import parax as prx
import parax.bounded as bounded

# 1. Define a trivial model with bounded parameters.
class SimpleModel(eqx.Module):
    # We can applying scaling (or any transformation)
    # and the bounds apply in the space we define them
    x: prx.ParamLike = prx.constrained(0.0, prx.Interval(-5.0, 5.0))
    y: prx.ParamLike = prx.physical(prx.Constrained(1.0, prx.Positive()), scale=1e-3)

    def __call__(self):
        # Dummy objective: minimize (x - 3e-3)^2 + 1e-6 * (y - 2)^2
        return (self.x - 3.0)**2 + 1e6 * (self.y - 2.0e-3)**2

initial_model = SimpleModel()

# 2. Extract base values and structural bounds for the optimizer
initial_base = bounded.tree_base(initial_model)
lower_bounds, upper_bounds = bounded.tree_bounds(initial_model)

# 3. Partition out static metadata
filter_spec = eqx.is_inexact_array
params, static = eqx.partition(initial_base, filter_spec, is_leaf=prx.is_constant)
lower, _ = eqx.partition(lower_bounds, filter_spec, is_leaf=prx.is_constant)
upper, _ = eqx.partition(upper_bounds, filter_spec, is_leaf=prx.is_constant)

# 4. Define the objective function for JAXopt
def objective(p, static_structure):
    # Recombine the base state
    current_base = eqx.combine(p, static_structure)
    model = prx.unwrap(current_base)
    return model()

# 5. Run the optimization
solver = jaxopt.ScipyBoundedMinimize(fun=objective)
results = solver.run(
    init_params=params, 
    bounds=(lower, upper), 
    static_structure=static
)

# 6. Reconstruct the final optimized model
opt_base = eqx.combine(results.params, static)
final_model = bounded.tree_update(initial_model, opt_base)

print(f"Optimized x: {final_model.x.value}") # ~3.0
print(f"Optimized y: {final_model.y.value}") # ~2.0
```