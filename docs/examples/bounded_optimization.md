# Bounded Optimization (JAXopt)

`parax.bounded` caters for easy extraction of bounds from PyTrees containing `parax.AbstractBounded` nodes (e.g. `parax.Constrained` variables). We simply have to project our model to the `base` constrained space before feeding it to the optimizer, and then update it afterwards using `parax.bounded.tree_update`.

```python
import jax.numpy as jnp
import equinox as eqx
import jaxopt
import parax as prx
import parax.bounded as prxb

class DummyModel(eqx.Module):
    # We can applying scaling (or any transformation)
    # and the bounds apply in the space we define them
    x: prx.ParamLike = prx.constrained(0.0, prx.Interval(-5.0, 5.0))
    y: prx.ParamLike = prx.physical(prx.Constrained(1.0, prx.Positive()), scale=1e-3)

    def __call__(self):
        return (self.x - 3.0)**2 + 1e6 * (self.y - 2.0e-3)**2

initial_model = DummyModel()

# Extract base values and bounds
initial_base = prxb.tree_base(initial_model)
lower_bounds, upper_bounds = prxb.tree_bounds(initial_model)

# Partition and define the objective
filter_spec = eqx.is_inexact_array
params, static = eqx.partition(initial_base, filter_spec, is_leaf=prx.is_constant)
lower, _ = eqx.partition(lower_bounds, filter_spec, is_leaf=prx.is_constant)
upper, _ = eqx.partition(upper_bounds, filter_spec, is_leaf=prx.is_constant)

def objective(p, static_structure):
    model = prx.unwrap(eqx.combine(p, static_structure))
    return model()

# Run the optimization
solver = jaxopt.ScipyBoundedMinimize(fun=objective)
results = solver.run(
    init_params=params, 
    bounds=(lower, upper), 
    static_structure=static
)

# Reconstruct the optimized model
opt_base = eqx.combine(results.params, static)
final_model = prxb.tree_update(initial_model, opt_base)

print(f"Optimized x: {jnp.array(final_model.x)}") # ~3.0
print(f"Optimized y: {jnp.array(final_model.y)}") # ~0.002
```