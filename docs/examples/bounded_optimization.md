# Bounded Optimization (JAXopt)

`parax.bounded` caters for easy extraction of bounds from PyTrees containing `parax.AbstractBounded` nodes (e.g. `parax.Constrained` variables). We simply have to project our model to the `base` constrained space before feeding it to the optimizer, and then update it afterwards using `parax.bounded.tree_update`.

First we initialize a dummy model:

```python
import parax as prx
import equinox as eqx

class DummyModel(eqx.Module):
    x: prx.ParamLike = prx.constrained(0.0, prx.Interval(-5.0, 5.0))
    y: prx.ParamLike = prx.physical(prx.Constrained(1.0, prx.Positive()), scale=1e-3)

    def __call__(self):
        return (self.x - 3.0)**2 + 1e6 * (self.y - 2.0e-3)**2

initial_model = DummyModel()
```

Note that we can nest parameters (like `y` above) and the constraints apply on the level we define them.

Next, we extract the base values and bounds, and partition the model and bounds into parameters and static metadata.

<!-- pytest-codeblocks:cont -->

```python
import parax.bounded as prxb

initial_base = prxb.tree_base(initial_model)
lower_bounds, upper_bounds = prxb.tree_bounds(initial_model)

filter_spec = eqx.is_inexact_array
params, static = eqx.partition(initial_base, filter_spec, is_leaf=prx.is_constant)
lower, _ = eqx.partition(lower_bounds, filter_spec, is_leaf=prx.is_constant)
upper, _ = eqx.partition(upper_bounds, filter_spec, is_leaf=prx.is_constant)
```

Finally, we define our objective and run the optimizer:

<!-- pytest-codeblocks:cont -->
```python
import jaxopt

def objective(p):
    unwrapped_model = prx.unwrap(eqx.combine(p, static))
    return unwrapped_model()

solver = jaxopt.ScipyBoundedMinimize(fun=objective)
results = solver.run(
    init_params=params, 
    bounds=(lower, upper), 
)

opt_base = eqx.combine(results.params, static)
final_model = prxb.tree_update(initial_model, opt_base)
```

Our parameters match the minimum of the dummy model:
```python
final_model.x.value
# ~3.0

final_model.y.value
# ~0.002
```