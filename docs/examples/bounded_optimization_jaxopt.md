# Overview

In this example, we optimize a bounded dummy model using `jaxopt`.

## 1. Defining the model

`parax.bounded` caters for easy extraction of bounds from PyTrees containing `parax.AbstractBounded` nodes (e.g. `parax.Constrained` variables). We simply have to unwrap any bounded leaves before feeding our model to the optimizer, and then re-wrap the final optimized result.

First we initialize a dummy model:

```python
import parax as prx
from parax.constraints import Positive, Interval
from parax.transforms import Scale
import equinox as eqx

class DummyModel(eqx.Module):
    x: prx.Param = prx.constrained(Interval(-5.0, 5.0), 0.0)
    y: prx.Param = prx.derived(Scale(1e-3), prx.Constrained(Positive(), 1.0))

    def __call__(self):
        return (self.x - 3.0)**2 + 1e6 * (self.y - 2.0e-3)**2

initial_model = DummyModel()
```

Note that we can nest parameters (like `y` above) and the constraints apply on the level we define them.

## 2. Setting up the loss

Next, we extract the bounds and unwrap the bounded values, and partition/filter the bounded model and bounds into parameters and static metadata.

<!-- pytest-codeblocks:cont -->
```python
initial_bounded = prx.unwrap(initial_model, only_if=prx.is_bounded)
lower_all, upper_all = prx.bounded.tree_bounds(initial_model)

params, static = eqx.partition(initial_bounded, eqx.is_inexact_array, is_leaf=prx.is_constant)
lower = prx.remove(lower_all, prx.is_constant)
upper = prx.remove(upper_all, prx.is_constant)
```

Notice how we only unwrap bounded nodes by passing `only_if=parax.is_bounded` to `parax.unwrap`. This delays any unwrapping until a bounded node is encountered, resulting in a PyTree whose structure will naturally match that returned by `parax.bounded.tree_bounds`.

Notice also that we must use `prx.remove` (to remove any constant values) so that the shape of our bounds align with our parameters.

Now we can define our objective:
<!-- pytest-codeblocks:cont -->
```python
def objective(p):
    unwrapped_model = prx.unwrap(eqx.combine(p, static))
    return unwrapped_model()
```

## 3. Running the optimizer

Finally, we can run the optimizer and re-wrap the results (passing `only_if=parax.is_bounded` again):

<!-- pytest-codeblocks:cont -->
```python
import jaxopt

solver = jaxopt.ScipyBoundedMinimize(fun=objective)
results = solver.run(
    init_params=params, 
    bounds=(lower, upper), 
)

opt_bounded = eqx.combine(results.params, static)
final_model = prx.wrap(initial_model, opt_bounded, only_if=prx.is_bounded)
```

Our parameters match the minimum of the dummy model:
<!-- pytest-codeblocks:cont -->
```python
final_model.x.value
# ~3.0

final_model.y.value
# ~0.002
```