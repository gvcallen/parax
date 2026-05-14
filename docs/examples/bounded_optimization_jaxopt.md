# Overview

In this example, we optimize a bounded dummy model using `jaxopt`.

## 1. Defining the model

`parax.bounds` caters for easy extraction of bounds from PyTrees containing `parax.AbstractBounded` nodes (e.g. `parax.Constrained` variables). We simply have to unwrap any bounded leaves before feeding our model to the optimizer, and then optionally re-wrap the optimized result.

First we initialize a dummy model:

```python
import parax as prx
from parax.constraints import Positive, Interval
import equinox as eqx

class DummyModel(eqx.Module):
    x: prx.Param = prx.Constrained(Interval(-5.0, 5.0), 0.0)
    y: prx.Param = prx.Derived(lambda x: x*1e-3, prx.Constrained(Positive(), 1.0))

    def __call__(self):
        return (self.x - 3.0)**2 + 1e6 * (self.y - 2.0e-3)**2

model = DummyModel()
```

Note that we can nest parameters (like `y` above) and the constraints apply on the level we define them.

## 2. Setting up the loss

Next, we partition the model into dynamic and static halves, and then unwrap the bounded values and extract their associated bounds:

<!-- pytest-codeblocks:cont -->
```python
dynamic, static = eqx.partition(model, prx.bounds.is_dynamic, is_leaf=prx.bounds.is_leaf)

params = prx.unwrap(dynamic, only_if=prx.is_bounded)
lower, upper = prx.bounds.tree_bounds(dynamic)
```

Note the elegance of the above code. `prx.bounds.is_dynamic` and `prx.bounds.is_leaf` work together to remove any constant values and static data in a way that ensure the unwrapped `params` matches `lower` and `upper`, meaning we only need to perform a single partition.

Now we can define our objective:
<!-- pytest-codeblocks:cont -->
```python
def objective(p):
    unwrapped_model = prx.unwrap(eqx.combine(p, static, is_leaf=prx.bounds.is_leaf))
    return unwrapped_model()
```

Note that `is_leaf` is critical if our model contains non-unwrappable `parax.bounds.AbstractBounded` nodes.

## 3. Running the optimizer

Finally, we can run the optimizer and optionally re-wrap the results (passing `only_if=parax.is_bounded` and `is_leaf` again):

<!-- pytest-codeblocks:cont -->
```python
import jaxopt

solver = jaxopt.ScipyBoundedMinimize(fun=objective)
results = solver.run(
    init_params=params, 
    bounds=(lower, upper), 
)

opt_dynamic = prx.wrap(dynamic, results.params, only_if=prx.is_bounded)
opt_model = eqx.combine(opt_dynamic, static, is_leaf=prx.bounds.is_leaf)
```

Our parameters match the minimum of the dummy model:
<!-- pytest-codeblocks:cont -->
```python
opt_model.x.value
# ~3.0

opt_model.y.value
# ~0.002
```