# Overview

Parax was originally designed to be used as a lower-level library for other frameworks to be built on top of. For example, [ParamRF](https://github.com/gvcallen/paramrf) uses Parax as its backend. In this quick demonstration, we create a custom Parax variables that wraps another variable while adding a name, scale and metadata, to demonstrate how Parax can be used for this purpose.

## 1. Defining the Param class

Parax makes use of composable, nested wrappers for variable definitions. While we could make use of `parax.Tagged` and `parax.Transformed` variables to implement a name and scale in our framework, these wrappers are *wrapper*. Instead, this is an ideal use-case to implement a custom `parax.AbstractVariable` class, providing the users of our framework with domain-specific properties and added type safety, while inheriting all of Parax's unwrapping and metadata features.

Below we create a `Param` class that wraps another arbitrary variable, allowing all of Parax's built-in variables (e.g. `parax.Constrained`, `parax.Random` etc.) while providing easy access to a `.name` and `.scale` property:

```python
from typing import Any, Self

from equinox import field
import jax
import jax.numpy as jnp
import equinox as eqx
import parax as prx
from parax.annotation import AbstractAnnotated

class Param(prx.AbstractVariable, prx.AbstractWrappable[jax.Array], AbstractAnnotated[Any]):
    raw_value: prx.AbstractVariable
    scale: float = eqx.field(default=1.0, static=True)
    name: str | None = field(default=None, kw_only=True, static=True)
    metadata: Any = field(default=None, kw_only=True, static=True)

    @property
    def value(self) -> jax.Array:
        base_value = jnp.array(self.raw_value)
        if self.scale != 1.0:
            return base_value * self.scale
        return base_value

    def wrap(self, value: jax.Array) -> Self:
        new_raw_value = self.raw_value.wrap(value / self.scale)
        return eqx.tree_at(lambda x: x.raw_value, self, new_raw_value)
```

Note that we use `Equinox` to mark specific fields as static.

## 2. Creating a parameter

Now we can easily create a parameter instance:

<!-- pytest-codeblocks:cont -->
```python
from parax.constraints import Positive

my_param = Param(prx.Constrained(constraint=Positive(), value=2.0), scale=1e-3, name='my_param')

my_param.value
# Array(0.002, dtype=float32)

my_param.raw_value.raw_value
# Array(1.8545866, dtype=float32)

my_param.name
# 'my_param'
```

We can than perform tree mapping just as usually. For example, to extract the bounds:

<!-- pytest-codeblocks:cont -->
```python
bounds = prx.bounds.tree_bounds(my_param)

bounds[0].raw_value
# Array(0., dtype=float32)

bounds[1].raw_value
# Array(inf, dtype=float32)
```