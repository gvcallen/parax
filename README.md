# Parax

**Parax** is a library for parametric modeling in [JAX](https://github.com/jax-ml/jax). Features include:

- Derived/constrained parameters with metadata
- Computed PyTrees and callable parameterizations
- Abstract interfaces for fixed, bounded, and probabilistic PyTrees
- Associated filtering and tree manipulation tools

Parax is *not a framework*, and is designed to be both extendable and interoperable with other JAX libraries (such as [Equinox](https://github.com/patrick-kidger/equinox)).

## Installation
Parax can be installed using pip:

``
pip install parax
``

For some constraints and probabilistic features, you may need this `distreqx` branch:

``
pip install git+https://github.com/gvcallen/distreqx.git
``

## Quick example

Parax provides array-like variables that hold metadata and can be parameterized/constrained:

```python
import parax as prx
import jax.numpy as jnp

p1 = prx.Param(1.0, metadata={'hello', 'world'})
p2 = prx.Constrained(8.0, prx.Interval(0.0, 10.0))

p2.raw_value, p2.bounds
# Array(1.3862944), (Array(0.0), Array(10.0))

jnp.sin(p1) + (2 * p2)
# Array(16.84147)
```

You can also apply arbitrary computations to PyTrees and parameters using unwrapping:
```python
pytree = {'a': 1.0, 'b': {'x': 2.0, 'y': prx.Derived(3.0, jnp.log)}}
wrapped = prx.Computed(pytree, jnp.exp)

prx.unwrap(wrapped)
# {'a': Array(2.7182817),
#  'b': {'x': Array(7.389056), 
#        'y': Array(3.0)}}
```
In the above example, `prx.Computed` operates on the whole PyTree, while `prx.Derived` is an array-like `prx.AbstractVariable`.

## Documentation

Documentation is available [here](https://gvcallen.github.io/parax/), with examples on unconstrained/bounded optimization and more.

## Related

The library's design was inspired by several others that deserve mention, including [Flax](https://github.com/google/flax), [paramax](https://github.com/danielward27/paramax), and [PyTorch](https://github.com/pytorch/pytorch).