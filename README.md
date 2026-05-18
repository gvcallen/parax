# Parax

**Parax** is a library for parametric modeling in [JAX](https://github.com/jax-ml/jax). Features include:

- Parameters with metadata
- PyTrees parameterization via unwrapping
- Built-in higher-level bijective constraints via `distreqx`
- Derived, constrained, fixed, and random array-like variables
- Abstract interfaces and associated tree manipulation tools

This makes Parax great for:

- Constraints for machine learning
- Bounded optimization for scientific modeling
- Probabilistic modeling and Bayesian inference
- Deep, nested PyTrees
- Combinations of the above

Note that Parax is *not a framework*, though it can be used to make one. Rather, it is focused on extensibility and interoperability with other JAX libraries (especially [Equinox](https://github.com/patrick-kidger/equinox)).

## Installation
Parax can be installed using pip:

``
pip install parax
``

For some built-in constraints and probabilistic features, you may need this `distreqx` branch:

``
pip install git+https://github.com/gvcallen/distreqx.git
``

## Documentation

Documentation is available [here](https://gvcallen.github.io/parax/).

## Quick example

Parax provides array-like variables that hold metadata and can be parameterized/constrained:

```python
import parax as prx
import jax.numpy as jnp

p1 = prx.Tagged(1.0, metadata={'hello', 'world'})
p2 = prx.Constrained(prx.constraints.Interval(0.0, 10.0), value=8.0)

p2.raw_value, p2.bounds
# Array(1.3862944), (Array(0.0), Array(10.0))

jnp.sin(p1) + (2 * p2)
# Array(16.84147)
```

You can also apply arbitrary computations to PyTrees and parameters using explicit unwrapping:
```python
pytree = {'a': 1.0, 'b': {'x': 2.0, 'y': prx.Derived(jnp.log, 3.0)}}
wrapped = prx.Apply(jnp.exp, pytree)

prx.unwrap(wrapped)
# {'a': Array(2.7182817),
#  'b': {'x': Array(7.389056), 
#        'y': Array(3.0)}}
```
In the above example, `prx.Apply` operates on the whole PyTree's array-like nodes, while `prx.Derived` is an array-like `prx.AbstractVariable`.

## Motivation

Usually, PyTrees are just "dumb" containers. However, it is often desirable to attach some metadata/parameterization to a specific node. This can be done by "unwrapping" the metadata or constraint during model preparation or computation.

Compared to other approaches, this provides a middle ground between purity and rigidity:
- The "purist" approach is using *shadow PyTrees* i.e. parallel trees that hold the relevant metadata/parameterization. However, these are tedious to define for nested models, and require the entire library to manage parallel structures.
- The "standard" approach is using properties and attributes i.e. defining the metadata/parameterization implicitly within the model. This is straight-forward, but tightly couples the extra state with the model, resulting in unnecessary fields and computations.

## Next steps

Several more involved examples are available in the [documentation](https://gvcallen.github.io/parax/), for example on bounded optimization and Bayesian sampling.

## Related

The library's design was inspired by several others that deserve mention, including [Flax](https://github.com/google/flax), [paramax](https://github.com/danielward27/paramax), and [PyTorch](https://github.com/pytorch/pytorch).