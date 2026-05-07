# Parax

**Parax** is a library for parametric modeling in [JAX](https://github.com/jax-ml/jax). Features include:

- Parameters with metadata
- Computed PyTrees and callable parameterizations
- Derived, constrained, fixed, and random variables
- Arbitrary nesting of the above
- Abstract interfaces and associated tree manipulation tools

This makes Parax great for:

- Parameterizations for machine learning
- Bounded optimization for scientific modeling
- Probabilistic modeling and Bayesian inference
- Deep, nested PyTrees
- Combinations of the above

Note that Parax is *not a framework*, though it can be used to make one. Rather, it is focused on extendability and interoperability with other JAX libraries (especially [Equinox](https://github.com/patrick-kidger/equinox)).

## Installation
Parax can be installed using pip:

``
pip install parax
``

For some constraints and probabilistic features, you may need this `distreqx` branch:

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

You can also apply arbitrary computations to PyTrees and parameters using unwrapping:
```python
pytree = {'a': 1.0, 'b': {'x': 2.0, 'y': prx.Derived(jnp.log, 3.0)}}
wrapped = prx.Computed(jnp.exp, pytree)

prx.unwrap(wrapped)
# {'a': Array(2.7182817),
#  'b': {'x': Array(7.389056), 
#        'y': Array(3.0)}}
```
In the above example, `prx.Computed` operates on the whole PyTree's array-like nodes, while `prx.Derived` is an array-like `prx.AbstractVariable`.

## Next steps

Several tutorials are available in the documentation, for example:

- [Regular optimization](https://gvcallen.github.io/parax/examples/modeling_and_optimization/) (Optimistix)
- [Bounded optimization](https://gvcallen.github.io/parax/examples/bounded_optimization/) (JAXopt)
- [Bayesian inference](https://gvcallen.github.io/parax/examples/bayesian_inference/) (BlackJAX)

## Related

The library's design was inspired by several others that deserve mention, including [Flax](https://github.com/google/flax), [paramax](https://github.com/danielward27/paramax), and [PyTorch](https://github.com/pytorch/pytorch).