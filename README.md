![parax logo](assets/logo.png)

# Parax: Parametric modeling in JAX

**Parax**, is a declarative, parametric modelling library built on top of [JAX](https://github.com/jax-ml/jax) and [Equinox](https://github.com/patrick-kidger/equinox).

At its core, the library provides a `Parameter` class which can be set as fixed for training, as well as assigned arbitrary metadata. Core metadata includes assigning a name, description, scale, units, bounds, probability distribution, and bijector/transform.

The transform and scale metadata are particularly useful. The raw `value` inside a parameter is stored *untransformed* and *unscaled*. However, parameters can be used directly in mathematical expression as if they were JAX arrays, at which point these transforms are applied. This completely abstract the underlying *value* (to be used in optimization) from the user.

To make optimization easy, `Parax` comes with a built-in `parax.partition` function, which partitions a model into trainable parameters. If a model is built purely using `Parameter`'s, this removes the need for any conditional logical that would usually be done manually during `eqx.partition`.

Further, `Parax` also provides an extended version of Equinox's `Module` in `parax.Module`. This allows for parameter-aware module inspection and manipulation. For example, parameters can easily be flattened, updated using a single string assigned using the hierarchy, and mapped in batches.

The library is mainly intended for us in domain-specific scientific modeling, but can easily be applied to broader applications.

| **Parax** |  |
|-------------|-------|
| **Author**  | Gary Allen |
| **Homepage** | [github.com/parax/parax](https://github.com/parax/parax) |
| **Docs** | [gvcallen.github.io/parax](https://gvcallen.github.io/parax) |

## Installation
Parax can be installed using pip directly:

``
pip install parax
``
