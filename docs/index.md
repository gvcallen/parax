# Parax: Parametric Modeling in JAX

**Parax** is a declarative, parametric modelling library built on top of [JAX](https://github.com/jax-ml/jax) and [Equinox](https://github.com/patrick-kidger/equinox).

At its core, the library provides a `Parameter` class which can be set as fixed for training, as well as assigned arbitrary metadata. Core metadata includes assigning a name, description, scale, bounds, probability distribution, and bijector (invertible transformation).

The transform and scale metadata are particularly useful. The raw `value` inside a parameter is stored in "latent" space i.e., *untransformed* and *unscaled*. However, parameters can be used directly in mathematical expressions as if they were JAX arrays, at which point the bijection and scaling are applied. This completely abstracts the underlying latent value (to be used in optimization) from the user, bypassing the need to explicitly apply the transform.

To make optimization easy, `Parax` comes with a built-in `parax.partition` function, which partitions a model into trainable parameters. If a model is built purely using `Parameter`s, this removes the need for any conditional logic that would usually be done manually during `eqx.partition`.

Further, `Parax` also provides an extended version of Equinox's `Module` in `parax.Module`. This allows for parameter-aware module inspection and manipulation. For example, parameters can easily be flattened, updated using a single string assigned using the hierarchy, and mapped in batches.

The library is mainly intended for use in domain-specific scientific modeling, but can easily be applied to broader applications.

| | |
|---|---|
| **Author** | Gary Allen |
| **Homepage** | [https://github.com/parax/parax](https://github.com/parax/parax) |
| **Docs** | [https://gvcallen.github.io/parax](https://gvcallen.github.io/parax) |

*(Note: MkDocs doesn't natively support the Sphinx `|release|` dynamic substitution out of the box without a plugin, so it has been omitted here. If you want dynamic versioning injected into your markdown, you can use the `mkdocs-macros-plugin` later.)*

## Installation

Parax can be installed directly using pip:

``
pip install parax
``