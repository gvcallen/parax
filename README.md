![parax logo](assets/logo.png)

# Parax: Declarative, Parametric Modeling in JAX

**Parax**, is a parametric modelling library built on top of [JAX](https://github.com/jax-ml/jax) and [Equinox](https://github.com/patrick-kidger/equinox).

At its core, the library provides a `Parameter` class (derived from `eqx.Module`) which can be set as fixed (non-trainable) as well as assigned metadata. Example metadata includes a name, scale and distribution (via [numpyro](https://github.com/pyro-ppl/numpyro)). However, `Parax` also provides an extend version of Equinox's `Module` itself in `parax.Module`. This allows for module naming, automated hierarchical parameter access, parameter grouping and mapping, and more.

The library is mainly intended for use in domain-specific scientific modeling (see [ParamRF](https://github.com/gvcallen/paramrf) for an example), but can easily be applied to broader applications.

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
