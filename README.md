[![Tests](https://github.com/parax/parax/actions/workflows/tests.yml/badge.svg)](https://github.com/parax/parax/actions/workflows/tests.yml)
[![Docs](https://github.com/parax/parax/actions/workflows/deploy_docs.yml/badge.svg)](https://github.com/parax/parax/actions/workflows/deploy_docs.yml)

![parax logo](assets/logo.png)

# Parax: Parametric Modeling in JAX

**Parax**, is a parametric modelling library built on top of [JAX](https://github.com/jax-ml/jax) and [Equinox](https://github.com/patrick-kidger/equinox).

At its core, the library provides a `Parameter` class which can be set as fixed and assigned metadata, such as a probability distribution, name scale etc. However, `Parax` also provides an extend version of Equinox's `Module` in `parax.Module`, to allow for module naming, functional inspection and updates of parameters, and more.

The library is mainly intended for us in domain-specific scientific modeling, but can easily be applied to broader applications.

| **Parax** |  |
|-------------|-------|
| **Author**  | Gary Allen |
| **Homepage** | [github.com/parax/parax](https://github.com/parax/parax) |
| **Docs** | [parax.github.io/parax](https://parax.github.io/parax) |

## Installation
Parax can be installed using pip directly:

``
pip install parax
``
