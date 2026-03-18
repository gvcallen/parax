[![Tests](https://github.com/parax/parax/actions/workflows/tests.yml/badge.svg)](https://github.com/parax/parax/actions/workflows/tests.yml)
[![Docs](https://github.com/parax/parax/actions/workflows/deploy_docs.yml/badge.svg)](https://github.com/parax/parax/actions/workflows/deploy_docs.yml)

![parax logo](assets/logo.png)

# Parax: Parametric Modelling in JAX and Equinox

**Parax**, or ``pmrf``, is an open-source radio frequency (RF) modelling framework. It provides an object-orientated means of modelling, fitting and sampling both circuit and numerical RF models in the frequency-domain, using [JAX](https://github.com/jax-ml/jax) and [Equinox](https://github.com/patrick-kidger/equinox).

| **Parax** |  |
|-------------|-------|
| **Author**  | Gary Allen |
| **Homepage** | [github.com/parax/parax](https://github.com/parax/parax) |
| **Docs** | [parax.github.io/parax](https://parax.github.io/parax) |
| **Paper** | [Parax: A JAX-Native Framework for Declarative Circuit Modelling](https://doi.org/10.48550/arXiv.2510.15881) |

## Installation
Parax can be installed using pip directly:

``
pip install parax
``

### Optional dependencies
Several additional dependencies are required/recommended for more advanced use-cases.

For PolyChord fitting:

``
pip install git+https://github.com/PolyChord/PolyChordLite.git anesthetic mpi4py
``

For BlackJAX fitting:

``
pip install git+https://github.com/handley-lab/blackjax@nested_sampling anesthetic
``

For eqx-learn surrogate modeling:

``
pip install git+https://github.com/eqx-learn/eqx-learn
``


## Citation

If you have used Parax for academic work, please cite the original [paper](https://doi.org/10.48550/arXiv.2510.15881):

> G.V.C. Allen, D.I.L. de Villiers, (2025). Parax: A JAX-native Framework for Declarative Circuit Modelling. arXiv, https://doi.org/10.48550/arXiv.2510.15881.

or with BibTeX:

```bibtex
@article{parax,
    doi = {10.48550/arXiv.2510.15881},
    url = {https://doi.org/10.48550/arXiv.2510.15881}, 
    year = {2025},
    month = {Oct},
    title = {Parax: A JAX-native Framework for Declarative Circuit Modelling}, 
    author = {Gary V. C. Allen and Dirk I. L. de Villiers},
    eprint = {2510.15881},
    archivePrefix = {arXiv},
    primaryClass = {cs.OH},
}
```

## Contributions

Since the library is in very early stages and likely contains several bugs, contributions are more than welcome by submitting a PR (especially for the docs!). However, before making any major changes, please open an issue so we can discuss the general intention of the contribution.
