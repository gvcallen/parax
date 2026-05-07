# 0. Overview

In this advanced example, we optimize a correlated probabilistic model using Bayesian inference in the unit hypercube. Optimizing in the hypercube provides benefit when parameters are correlated, allowing the optimizer to more easily explore the space of possible values. We demonstrate several concepts:

- Modeling of correlated parameters using `parax.Random`.
- Using `parax.probabilistic` for optimization instead of inference.
- Using `icdf` and `cdf` functions as a tree hypercube transform.
