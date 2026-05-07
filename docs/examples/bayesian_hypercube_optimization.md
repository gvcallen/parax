# Overview

In this example, we perform Bayesian optimization (*maximum a posteriori estimation*) on a correlated probabilistic model within the unit hypercube. Optimizing in the hypercube provides numerous benefits when parameters are correlated (for example, it allows the optimizer to directly explore the prior in a bounded, regularized space and without custom bijectors).

This is an advanced example building on previous examples. It demonstrates several concepts at once:

- Modeling of correlated parameters and derived random variables using `parax.Random` and `parax.Derived`.
- Extracting both the joint and one-by-one distributions using `tree_joint` and `tree_distribution` in `parax.probabilistic`.
- Perform a hypercube transform and evaluating the joint log probability using `icdf`, `cdf` and `log_prob`.

## 1. Defining the model
Similar to the Bayesian sampling example, we define a linear regression model $y = w \cdot x + b$. However, instead of defining the weight and bias with individual priors, we model them using a joint covariance matrix. We use a derived variable via a Cholesky decomposition (as opposed to using a multivariate normal from `distreqx.distributions`) so that we can map independent standard `Normal` distribution to the unit hypercube using their native `cdf` and `icdf` functions.

```python
import parax as prx
import equinox as eqx
import jax
import jax.numpy as jnp
from distreqx.distributions import Normal, LogNormal

class CorrelatedBayesianModel(eqx.Module):
    theta: prx.Param
    sigma: prx.Param

    def __init__(self):
        loc = jnp.array([0.0, 0.0])
        cov = jnp.array([[5.0, 4.0], 
                         [4.0, 5.0]])
        chol_L = jax.scipy.linalg.cholesky(cov, lower=True)

        def to_correlated(z):
            return loc + chol_L @ z

        self.theta = prx.Derived(
            fn=to_correlated,
            raw_value=prx.Random(Normal(jnp.zeros(2), jnp.ones(2)))
        )

        self.sigma = prx.Random(LogNormal(0.0, 1.0))

    def __call__(self, x):
        weight, bias = self.theta
        return weight * x + bias

initial_model = CorrelatedBayesianModel()
```

## 2. Setting up the negative log posterior

To setup the log posterior, we unwrap the model into the bounded space, and then extract the distributions and initial hypercube values:

<!-- pytest-codeblocks:cont -->
```python
import parax.probabilistic as prxp

initial_bounded = prx.unwrap(initial_model, only_if=prx.is_probabilistic)
base_distributions = prxp.tree_distribution(initial_model)
initial_cube = jax.tree.map(lambda d, b: d.cdf(b), base_distributions, initial_bounded, is_leaf=prx.is_distribution)

def hypercube_transform(cube):
    eps = jnp.finfo(jnp.float32).eps
    safe_cube = jax.tree.map(lambda x: jnp.clip(x, eps, 1.0 - eps), cube)
    return jax.tree.map(lambda d, u: d.icdf(u), base_distributions, safe_cube, is_leaf=prx.is_distribution)
```

Note that, to extract the hypercube values, we used `parax.probabilistic.tree_distribution` as opposed to `parax.probabilistic.tree_joint`. This is because the former simply extracts the individual underlying distributions directly, as opposed to returning a single joint distribution for the entire model.

Now, we can easily transform the initial base model values to the hypercube using `jax.tree.map` and the underlying normal `cdf` function. We then partition these models using `eqx.partition` and extract lower and upper bounds for the optimizer:

<!-- pytest-codeblocks:cont -->

```python
params, static = eqx.partition(initial_cube, eqx.is_inexact_array, is_leaf=prx.is_constant)
zeros = jax.tree.map(jnp.zeros_like, params)
ones = jax.tree.map(jnp.ones_like, params)
```

Finally, we can extract the full joint distribution of the model and setup the negative log posterior to return the negative of the joint log prior plus the log likelihood.

<!-- pytest-codeblocks:cont -->
```python
base_joint = prxp.tree_joint(initial_model)
def negative_log_posterior(params, static, x_data, y_data):
    cube_model = eqx.combine(params, static)
    base_model = hypercube_transform(cube_model)
    log_prior = base_joint.log_prob(base_model)

    unwrapped_model = prx.unwrap(base_model)
    y_pred = jax.vmap(unwrapped_model)(x_data)
    log_likelihood = jnp.sum(Normal(y_pred, unwrapped_model.sigma).log_prob(y_data))
    return -(log_prior + log_likelihood)
```

## 3. Running the optimizer

Finally, we can run the optimizer. We generate some dummy data,

<!-- pytest-codeblocks:cont -->
```python
key = jax.random.key(0)
x_data = jnp.linspace(-3, 3, 100)

true_w = 2.5
true_b = -1.0
true_sigma = 0.5
y_data = true_w * x_data + true_b + true_sigma * jax.random.normal(key, x_data.shape)
```

and then run JAxopt:

<!-- pytest-codeblocks:cont -->
```python
import jaxopt

solver = jaxopt.ScipyBoundedMinimize(fun=negative_log_posterior)
results = solver.run(
    init_params=params, 
    bounds=(zeros, ones),
    static=static,
    x_data=x_data,
    y_data=y_data,
)

cube_model = eqx.combine(results.params, static)
bounded_model = hypercube_transform(cube_model)
map_model = prx.wrap(initial_model, bounded_model, only_if=prx.is_probabilistic)
```

If we print out the results, we see our maximum a posteriori estimate aligns with our simulated data:

<!-- pytest-codeblocks:cont -->
```python
print("--- Inline Correlated MAP Estimation ---")
print(f"True Weight: {true_w:<5} | MAP Weight: {map_model.theta.value[0]:.3f}")
# True: 2.5, MAP: 2.499

print(f"True Bias:   {true_b:<5} | MAP Bias:   {map_model.theta.value[1]:.3f}")
# True: -1.0, MAP: -0.942

print(f"True Sigma:  {true_sigma:<5} | MAP Sigma:  {map_model.sigma.value:.3f}")
# True: 0.5, MAP: 0.470
```