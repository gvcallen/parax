# Overview

In this example, we perform Bayesian optimization (*maximum a posteriori estimation*) on a correlated probabilistic model within the unit hypercube. Optimizing in the hypercube provides numerous benefits when parameters are correlated (for example, it allows the optimizer to directly explore the prior in a bounded, regularized space and without custom bijectors).

This is an advanced example building on previous examples. It demonstrates several concepts at once:

- Modeling of correlated parameters and derived random variables using `parax.Random` and `parax.Derived`.
- Extracting both the joint and individual distribution(s) using `tree_joint_distribution` and `tree_distributions` in `parax.probabilistic`.
- Performing a hypercube transform and evaluating the joint log probability using `icdf`, `cdf` and `log_prob`.

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

To setup the objective negative log posterior, we first need to perform model *extraction* and then *filtering*.

First, we *partially unwrap* the model to resolve it into its constrained, probabilistic values, and we also extract its individual and joint distribution(s) which match the shape of these values:

<!-- pytest-codeblocks:cont -->
```python
init_constrained = prx.unwrap(initial_model, only_if=prx.is_probabilistic)
distributions_all = prx.probabilistic.tree_distributions(initial_model)
joint_all = prx.probabilistic.tree_joint_distribution(initial_model)
```

Next, we partition the model to remove static metadata and constant values, and also remove any constant nodes from the distributions (similar to other example):

<!-- pytest-codeblocks:cont -->
```python
init_params, static = eqx.partition(init_constrained, eqx.is_inexact_array, is_leaf=prx.is_constant)
distributions = prx.remove(distributions_all, prx.is_constant, stop_if=prx.is_distribution)
joint = prx.remove(joint_all, prx.is_constant, stop_if=prx.is_distribution)
```

Finally, we can transform our parameters into the hypercube using the `cdf`, and define the inverse hypercube transform using the `icdf`, as well as the negative log posterior which combines everything together:

<!-- pytest-codeblocks:cont -->
```python
init_cube_params = jax.tree.map(lambda d, b: d.cdf(b), distributions, init_params, is_leaf=prx.is_distribution)

def hypercube_transform(distributions, cube_params):
    eps = jnp.finfo(jnp.float32).eps
    safe_cube = jax.tree.map(lambda x: jnp.clip(x, eps, 1.0 - eps), cube_params)
    return jax.tree.map(lambda d, u: d.icdf(u), distributions, safe_cube, is_leaf=prx.is_distribution)

def negative_log_posterior(cube_params, distributions, static, joint, x_data, y_data):
    params = hypercube_transform(distributions, cube_params)
    unwrapped = prx.unwrap(eqx.combine(params, static))
    
    log_prior = joint.log_prob(params)
    y_pred = jax.vmap(unwrapped)(x_data)
    log_likelihood = jnp.sum(Normal(y_pred, unwrapped.sigma).log_prob(y_data))
    return -(log_prior + log_likelihood)
```

## 3. Running the optimizer

We can now run the optimizer! First, we generate some dummy data:

<!-- pytest-codeblocks:cont -->
```python
key = jax.random.key(0)
x_data = jnp.linspace(-3, 3, 100)

true_w = 2.5
true_b = -1.0
true_sigma = 0.5
y_data = true_w * x_data + true_b + true_sigma * jax.random.normal(key, x_data.shape)
```

and then define our hypercube bounds and run JAXopt:

<!-- pytest-codeblocks:cont -->
```python
import jaxopt

zeros = jax.tree.map(jnp.zeros_like, init_cube_params)
ones = jax.tree.map(jnp.ones_like, init_cube_params)

solver = jaxopt.ScipyBoundedMinimize(fun=negative_log_posterior)
results = solver.run(
    init_params=init_cube_params,
    bounds=(zeros, ones),
    distributions=distributions,
    static=static,
    joint=joint,
    x_data=x_data,
    y_data=y_data,
)
```

Finally, we can map the optimized hypercube parameters back to our model:
<!-- pytest-codeblocks:cont -->
```python
opt_cube_params = results.params
opt_params = hypercube_transform(distributions, opt_cube_params)
opt_values = eqx.combine(opt_params, static)
opt_model = prx.wrap(initial_model, opt_values, only_if=prx.is_probabilistic)
```

If we print out the results, we see our maximum a posteriori estimate aligns with our simulated data:

<!-- pytest-codeblocks:cont -->
```python
print("--- Inline Correlated MAP Estimation ---")
print(f"True Weight: {true_w:<5} | MAP Weight: {opt_model.theta.value[0]:.3f}")
# True: 2.5, MAP: 2.499

print(f"True Bias:   {true_b:<5} | MAP Bias:   {opt_model.theta.value[1]:.3f}")
# True: -1.0, MAP: -0.942

print(f"True Sigma:  {true_sigma:<5} | MAP Sigma:  {opt_model.sigma.value:.3f}")
# True: 0.5, MAP: 0.470
```