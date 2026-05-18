# Overview

This example demonstrates Bayesian sampling of a linear regression model with independent priors using `blackjax`.

## 1. Defining the model

Let's define a simple linear regression model: $y = w \cdot x + b$. Instead of regular parameters, we will assign probability distributions to our variables using `prx.Random` variables to establish our priors.

We'll define the model class and assign a normal and uniform prior to our weight and bias respectively.

```python
import parax as prx
import jax.numpy as jnp
from distreqx.distributions import Uniform, Normal

model = {
    'weight': prx.Random(Normal(0.0, 5.0)),
    'bias': prx.Random(Uniform(0.0, 2.0))
}

def predict(params: dict, x: jnp.ndarray):
    return params['weight'] * x + params['bias']
```

## 2. Setting up the log posterior

In this tutorial, we will use `blackjax` for Bayesian sampling. Since we will be using an unconstrained MCMC sampler, we need to provide a function that takes our model parameters in an unconstrained space and returns an unnormalized log-posterior. To accomplish this, we will explicitly use *bijectors*.

<!-- pytest-codeblocks:cont -->
```python
params = prx.unwrap(model, only_if=prx.is_probabilistic)
unconstrained_prior = prx.probability.tree_unconstrained_distribution(model)
bijector_to_constrained = prx.constraints.tree_leafwise_bijector(model)
```

Note that we need to use the log prior that corresponds to the *unconstrained space*, since it must accurately represent the geometry explored by the sampler.

Finally, we project our constrained parameters to the unconstrained space, and define the log posterior to be Gaussian likelihood with a standard deviation of `1.0`.
<!-- pytest-codeblocks:cont -->
```python
import jax

unconstrained_params = bijector_to_constrained.inverse(params)

def log_posterior_fn(unconstrained_params, bijector_to_constrained, unconstrained_prior, x_data, y_true):
    params = bijector_to_constrained.forward(unconstrained_params)
    
    log_prior = unconstrained_prior.log_prob(unconstrained_params)
    y_pred = jax.vmap(predict, in_axes=(None, 0))(params, x_data)
    log_likelihood = jnp.sum(Normal(y_pred, 1.0).log_prob(y_true))

    return log_prior + log_likelihood
```

## 3. Running the sampler

Now we generate some noisy dummy data (with a true weight of `2.5` and a true bias of `1.0`) and set up our MCMC sampler. We'll use the No-U-Turn Sampler (NUTS) without window adaptation, initialize the state, define the sampling loop using `jax.lax.scan`, and finally run the sampler.

<!-- pytest-codeblocks:cont -->
```python
import blackjax
import jax.random as jr
from jax.flatten_util import ravel_pytree

rng_key = jr.key(42)
x_data = jnp.linspace(-2, 2, 50)
y_data = 2.5 * x_data + 1.0 + jr.normal(rng_key, (50,))

logprob = lambda p: log_posterior_fn(p, bijector_to_constrained, unconstrained_prior, x_data, y_data)
inv_mass_matrix = jnp.ones_like(ravel_pytree(unconstrained_params)[0])
nuts = blackjax.nuts(logprob, step_size=1e-2, inverse_mass_matrix=inv_mass_matrix)

init_state = nuts.init(unconstrained_params)

def run_mcmc(key, state, num_steps):
    @jax.jit
    def step_fn(state, rng_key):
        state, _ = nuts.step(rng_key, state)
        return state, state.position

    keys = jr.split(key, num_steps)
    _, samples = jax.lax.scan(step_fn, state, keys)
    return samples

sample_key = jr.key(0)
unconstrained_param_samples = run_mcmc(sample_key, init_state, num_steps=2000)
```

## 4. Evaluating the results

Because `blackjax` preserves the PyTree structure of our inputs, `mcmc_samples` has the exact same structure as our params tree. Using this, we can easily our plot parameters and functional posteriors!

We'll discard the first 500 steps as warmup/burn-in.

Finally, we generate predictions across the input space and plot the results.

<!-- pytest-codeblocks:cont -->
```python
import matplotlib.pyplot as plt

clean_unconstrained_param_samples = jax.tree.map(lambda x: x[500:], unconstrained_param_samples)
constrained_param_samples = bijector_to_constrained.forward(clean_unconstrained_param_samples)

x_plot = jnp.linspace(-3, 3, 100)
y_preds = jax.vmap(predict, in_axes=(None, 0))(constrained_param_samples, x_plot).T
y_mean = jnp.mean(y_preds, axis=0)
y_lower, y_upper = jnp.percentile(y_preds, jnp.array([2.5, 97.5]), axis=0)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(constrained_param_samples['weight'], bins=30, density=True, color='steelblue', edgecolor='black')
axes[0].axvline(2.5, color='red', linestyle='dashed', linewidth=2, label='True Value')
axes[0].set_title('Weight Posterior')
axes[0].legend()

axes[1].hist(constrained_param_samples['bias'], bins=30, density=True, color='seagreen', edgecolor='black')
axes[1].axvline(1.0, color='red', linestyle='dashed', linewidth=2, label='True Value')
axes[1].set_title('Bias Posterior')
axes[1].legend()

axes[2].scatter(x_data, y_data, color='black', s=15, label='Data', zorder=3)
axes[2].plot(x_plot, y_mean, color='darkorange', linewidth=2, label='Mean Prediction')
axes[2].fill_between(x_plot, y_lower, y_upper, color='orange', alpha=0.3, label='95% HDI')
axes[2].plot(x_plot, 2.5 * x_plot + 1.0, color='red', linestyle='dashed', label='True Function')
axes[2].set_title('Functional Posterior')
axes[2].legend()

plt.tight_layout()
plt.show()
```

## 5. Advantages of Parax

You may have noticed that we could have accomplished the above without the added abstraction of `parax.Random` variable wrappers (i.e. by defining our models using `distreqx` distributions and using tree mapping directly). However, building models using Parax variables (`parax.AbstractVariables`) has a number of quality-of-life benefits:

- *Parameters as first-class citizens*. Models contain parameters - not distributions. By prioritizing a parameter-centered approach and using tree tools to setup our model, we maintain a clear separation of concerns.
- *Variable manipulation*. For example, you can't "fix a distribution" after-the-fact without complex filtering, but you can easily wrap a `parax.Random` variable in a `parax.Fixed` variable.
- *Compatibility with optimization*. It is common to want to swap between optimization and Bayesian sampling. Because `parax.Random` implement `parax.bounds.AbstractBounded`, we can easily optimize within the support of the distribution.