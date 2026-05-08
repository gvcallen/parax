# Overview

This example demonstrates Bayesian sampling of a linear regression model with independent priors using `blackjax`.

## 1. Defining the model

Let's define a simple linear regression model: $y = w \cdot x + b$. Instead of regular parameters, we will assign probability distributions to our variables using `prx.Random` variables to establish our priors.

We'll define the model class, assign a normal and uniform prior to our weight and bias respectively, and initialize the model with our initial guesses.

```python
import equinox as eqx
import parax as prx
from distreqx.distributions import Uniform, Normal

class BayesianLinearModel(eqx.Module):
    weight: prx.Param = prx.random(Normal(0.0, 5.0))
    bias: prx.Param = prx.random(Uniform(0.0, 2.0))

    def __call__(self, x):
        return self.weight * x + self.bias

initial_model = BayesianLinearModel(weight=1.0, bias=1.0)
```

## 2. Setting up the log posterior

In this tutorial, we will use `blackjax` for Bayesian sampling. We need to provide a function that takes our model parameters in an unconstrained space and returns an unnormalized log-posterior.

First, we use `parax.probabilistic` to extract the initial model values in the probability space (where the distributions are defined), as well as our joint prior. We then partition the values into parameters and static metadata.

<!-- pytest-codeblocks:cont -->
```python
unconstrained_prior = prx.probabilistic.tree_unconstrained_distribution(initial_model)
bijector_to_constrained = prx.probabilistic.tree_leafwise_bijector(initial_model)

initial_constrained = prx.unwrap(initial_model, only_if=prx.is_probabilistic)
initial_unconstrained = bijector_to_constrained.inverse(initial_constrained)

params, static = eqx.partition(initial_unconstrained, eqx.is_inexact_array, is_leaf=prx.is_constant)
```

Next, we define the log posterior. We assume Gaussian noise with a standard deviation of `1.0`.

Note how we do all probabilistic calculations in the probability space, and only unwrap the model for the forward pass.
<!-- pytest-codeblocks:cont -->
```python
import jax
import jax.numpy as jnp

def log_posterior_fn(params, static, x_data, y_true):
    unconstrained = eqx.combine(params, static)
    
    log_prior = unconstrained_prior.log_prob(unconstrained)
    constrained = bijector_to_constrained.forward(unconstrained)
    
    unwrapped = prx.unwrap(constrained)
    y_pred = jax.vmap(unwrapped)(x_data)
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

logprob = lambda p: log_posterior_fn(p, static, x_data, y_data)
inv_mass_matrix = jnp.ones_like(ravel_pytree(params)[0])
nuts = blackjax.nuts(logprob, step_size=1e-2, inverse_mass_matrix=inv_mass_matrix)

initial_state = nuts.init(params)

@eqx.filter_jit
def run_mcmc(key, state, num_steps):
    def step_fn(state, rng_key):
        state, _ = nuts.step(rng_key, state)
        return state, state.position

    keys = jr.split(key, num_steps)
    _, samples = jax.lax.scan(step_fn, state, keys)
    return samples

sample_key = jr.key(0)
param_samples = run_mcmc(sample_key, initial_state, num_steps=2000)
```

## 4. Evaluating the results

Because `blackjax` preserves the PyTree structure of our inputs, `mcmc_samples` has the exact same structure as our partitioned params tree. We can use `eqx.filter_jit` along with `parax.wrap` to reconstruct the model and easily plot parameters and functional posteriors!

We'll discard the first 500 steps as warmup/burn-in. Then, we create the combined model and update it. Although we technically didn't use any parax "unwrappables" on top of our probability space, it is always best practice to unwrap before evaluation.

Finally, we generate predictions across the input space and plot the results.

<!-- pytest-codeblocks:cont -->
```python
import matplotlib.pyplot as plt

clean_param_samples = jax.tree.map(lambda x: x[500:], param_samples)
unconstrained_samples = eqx.combine(clean_param_samples, static)
constrained_samples = bijector_to_constrained.forward(unconstrained_samples)

model_samples = eqx.filter_vmap(eqx.Partial(prx.wrap, only_if=prx.is_probabilistic), in_axes=(None, eqx.if_array(0)))(initial_model, constrained_samples)
unwrapped_models = prx.unwrap(model_samples)

x_plot = jnp.linspace(-3, 3, 100)
y_preds = eqx.filter_vmap(unwrapped_models)(x_plot).T
y_mean = jnp.mean(y_preds, axis=0)
y_lower, y_upper = jnp.percentile(y_preds, jnp.array([2.5, 97.5]), axis=0)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(unwrapped_models.weight, bins=30, density=True, color='steelblue', edgecolor='black')
axes[0].axvline(2.5, color='red', linestyle='dashed', linewidth=2, label='True Value')
axes[0].set_title('Weight Posterior')
axes[0].legend()

axes[1].hist(unwrapped_models.bias, bins=30, density=True, color='seagreen', edgecolor='black')
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
- *Compatibility with optimization*. It is common to want to swap between optimization and Bayesian sampling. Using Parax, we could easily define a factory that wraps a `parax.Random` variable in a `parax.Constrained`, allowing us to toggle between bounded optimation and inference.