# 1. Defining the model

Let's define a simple linear regression model: $y = w \cdot x + b$. Instead of regular parameters, we will assign probability distributions to our variables using `prx.Random` variables to establish our priors.

We'll define the model class, assign normal priors to our weight and bias, and initialize the model with our initial guesses.

```python
import equinox as eqx
import parax as prx
from distreqx.distributions import Normal

class BayesianLinearModel(eqx.Module):
    weight: prx.ParamLike = prx.random(Normal(0.0, 5.0))
    bias: prx.ParamLike = prx.random(Normal(0.0, 2.0))

    def __call__(self, x):
        return self.weight * x + self.bias

initial_model = BayesianLinearModel(weight=1.0, bias=0.0)
```

# 2. Setting up the log posterior
In this tutorial, we will use `blackjax` for Bayesian sampling. We need to provide a function that takes our model parameters in an unconstrained space and returns an unnormalized log-posterior.

First, we use `parax.probabilistic` to extract the initial model values in the "base" probability space (where the distributions are defined), as well as our joint prior.  We then partition the base model and prior values into parameters and static metadata.

<!-- pytest-codeblocks:cont -->
```python
import jax
import jax.numpy as jnp
import parax.probabilistic as prxp

initial_base = prxp.tree_base(initial_model)
prior_dist = prxp.tree_joint(initial_model)

filter_spec = eqx.is_inexact_array
params, static = eqx.partition(initial_base, filter_spec, is_leaf=prx.is_constant)
prior, _ = eqx.partition(prior_dist, filter_spec, is_leaf=prx.is_constant)
```

Next, we define the log posterior. We assume Gaussian noise with a standard deviation of `1.0`.
<!-- pytest-codeblocks:cont -->
```python
def log_posterior_fn(p, static, x_data, y_true):
    unwrapped_model = prx.unwrap(eqx.combine(p, static))
    log_prior = prior_dist.log_prob(unwrapped_model)
    
    y_pred = jax.vmap(unwrapped_model)(x_data)
    log_likelihood = jnp.sum(Normal(y_pred, 1.0).log_prob(y_true))
    
    return log_prior + log_likelihood
```

# 3. Running the sampler

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
base_samples = run_mcmc(sample_key, initial_state, num_steps=2000)
```

# 4. Evaluating the results
Because `blackjax` preserves the PyTree structure of our inputs, `base_samples` has the exact same structure as our partitioned params tree. We can use `eqx.filter_jit` along with `parax.probabilistic.tree_update` to reconstruct the model and easily plot parameters and functional posteriors!

We'll discard the first 500 steps as warmup/burn-in. Then, we create the combined model and update it. Although we technically didn't use any parax "unwrappables" on top of our base space, it is always best practice to unwrap before evaluation.

Finally, we generate predictions across the input space and plot the results.

<!-- pytest-codeblocks:cont -->
```python
import matplotlib.pyplot as plt

clean_samples = jax.tree.map(lambda x: x[500:], base_samples)

base_models = eqx.combine(clean_samples, static)
sampled_models = prxp.tree_update(initial_model, base_models)

unwrapped_models = prx.unwrap(sampled_models)

x_plot = jnp.linspace(-3, 3, 100)
y_preds = eqx.filter_vmap(unwrapped_models)(x_plot).T
y_mean = jnp.mean(y_preds, axis=0)
y_lower, y_upper = jnp.percentile(y_preds, jnp.array([2.5, 97.5]), axis=0)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(clean_samples.weight, bins=30, density=True, color='steelblue', edgecolor='black')
axes[0].axvline(2.5, color='red', linestyle='dashed', linewidth=2, label='True Value')
axes[0].set_title('Weight Posterior')
axes[0].legend()

axes[1].hist(clean_samples.bias, bins=30, density=True, color='seagreen', edgecolor='black')
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

# 5. Advantages of Parax
You may have noticed that we could have accomplished the above without the added abstraction of `parax.Random` variable wrappers (i.e. by defining our models using `distreqx` distributions directly). However, building models using Parax variables (`parax.AbstractVariables`) has a number of quality-of-life benefits:

- *Easy fixing of variables*. You can't "fix a distribution" after-the-fact without complex filtering, but you can easily wrap a `parax.Random` variable in a `parax.Fixed` variable.
- *Compatibility with optimization*. It is common to want to swap between optimization and Bayesian sampling. Using Parax, we could easily define a factory that wraps a `parax.Constrained` in a `parax.Random`, allowing us to toggle between bounded optimation and inference.
- *Parameters as first-class citizens*. Models contain parameters - not distributions. By prioritizing a parameter-centred approach, and simply attaching priors as metadata and using tree tools at model setup, we maintain a clear separation of concerns. This naturally has downstream benefits.