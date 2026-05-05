# 1. Defining the model

Let's define a simple linear regression model: $y = w \cdot x + b$. Instead of regular parameters, we will assign probability distributions to our variables using `prx.Random` variables to establish our priors.

```python
import equinox as eqx
import parax as prx
from distreqx.distributions import Normal

# Define the probabilistic model
class BayesianLinearModel(eqx.Module):
    # Assign normal priors to our weight and bias
    weight: prx.ParamLike = prx.random(Normal(0.0, 5.0))
    bias: prx.ParamLike = prx.random(Normal(0.0, 2.0))

    def __call__(self, x):
        # Math operations instantly unwrap variables to JAX arrays
        return self.weight * x + self.bias

# Initialize the model with our initial guesses (raw values)
initial_model = BayesianLinearModel(weight=1.0, bias=0.0)
```

# 2. Setting up the log posterior

In this tutorial, we will use `blackjax` for Bayesian sampling. We need to provide a function that takes our model parameters in an unconstrained space and returns an unnormalized log-posterior.

First, we use `parax.probabilistic` to extract the initial model values in the "base" probability space (where the distributions are defined) as well as our joint prior. Then, we partition the model into parameters and static metadata. Finally, in the log posterior function, we combine + unwrap and evaluate our data.

<!-- pytest-codeblocks:cont -->
```python
import jax
import jax.numpy as jnp
import parax.probabilistic as prxp

# Extract the unconstrained base values and the joint prior before inference
initial_base = prxp.tree_base(initial_model)
prior_dist = prxp.tree_joint(initial_model)

# Partition the base model and prior values
filter_spec = eqx.is_inexact_array
params, static = eqx.partition(initial_base, filter_spec, is_leaf=prx.is_constant)
prior, _ = eqx.partition(prior_dist, filter_spec, is_leaf=prx.is_constant)

# Define the log posterior
def log_posterior_fn(p, static, x_data, y_true):
    unwrapped_model = prx.unwrap(eqx.combine(p, static))
    log_prior = prior_dist.log_prob(unwrapped_model)
    
    # We assume Gaussian noise with a std of 1.0
    y_pred = jax.vmap(unwrapped_model)(x_data)
    log_likelihood = jnp.sum(Normal(y_pred, 1.0).log_prob(y_true))
    
    return log_prior + log_likelihood
```

# 3. Running the sampler

Now we generate some noisy dummy data and set up our MCMC sampler. We'll use the No-U-Turn Sampler (NUTS) without window adaptation.

<!-- pytest-codeblocks:cont -->
```python
import blackjax
import jax.random as jr
from jax.flatten_util import ravel_pytree # needed for inverse mass matrix

# Generate dummy data (true weight = 2.5, true bias = 1.0)
rng_key = jr.key(42)
x_data = jnp.linspace(-2, 2, 50)
y_data = 2.5 * x_data + 1.0 + jr.normal(rng_key, (50,))

# Set up the NUTS sampler
logprob = lambda p: log_posterior_fn(p, static, x_data, y_data)
inv_mass_matrix = jnp.ones_like(ravel_pytree(params)[0])
nuts = blackjax.nuts(logprob, step_size=1e-2, inverse_mass_matrix=inv_mass_matrix)

# Initialize state
initial_state = nuts.init(params)

# Define the sampling loop
@eqx.filter_jit
def run_mcmc(key, state, num_steps):
    def step_fn(state, rng_key):
        state, _ = nuts.step(rng_key, state)
        return state, state.position
    
    keys = jr.split(key, num_steps)
    _, samples = jax.lax.scan(step_fn, state, keys)
    return samples

# Run the sampler!
sample_key = jr.key(0)
base_samples = run_mcmc(sample_key, initial_state, num_steps=2000)
```

# 4. Evaluating the results

Because `blackjax` preserves the PyTree structure of our inputs, `base_samples` has the exact same structure as our partitioned params tree. We can use `eqx.filter_jit` along with `parax.probabilistic.tree_update` to reconstruct the model and plot parameters and functional posteriors!

<!-- pytest-codeblocks:cont -->
```python
import matplotlib.pyplot as plt

# Discard the first 500 steps as warmup/burn-in
clean_samples = jax.tree.map(lambda x: x[500:], base_samples)

# Create the combined model
base_models = eqx.combine(clean_samples, static)
sampled_models = prxp.tree_update(initial_model, base_models)

# Although we technically didn't use any parax "unwrappables" on top
# of our base space, it is always best to unwrap before evaluation
unwrapped_models = prx.unwrap(sampled_models)

# Generate predictions across the input space.
# Transpose so that we have nsamples x 100
x_plot = jnp.linspace(-3, 3, 100)
y_preds = eqx.filter_vmap(unwrapped_models)(x_plot).T
y_mean = jnp.mean(y_preds, axis=0)
y_lower, y_upper = jnp.percentile(y_preds, jnp.array([2.5, 97.5]), axis=0)

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Plot Weight Posterior
axes[0].hist(clean_samples.weight, bins=30, density=True, color='steelblue', edgecolor='black')
axes[0].axvline(2.5, color='red', linestyle='dashed', linewidth=2, label='True Value')
axes[0].set_title('Weight Posterior')
axes[0].legend()

# Plot Bias Posterior
axes[1].hist(clean_samples.bias, bins=30, density=True, color='seagreen', edgecolor='black')
axes[1].axvline(1.0, color='red', linestyle='dashed', linewidth=2, label='True Value')
axes[1].set_title('Bias Posterior')
axes[1].legend()

# Plot Functional Posterior
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

You may have noticed that we could have accomplish the above without the added abstraction of `parax.Random` variable wrappers (i.e. by defining our models using `distreqx` distributions directly). However, building models using Parax variables (`parax.AbstractVariables`) has a number of quality-of-live benefits:

- **Easy fixing of variables**. You can't "fix a distribution" after-the-fact without complex filtering, but you can easily wrap a `parax.Random` variable in a `parax.Fixed` variable.
- **Compatibility with optimization**. It is common to want to swap between optimization and Bayesian sampling. For example, we could easily define a factory that wraps a `parax.Random` in a `parax.Constrained`, allowing us to toggle between bounded optimation and inference.
- **Parameters as first-class citizens**. Models contain parameters - not distributions. By prioritizing a parameter-centred approach, and simply attaching priors as metadata and using tree tools at model setup, we maintain a clear separation of concerns. This naturally has downstream benefits.