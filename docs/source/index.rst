Parax: Parametric Modeling in JAX
===================================================================

**Parax**, is a declarative, parametric modelling library built on top of `JAX <https://github.com/jax-ml/jax>`_ and `Equinox <https://github.com/patrick-kidger/equinox>`_.

At its core, the library provides a `Parameter` class which can be set as fixed for training, as well as assigned arbitrary metadata. Core metadata includes assigning a name, description, scale, units, bounds, probability distribution and bijector/transform.

The transform and scale metadata are particularly useful. The raw `value` inside a parameter is stored *untransformed* and *unscaled*. However, parameters can be used directly in mathematical expression as if they were JAX arrays, at which point these transforms are applied. This completely abstracts the underlying *value* (to be used in optimization) from the user, bypassing the need to explicitly apply the transform.

To make optimization easy, `Parax` comes with a built-in `parax.partition` function, which partitions a model into trainable parameters. If a model is built purely using `Parameter`'s, this removes the need for any conditional logical that would usually be done manually during `eqx.partition`.

Further, `Parax` also provides an extended version of Equinox's `Module` in `parax.Module`. This allows for parameter-aware module inspection and manipulation. For example, parameters can easily be flattened, updated using a single string assigned using the hierarchy, and mapped in batches.

The library is mainly intended for us in domain-specific scientific modeling, but can easily be applied to broader applications.

+----------+------------------------------------------------+
| Version  | |release|                                      |
+----------+------------------------------------------------+
| Author   | Gary Allen                                     |
+----------+------------------------------------------------+
| Homepage | https://github.com/parax/parax                 |
+----------+------------------------------------------------+
| Docs     | https://gvcallen.github.io/parax               |
+----------+------------------------------------------------+

Installation
=====================
Parax can be installed directly using pip:

``pip install parax``

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   api/index
   license


Example
=====================

In this example, we define a simple quadratic model ($y = ax^2 + bx + c$). We fix the y-intercept, leave the other coefficients free, and use JAX and ``optimistix`` to fit the model to some noisy data.

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optimistix as optx

    import parax as prx
    from parax.parameters import Free, Fixed

    # 1. Define the Parametric Model
    class Quadratic(eqx.Module):
        """A generic quadratic curve: y = a*x^2 + b*x + c"""
        
        a: prx.Parameter
        b: prx.Parameter
        c: prx.Parameter

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            return self.a * (x ** 2) + self.b * x + self.c
        
    model = Quadratic(a=Free(1.5), b=Free(0.5), c=Fixed(10.0))

    # 2. Generate some dummy "ground truth" data with noise
    x_true = jnp.linspace(-5.0, 5.0, 100)
    y_true = 3.0 * (x_true ** 2) - 2.0 * x_true + 10.0 # True a=3.0, b=-2.0
    y_true = y_true + jax.random.normal(jax.random.key(0), x_true.shape)

    params, static = prx.partition(model)

    # 3. Define the loss Function
    def loss_fn(params, args=None):
        model = eqx.combine(params, static)
        y_pred = model(x_true)
        return jnp.mean((y_pred - y_true)**2)

    # 4. Run the BFGS optimizer
    solver = optx.LBFGS(rtol=1e-6, atol=1e-6)
    solution = optx.minimise(
        fn=loss_fn,
        y0=params,
        solver=solver,
        args=(x_true, y_true, static),
    )

    # 5. Recombine to get the final fitted model
    fitted_model = eqx.combine(solution.value, static)

    print(f"Fitted 'a': {jnp.array(fitted_model.a):.8f} (Expected ~3.0)")
    print(f"Fitted 'b': {jnp.array(fitted_model.b):.8f} (Expected ~-2.0)")
    print(f"Fixed 'c':  {jnp.array(fitted_model.c):.8f} (Remained 10.0)")
    print(f'Final loss: {loss_fn(fitted_model)}')
    print(solution.result)