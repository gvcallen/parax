"""
Parameter factories with pre-defined probability distributions.
"""

from typing import Sequence

import jax.numpy as jnp
import distreqx.distributions as dist

from parax.parameter import Parameter


def Uniform(low: float | Sequence[float], high: float | Sequence[float], value=None, add_bounds=True, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with a uniform distribution.
    
    Parameters
    ----------
    low : float | Sequence[float]
        The lower value of the distribution. Can be a sequence for a multi-valued Parameter.
    high : float | Sequence[float]
        The upper value of the distribution. Can be a sequence for a multi-valued Parameter.
    value : optional
        The initial value. If None, the midpoint of the distribution is used. Defaults to None.
    add_bounds : bool
        Whether or not bounds should automatically be added at `low` and `high`. Defaults to True.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created Parameter object.
    """
    low, high = jnp.array(low, dtype=float), jnp.array(high, dtype=float)
    dists = dist.Uniform(low, high)
    
    if add_bounds:
        kwargs.setdefault('bounds', jnp.stack([low, high], axis=-1))
    
    if 'latent_value' in kwargs:
        return Parameter(distribution=dists, **kwargs)
        
    values = (low + high) / 2.0 if value is None else value
    return Parameter(value=values, distribution=dists, **kwargs)


def RelativeUniform(mean: float | Sequence[float], deviation_fraction: float | Sequence[float], *args, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with a uniform distribution defined by a fractional deviation.

    The bounds are calculated as: `mean * (1 +/- deviation_fraction)`

    Parameters
    ----------
    mean : float | Sequence[float]
        The center (mean) of the distribution.
    deviation_fraction : float | Sequence[float]
        The relative radius of the distribution bounds as a fraction of the mean.
        e.g., 0.1 results in bounds of [0.9 * mean, 1.1 * mean].
    **kwargs
        Additional keyword arguments passed to [`parax.Uniform`][].

    Returns
    -------
    Parameter
    """
    mean_arr = jnp.array(mean, dtype=float)
    frac_arr = jnp.array(deviation_fraction, dtype=float)
    
    # Calculate the absolute deviation (radius)
    delta = jnp.abs(mean_arr * frac_arr)
    
    return Uniform(mean_arr - delta, mean_arr + delta, *args, **kwargs)


def CenteredUniform(mean: float | Sequence[float], half_width: float | Sequence[float], *args, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with a uniform distribution.

    Parameters
    ----------
    mean : float | Sequence[float]
        The mean value of the distribution. Can be a sequence for a multi-valued Parameter.
    half_width : float | Sequence[float]
        The half-width value of the distribution. Can be a sequence for a multi-valued Parameter.
    **kwargs
        Additional keyword arguments passed to [`parax.Uniform`][].

    Returns
    -------
    Parameter
        The created Parameter object.
    """
    low = mean - half_width
    high = mean + half_width
    
    return Uniform(low, high, *args, **kwargs)


def Normal(mean: float | Sequence[float], std: float | Sequence[float], value=None, icdf_bounds=0.001, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with a normal (Gaussian) distribution.

    Parameters
    ----------
    mean : float | Sequence[float]
        The mean of the distribution. Can be a sequence for a multi-valued Parameter.
    std : float | Sequence[float]
        The standard deviation of the distribution. Can be a sequence for a multi-valued Parameter.
    value : optional
        The initial value. If None, the mean of the distribution is used. Defaults to None.
    icdf_bounds : float | None
        The percentile to place bounds at using the inverse CDF.
        Bounds are placed so that `min = normal.icdf(icdf_bounds)` and `max = normal.icdf(1.0-icdf_bounds)`.
        Defaults to 1%. Can be None for no bounds.
    **kwargs
        Additional keyword arguments forward to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created Parameter object.
    """
    mean, std = jnp.array(mean, dtype=float), jnp.array(std, dtype=float)
    dists = dist.Normal(mean, std)
    
    if icdf_bounds is not None:
        min, max = dists.icdf(icdf_bounds), dists.icdf(1.0-icdf_bounds)
        kwargs.setdefault('bounds', jnp.stack([min, max], axis=-1))
    
    if 'latent_value' in kwargs:
        return Parameter(distribution=dists, **kwargs)\
        
    values = mean if value is None else value
    return Parameter(value=values, distribution=dists, **kwargs)


def RelativeNormal(mean: float | Sequence[float], std_fraction: float | Sequence[float], **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with a normal distribution defined by a relative standard deviation.

    The scale (sigma) is calculated as: `mean * std_fraction`

    Parameters
    ----------
    mean : float | Sequence[float]
        The center (mean) of the distribution.
    std_fraction : float | Sequence[float]
        The standard deviation expressed as a fraction of the mean 
        (also known as the coefficient of variation).
        e.g., 0.1 results in a distribution with sigma = 0.1 * mean.
    **kwargs
        Additional keyword arguments passed to [`parax.Normal`][].

    Returns
    -------
    Parameter
    """
    mean_arr = jnp.array(mean, dtype=float)
    frac_arr = jnp.array(std_fraction, dtype=float)
    
    # Calculate absolute standard deviation
    sigma = jnp.abs(mean_arr * frac_arr)
    
    return Normal(mean=mean_arr, std=sigma, **kwargs)


def Fixed(value=None, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` that is marked as fixed.
    
    This sets the `fixed` flag of the parameter to `True`.

    Parameters
    ----------
    value : optional
        The value of the parameter. If None, `latent_value` must be provided in kwargs.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created fixed Parameter object.
    """
    if 'latent_value' in kwargs:
        return Parameter(fixed=True, **kwargs)
        
    if value is None:
        raise ValueError("Must provide either `value` or `latent_value`.")
        
    value = jnp.array(value, dtype=float)
    return Parameter(value=value, fixed=True, **kwargs)


def Free(value=None, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` that is marked as free (i.e., free to vary).
    
    This sets the `fixed` flag of the parameter to `False`.

    Parameters
    ----------
    value : optional
        The value of the parameter. If None, `latent_value` must be provided in kwargs.
    n : int, optional
        The number of identical parameters to create in an array. Defaults to None.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created free Parameter object.
    """
    if 'latent_value' in kwargs:
        return Parameter(fixed=False, **kwargs)
        
    if value is None:
        raise ValueError("Must provide either `value` or `latent_value`.")
        
    value = jnp.array(value, dtype=float)
    return Parameter(value=value, fixed=False, **kwargs)


def Bounded(min: float | Sequence[float], max: float | Sequence[float], value=None, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with specified bounds.
    
    Parameters
    ----------
    min : float | Sequence[float]
        The minimum bounds of the parameter. Can be a sequence for a multi-valued Parameter.
    max : float | Sequence[float]
        The maximum bounds of the parameter. Can be a sequence for a multi-valued Parameter.
    value : optional
        The initial value. If None, the midpoint of the bounds is used. Defaults to None.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The bounded Parameter object.
    """
    min, max = jnp.array(min, dtype=float), jnp.array(max, dtype=float)
    
    if 'latent_value' in kwargs:
        return Parameter(**kwargs)
        
    values = (min + max) / 2.0 if value is None else value
    return Parameter(value=values, bounds=jnp.stack([min, max], axis=-1))