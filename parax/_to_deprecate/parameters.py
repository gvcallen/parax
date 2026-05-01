"""
Parameter factories with pre-defined probability distributions and constraints.
"""

from jaxtyping import ArrayLike

import jax.numpy as jnp
import distreqx.distributions as dist

from parax.parameter import Parameter
from parax.constraints import Interval

def Uniform(low: float | ArrayLike, high: float | ArrayLike, value=None, make_constraint: bool = True, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with a uniform distribution.
    
    Parameters
    ----------
    low : float | ArrayLike
        The lower value of the distribution. Can be a sequence for a multi-valued Parameter.
    high : float | ArrayLike
        The upper value of the distribution. Can be a sequence for a multi-valued Parameter.
    value : optional
        The initial value. If None, the midpoint of the distribution is used. Defaults to None.
    make_constraint : bool
        Creates an interval constraint with default bounds `low` and `high`. Defaults to True.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created Parameter object.
    """
    low, high = jnp.array(low, dtype=float), jnp.array(high, dtype=float)
    dists = dist.Uniform(low, high)
    
    if make_constraint:
        kwargs.setdefault('constraint', Interval(low, high))
    
    if 'raw_value' in kwargs:
        return Parameter(distribution=dists, **kwargs)
        
    values = (low + high) / 2.0 if value is None else value
    return Parameter(unscaled_value=values, distribution=dists, **kwargs)


def RelativeUniform(mean: float | ArrayLike, deviation_fraction: float | ArrayLike, *args, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with a uniform distribution defined by a fractional deviation.

    The bounds are calculated as: `mean * (1 +/- deviation_fraction)`

    Parameters
    ----------
    mean : float | ArrayLike
        The center (mean) of the distribution.
    deviation_fraction : float | ArrayLike
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


def CenteredUniform(mean: float | ArrayLike, half_width: float | ArrayLike, *args, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with a uniform distribution.

    Parameters
    ----------
    mean : float | ArrayLike
        The mean value of the distribution. Can be a sequence for a multi-valued Parameter.
    half_width : float | ArrayLike
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


def Normal(mean: float | ArrayLike, std: float | ArrayLike, value=None, make_constraint: bool = True, interval_std: float = 2.0, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with a normal (Gaussian) distribution.

    Parameters
    ----------
    mean : float | ArrayLike
        The mean of the distribution. Can be a sequence for a multi-valued Parameter.
    std : float | ArrayLike
        The standard deviation of the distribution. Can be a sequence for a multi-valued Parameter.
    value : optional
        The initial value. If None, the mean of the distribution is used. Defaults to None.
    make_constraint : bool
        Creates an interval constraint with default bounds `mean` +- `interval_std` * `std`. Defaults to True.
    interval_std : float
        Number of standard deviations from the mean to place the bounds if `make_constraint` is True.
        Defaults to 2.0.
    **kwargs
        Additional keyword arguments forward to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created Parameter object.
    """
    mean, std = jnp.array(mean, dtype=float), jnp.array(std, dtype=float)
    dists = dist.Normal(mean, std)
    
    if make_constraint:    
        lower = mean - interval_std * std
        upper = mean + interval_std * std
        kwargs.setdefault('constraint', Interval(lower, upper))
    
    if 'raw_value' in kwargs:
        return Parameter(distribution=dists, **kwargs)\
        
    values = mean if value is None else value
    return Parameter(unscaled_value=values, distribution=dists, **kwargs)


def RelativeNormal(mean: float | ArrayLike, std_fraction: float | ArrayLike, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with a normal distribution defined by a relative standard deviation.

    The scale (sigma) is calculated as: `mean * std_fraction`

    Parameters
    ----------
    mean : float | ArrayLike
        The center (mean) of the distribution.
    std_fraction : float | ArrayLike
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
        The value of the parameter. If None, `raw_value` must be provided in kwargs.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created fixed Parameter object.
    """
    if 'raw_value' in kwargs:
        return Parameter(fixed=True, **kwargs)
        
    if value is None:
        raise ValueError("Must provide either `value` or `raw_value`.")
        
    value = jnp.array(value, dtype=float)
    return Parameter(unscaled_value=value, fixed=True, **kwargs)


def Free(value=None, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` that is marked as free (i.e., free to vary).
    
    This sets the `fixed` flag of the parameter to `False`.

    Parameters
    ----------
    value : optional
        The value of the parameter. If None, `raw_value` must be provided in kwargs.
    n : int, optional
        The number of identical parameters to create in an array. Defaults to None.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created free Parameter object.
    """
    if 'raw_value' in kwargs:
        return Parameter(fixed=False, **kwargs)
        
    if value is None:
        raise ValueError("Must provide either `value` or `raw_value`.")
        
    value = jnp.array(value, dtype=float)
    return Parameter(unscaled_value=value, fixed=False, **kwargs)


def Bounded(lower: float | ArrayLike, upper: float | ArrayLike, value=None, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with specified bounds using an interval constraint.
    
    Parameters
    ----------
    lower : float | ArrayLike
        The lower bound of the parameter. Can be a sequence for a multi-valued Parameter.
    upper : float | ArrayLike
        The upper bound of the parameter. Can be a sequence for a multi-valued Parameter.
    value : optional
        The initial value. If None, the midpoint of the bounds is used. Defaults to None.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The bounded Parameter object.
    """
    lower, upper = jnp.array(lower, dtype=float), jnp.array(upper, dtype=float)
    
    if 'raw_value' in kwargs:
        return Parameter(**kwargs)
    
    kwargs.setdefault('constraint', Interval(lower, upper))
        
    values = (lower + upper) / 2.0 if value is None else value
    return Parameter(unscaled_value=values, **kwargs)