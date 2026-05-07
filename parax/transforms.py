"""
Common transformations for derived parameters.

This provides stateless callable transformations
which can easily be used with `parax.Derived` and `parax.Computed`
variables and PyTrees. They can be used over lambdas if preferred.

Unlike constraints, transformations are not required
to be bijective (invertible), and do not define bounds.
"""

from abc import abstractmethod
from typing import Union, Callable, Any, TypeGuard

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PyTree

from distreqx.bijectors import AbstractBijector


class AbstractTransform(eqx.Module):
    """
    The base class for all callable transformations in Parax.

    Transformations are applied to variables, typically within `parax.Derived`. 
    Because they inherit from `equinox.Module`, any JAX arrays stored as 
    attributes (e.g., learnable shifts or scales) can be tracked 
    by optimizers and serialization utilities.
    """

    @abstractmethod
    def __call__(self, x: Array) -> Array:
        """
        Applies the transformation to the input array.

        Args:
            x: The input array to transform.

        Returns:
            The transformed array.
        """
        pass


def is_transform(x: Any) -> TypeGuard[AbstractTransform]:
    """
    Returns True if `x` is an instance of `parax.AbstractTransform`.
    """
    return isinstance(x, AbstractTransform)


class Affine(AbstractTransform):
    """
    Applies a standard affine transformation: `f(x) = x * scale + shift`.

    Useful for basic standardizations, unit conversions, or parameterizing 
    linear adjustments in a scientific model.

    Attributes:
        shift: The translation applied to the input.
        scale: The multiplier applied to the input.
    """
    shift: Float[Array, "..."]
    scale: Float[Array, "..."]

    def __init__(self, shift: Union[float, Array] = 0.0, scale: Union[float, Array] = 1.0):
        """
        Args:
            shift: The translation scalar or array. Defaults to 0.0.
            scale: The multiplier scalar or array. Defaults to 1.0.
        """
        self.shift = jnp.asarray(shift, dtype=float)
        self.scale = jnp.asarray(scale, dtype=float)

    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: The input array.

        Returns:
            The shifted and scaled array.
        """
        return x * self.scale + self.shift
    

class Shift(Affine):
    """
    Applies a pure translation transformation: `f(x) = x + shift`.

    A convenience subclass of `parax.transforms.Affine` strictly for 
    shifting values without scaling.
    """
    
    def __init__(self, shift: Union[float, Array]):
        """
        Args:
            shift: The translation scalar or array.
        """
        super().__init__(shift=shift, scale=1.0)


class Scale(Affine):
    """
    Applies a pure scaling transformation: `f(x) = x * scale`.

    A convenience subclass of `parax.transforms.Affine` strictly for 
    scaling values without translation.
    """
    
    def __init__(self, scale: Union[float, Array]):
        """
        Args:
            scale: The multiplier scalar or array.
        """
        super().__init__(shift=0.0, scale=scale)


class Clip(AbstractTransform):
    """
    Clips (limits) the values in an array to a specified interval.

    **Corner Case Note:** This is a mathematically destructive transformation 
    (gradients at the boundaries are zero). It should be used for hard 
    thresholding, not as a replacement for smooth `parax.constraints`.

    Attributes:
        lower: The minimum allowable value.
        upper: The maximum allowable value.
    """
    lower: Float[Array, "..."]
    upper: Float[Array, "..."]

    def __init__(self, lower: Union[float, Array] = -jnp.inf, upper: Union[float, Array] = jnp.inf):
        """
        Args:
            lower: The minimum allowable value. Defaults to -inf.
            upper: The maximum allowable value. Defaults to inf.
        """
        self.lower = jnp.asarray(lower, dtype=float)
        self.upper = jnp.asarray(upper, dtype=float)

    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: The input array.

        Returns:
            The clipped array.
        """
        return jnp.clip(x, min=self.lower, max=self.upper)
    

class Reshape(AbstractTransform):
    """
    Reshapes an array to a specified target shape.

    Essential for bridging flat optimizer spaces with multi-dimensional 
    scientific or spatial models (e.g., reshaping a 1D parameter array 
    into a 2D spatial field).

    Attributes:
        shape: The target shape tuple. Can include `-1` to infer the 
            size of one dimension automatically.
    """
    shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, shape: tuple[int, ...]):
        """
        Args:
            shape: The desired target shape.
        """
        self.shape = shape

    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: The input array.

        Returns:
            The reshaped array.
        """
        return jnp.reshape(x, self.shape)


class Round(AbstractTransform):
    """
    Rounds array elements to a given number of decimals.

    Useful for quantizing continuous parameters into discrete physical 
    states (e.g., whole integer quantities). 

    **Corner Case Note:** This is a mathematically destructive, non-bijective 
    transformation with zero gradients almost everywhere. It will stop 
    standard autodiff dead in its tracks unless paired with a custom 
    gradient estimator (like a straight-through estimator).

    Attributes:
        decimals: The number of decimal places to round to.
    """
    decimals: int = eqx.field(static=True)

    def __init__(self, decimals: int = 0):
        """
        Args:
            decimals: Number of decimal places to round to. Defaults to 0 
                (rounds to nearest integer).
        """
        self.decimals = decimals

    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: The input array.

        Returns:
            The rounded array.
        """
        return jnp.round(x, decimals=self.decimals)


class Softmax(AbstractTransform):
    """
    Applies the softmax function over a specified axis.

    This is a classic non-bijective transformation mapping real numbers 
    to a probability simplex (values sum to 1.0). Usefl in ML and 
    categorical scientific modeling.

    Attributes:
        axis: The axis or axes along which the softmax should be computed.
    """
    axis: int | tuple[int, ...] | None = eqx.field(static=True)

    def __init__(self, axis: Union[int, tuple[int, ...], None] = -1):
        """
        Args:
            axis: The axis or axes along which to compute the softmax. 
                Defaults to -1 (the last axis).
        """
        self.axis = axis

    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: The input array.

        Returns:
            An array of the same shape where the specified axis forms a 
            probability distribution.
        """
        return jax.nn.softmax(x, axis=self.axis)
    

class LogSoftmax(AbstractTransform):
    """
    Applies the log-softmax function over a specified axis.

    Mathematically equivalent to `log(softmax(x))`, but implemented in JAX 
    to be vastly more numerically stable. Useful when modeling 
    log-probabilities or energy states to prevent underflow.

    Attributes:
        axis: The axis or axes along which the log-softmax should be computed.
    """
    axis: int | tuple[int, ...] | None = eqx.field(static=True)

    def __init__(self, axis: Union[int, tuple[int, ...], None] = -1):
        """
        Args:
            axis: The axis or axes along which to compute the log-softmax. 
                Defaults to -1 (the last axis).
        """
        self.axis = axis

    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: The input array.

        Returns:
            An array containing the log-probabilities along the specified axis.
        """
        return jax.nn.log_softmax(x, axis=self.axis)


class Normalize(AbstractTransform):
    """
    Normalizes an array to have zero mean and unit variance.
    
    **Corner Case Note:** This transform computes the mean and variance 
    dynamically across the provided input `x` at evaluation time. It does 
    *not* store running statistics (like a BatchNorm layer).

    Attributes:
        axis: The axis or axes along which to compute the statistics.
        epsilon: A small scalar added to the variance to prevent division by zero.
    """
    axis: int | tuple[int, ...] | None = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(self, axis: Union[int, tuple[int, ...], None] = None, epsilon: float = 1e-5):
        """
        Args:
            axis: The axis along which to normalize. If None, normalizes 
                across the entire flattened array. Defaults to None.
            epsilon: Small scalar to prevent division by zero. Defaults to 1e-5.
        """
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: The input array.

        Returns:
            The normalized array with mean 0 and variance 1 along the specified axis.
        """
        mean = jnp.mean(x, axis=self.axis, keepdims=True)
        variance = jnp.var(x, axis=self.axis, keepdims=True)
        return (x - mean) * jax.lax.rsqrt(variance + self.epsilon)
    

class Chain(AbstractTransform):
    """
    Composes a sequence of transformations into a single transformation.

    The transformations are applied in reverse order to match standard 
    mathematical function composition.
    Mathematically, `Chain([f, g, h])(x)` is equivalent to `f(g(h(x)))`.

    Attributes:
        transforms: A tuple containing the sequence of transformations.
    """
    transforms: tuple[AbstractTransform, ...]

    def __init__(self, transforms: list[AbstractTransform] | tuple[AbstractTransform, ...]):
        """
        Args:
            transforms: A sequence (list or tuple) of `AbstractTransform` instances.
                They will be applied from right-to-left (last element first).
        """
        # Convert to tuple to guarantee immutability as an Equinox PyTree node
        self.transforms = tuple(transforms)

    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: The input array.

        Returns:
            The array mapped sequentially through all transformations in reverse order.
        """
        for transform in reversed(self.transforms):
            x = transform(x)
        return x


class BijectorTransform(AbstractTransform):
    """
    A transformation powered by a distreqx bijector.

    Applies the forward pass of a given `distreqx.bijectors.AbstractBijector`.
    This is the standard bridge for injecting complex, mathematically rigorous 
    bijections into derived variables.

    Attributes:
        bijector: The underlying distreqx bijector.
    """
    bijector: AbstractBijector

    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: The input array.

        Returns:
            The array mapped through the bijector's forward pass.
        """
        return self.bijector.forward(x)
    

class TreeTransform(AbstractTransform):
    """
    Represents a PyTree of transformations mapping over a PyTree of inputs.
    
    Useful for applying heterogeneous transformations to complex nested structures 
    (like `equinox.Module` instances) simultaneously.

    Attributes:
        tree: The PyTree containing `AbstractTransform` leaves.
    """
    tree: PyTree[AbstractTransform]

    def __init__(
        self, 
        transforms: PyTree[AbstractTransform],
    ):
        """
        Args:
            transforms: A PyTree containing `AbstractTransform` leaves.
                Non-transform leaves are ignored.
        
        Raises:
            ValueError: If the provided PyTree contains no transform leaves.
        """
        # Local import prevents circular dependency at initialization time
        leaves = jax.tree.leaves(transforms, is_leaf=is_transform)
        if not leaves:
            raise ValueError("The pytree of transforms cannot be empty.")

        self.tree = transforms

    def __call__(self, x: PyTree[Array]) -> PyTree[Array]:
        """
        Maps each leaf transform over the corresponding node in the input PyTree.
        
        Args:
            x: The input PyTree of arrays. Must have a matching tree prefix 
                to the `transforms` PyTree.
                
        Returns:
            A new PyTree containing the transformed arrays.
        """
        def _apply_transform(transform: Any, val: Any) -> Any:
            if not is_transform(transform):
                return val
            return transform(val)

        return jax.tree_util.tree_map(_apply_transform, self.tree, x, is_leaf=is_transform)


class CustomTransform(AbstractTransform):
    """
    An escape hatch for power users who need to apply an arbitrary function 
    as a transformation while strictly adhering to the `AbstractTransform` type hierarchy.

    **Corner Case Note:** The underlying callable is marked as `static=True`. 
    This prevents JAX from attempting to flatten raw Python functions (like lambdas). 
    If your custom transformation requires learnable arrays or state, you should 
    subclass `AbstractTransform` directly instead of using this wrapper.

    Attributes:
        _custom_fn: The internal, user-defined callable.
    """
    _custom_fn: Callable = eqx.field(static=True)

    def __init__(
        self, 
        fn: Callable
    ):
        """
        Args:
            fn: The custom callable. Must accept a single array argument 
                and return a transformed array.
        """
        self._custom_fn = fn

    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: The input array.

        Returns:
            The array mapped through the custom function.
        """
        return self._custom_fn(x)