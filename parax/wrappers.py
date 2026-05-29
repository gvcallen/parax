"""
The interface and built-in classes for PyTree wrapping and unwrapping.

This module provides abstract interfaces for wrappables/unwrappables,
as well as tools for deferred evaluation, parameterization, gradient stopping
and more for JAX PyTrees.
"""
import functools

from abc import abstractmethod
from typing import Generic, TypeVar, TypeGuard, Callable, Any, Union, Self

import equinox as eqx
import jax
from jaxtyping import PyTree
from distreqx.distributions import AbstractDistribution

from parax.constants import AbstractConstant
from parax.annotation import AbstractAnnotated
from parax.bounds import AbstractBounded
from parax.probability import AbstractProbabilistic
from parax.constraints import AbstractConstraint, AbstractConstrainable


T = TypeVar("T")


class AbstractUnwrappable(eqx.Module, Generic[T]):
    """An abstract class representing a deferred or wrapped PyTree node.

    Unwrappables act as placeholders within a PyTree. When `parax.unwrap` 
    is called on the tree, these nodes execute custom logic (such as delayed 
    computation, parameter injection, or gradient stopping) and replace 
    themselves with their output.
    """

    @abstractmethod
    def unwrap(self) -> T:
        """Evaluates and returns the underlying unwrapped PyTree node.
        
        Returns:
            The resolved underlying value, assuming no wrapped subnodes exist.
        """
        pass


def is_unwrappable(x: Any) -> TypeGuard[AbstractUnwrappable]:
    """Checks if a given object is an unwrappable node.
    
    Args:
        x: The object to check.
        
    Returns:
        True if `x` is an instance of `AbstractUnwrappable`, False otherwise.
    """
    return isinstance(x, AbstractUnwrappable)


def unwrap(tree: Union[AbstractUnwrappable[T] | T], only_if: Callable[[Any], bool] = None) -> T:
    """Recursively resolves `AbstractUnwrappable` nodes within a PyTree.

    By default, unwrapping is performed inside-out (bottom-up) across the entire 
    tree. Every `AbstractUnwrappable` node is replaced by the result of its 
    `unwrap()` method.

    If the `only_if` predicate is provided, unwrapping is conditionally gated. 
    The tree is searched top-down, and unwrapping only triggers for subtrees 
    that satisfy the condition. Once a node satisfies `only_if`, that entire 
    subtree is fully unwrapped. 

    Behavior with `only_if`:
        1. If `only_if(node)` is True: The node and all of its `AbstractUnwrappable` 
           descendants are fully resolved.
        2. If `only_if(node)` is False: The node is left wrapped, but the search 
           continues recursively into its children.

    Args:
        tree: The PyTree to unwrap.
        only_if: An optional predicate function `Callable[[Any], bool]`. If provided, 
            only subtrees evaluating to True (and their descendants) are unwrapped.

    Returns:
        A new PyTree with the appropriate `AbstractUnwrappable` nodes resolved.
    """
    def _do_unwrap(node, *, include_self: bool):
        def _map_fn(leaf):
            if not is_unwrappable(leaf):
                return leaf
            # Recursively unwrap the children first (bottom-up)
            resolved_node = _do_unwrap(leaf, include_self=False)
            return resolved_node.unwrap()

        def is_leaf(x):
            included = True if x is not node else include_self
            return is_unwrappable(x) and included

        return jax.tree.map(_map_fn, node, is_leaf=is_leaf)
    
    def _search_and_unwrap(node, *, include_self: bool):
        if include_self and only_if(node):
            return _do_unwrap(node, include_self=True)
            
        def _map_fn(leaf):
            if only_if(leaf):
                return _do_unwrap(leaf, include_self=True)
            
            if is_unwrappable(leaf):
                return _search_and_unwrap(leaf, include_self=False)
            
            return leaf

        def is_leaf(x):
            included = True if x is not node else include_self
            return (is_unwrappable(x) or only_if(x)) and included

        return jax.tree.map(_map_fn, node, is_leaf=is_leaf)
    
    if only_if is None:  # fast path
        return _do_unwrap(tree, include_self=True)    
    return _search_and_unwrap(tree, include_self=True)


def unwrap_self(method):
    """
    Decorator for Equinox module methods. 
    Unwraps `self` before executing the method so all ties/deferred 
    parameters are resolved.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Explicitly use the data function on self
        unwrapped_self = unwrap(self)
        return method(unwrapped_self, *args, **kwargs)
    return wrapper


def as_unwrapped(tree: Union[T, PyTree[T]]) -> T:
    """Conditionally unwraps the root node if it is an `AbstractUnwrappable`.
    
    Unlike `unwrap`, this function does not recursively traverse the PyTree. 
    It only evaluates the top-level object.

    Args:
        tree: The tree or node to potentially unwrap.

    Returns:
        The unwrapped result if `tree` was an `AbstractUnwrappable`, 
        otherwise returns the original `tree` unmodified.
    """
    if isinstance(tree, AbstractUnwrappable):
        return tree.unwrap()
    return tree


class AbstractWrappable(eqx.Module, Generic[T]):
    """An abstract class representing a PyTree node capable of wrapping another tree.

    This interface is the counterpart to `parax.AbstractUnwrappable`. It is 
    typically used to define how an object should reconstruct or "re-wrap" a 
    tree that was previously unwrapped.
    """

    @abstractmethod
    def wrap(self, tree: T) -> Self:
        """Wraps the provided tree inside this node's structure.

        Args:
            tree: The unwrapped tree or raw value to be wrapped.

        Returns:
            A new instance of this node containing the wrapped tree.
        """
        pass


def is_wrappable(x: Any) -> TypeGuard[AbstractWrappable]:
    """Checks if a given object is a wrappable node.

    Args:
        x: The object to check.

    Returns:
        True if `x` is an instance of `AbstractWrappable`, False otherwise.
    """
    return isinstance(x, AbstractWrappable)


def wrap(template_tree: PyTree, unwrapped_tree: PyTree, only_if: Callable[[Any], bool] = None) -> PyTree:
    """Recursively resolves `AbstractWrappable` nodes to reconstruct a wrapped PyTree.

    This function maps across a `template_tree` and an `unwrapped_tree` simultaneously. 
    Wrapping is performed outside-in (top-down), perfectly inverting the inside-out 
    (bottom-up) process of `parax.unwrap`. 
    
    **Note:** This function is typically used to re-wrap a PyTree that was previously 
    unwrapped via `parax.unwrap` and `parax.AbstractUnwrappable`.

    If the `only_if` predicate is provided, the wrapping process is conditionally gated.
    The tree is searched top-down, and wrapping only triggers for subtrees that 
    satisfy the condition. Once a node satisfies `only_if`, that entire subtree 
    is fully wrapped.

    Behavior with `only_if`:
        1. If `only_if(node)` is True: The node and all of its `AbstractWrappable` 
           descendants are fully wrapped.
        2. If `only_if(node)` is False: The node itself bypasses wrapping, but the 
           search continues recursively into its children.

    Args:
        template_tree: The original (or blueprint) PyTree containing `AbstractWrappable` nodes.
        unwrapped_tree: The PyTree containing the raw, unwrapped values.
        only_if: An optional predicate function `Callable[[Any], bool]`. If provided, 
            only subtrees evaluating to True (and their descendants) are wrapped.

    Returns:
        A new PyTree where the appropriate values from `unwrapped_tree` have been 
        wrapped using the template nodes.
    """
    def _do_wrap(t_node, u_node, *, include_self: bool):
        def _map_fn(t_leaf, u_leaf):
            if not is_wrappable(t_leaf):
                return u_leaf
            
            partially_wrapped = t_leaf.wrap(u_leaf)
            return _do_wrap(t_leaf, partially_wrapped, include_self=False)

        def is_leaf(x):
            included = True if x is not t_node else include_self
            return is_wrappable(x) and included

        return jax.tree.map(_map_fn, t_node, u_node, is_leaf=is_leaf)

    def _search_and_wrap(t_node, u_node, *, include_self: bool):
        if include_self and only_if(t_node):
            return _do_wrap(t_node, u_node, include_self=True)

        def _map_fn(t_leaf, u_leaf):
            if only_if(t_leaf):
                return _do_wrap(t_leaf, u_leaf, include_self=True)

            if is_wrappable(t_leaf):
                # Bypass wrapping this node, but keep searching its children
                return _search_and_wrap(t_leaf, u_leaf, include_self=False)

            return u_leaf

        def is_leaf(x):
            included = True if x is not t_node else include_self
            return (is_wrappable(x) or only_if(x)) and included

        return jax.tree.map(_map_fn, t_node, u_node, is_leaf=is_leaf)

    if only_if is None:
        return _do_wrap(template_tree, unwrapped_tree, include_self=True)
    return _search_and_wrap(template_tree, unwrapped_tree, include_self=True)

class Parameterize(AbstractUnwrappable[T]):
    """
    Unwrap into an arbitrary object by calling a function with arguments.
    
    Useful for injecting dynamic generation (like neural network outputs 
    or complex parametrizations) into an otherwise static PyTree structure.
    """

    fn: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, fn: Callable, *args: Any, **kwargs: Any):
        """
        Args:
            fn: The callable to execute upon unwrapping.
            *args: Positional arguments to pass to `fn`.
            **kwargs: Keyword arguments to pass to `fn`.
        """
        self.fn = fn
        self.args = tuple(args)
        self.kwargs = kwargs

    def unwrap(self) -> T:
        return self.fn(*self.args, **self.kwargs)


class Apply(AbstractUnwrappable[T]):
    """
    Unwrap a PyTree by applying a function to its array-like leaves.
    
    **Corner Case Note:** This relies on `eqx.is_array_like`. Non-array 
    leaves (e.g., strings, standard integers, metadata) inside `tree` 
    will be bypassed and left intact, while any array-like objects
    (including booleans) while be mapped.
    """
    
    fn: Callable[..., T]
    tree: T
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, fn: Callable, tree: T, *args: Any, **kwargs: Any):
        """
        Args:
            tree: The target PyTree to map the computation over.
            fn: The function to apply to each array-like leaf in the tree.
            *args: Positional arguments passed to `fn` after the leaf.
            **kwargs: Keyword arguments passed to `fn`.
        """
        self.tree = tree
        self.fn = fn
        self.args = tuple(args)
        self.kwargs = kwargs

    def unwrap(self) -> T:
        def _map_fn(x):
            if not eqx.is_array_like(x):
                return x
            
            return self.fn(x, *self.args, **self.kwargs)

        return jax.tree.map(_map_fn, self.tree, is_leaf=eqx.is_array_like)


class Freeze(AbstractUnwrappable[T], AbstractWrappable[T], AbstractConstant[T]):
    """
    Applies `jax.lax.stop_gradient` to all array-like leaves before unwrapping.

    Implements the `AbstractConstant` interface so it can be filtered out 
    during optimization partitioning.

    **Corner Case Note:** The `__init__` automatically prevents double-wrapping. 
    If you pass a `Freeze` object into `Freeze`, it safely absorbs it rather 
    than nesting them.
    """
    tree: T

    def __init__(self, tree: T):
        """
        Args:
            tree: The PyTree to freeze.
        """
        if isinstance(tree, Freeze):
            tree = tree.tree
        self.tree = tree

    def free(self) -> T:
        return self.tree

    def unwrap(self) -> T:
        differentiable, static = eqx.partition(self.tree, eqx.is_array_like)
        return eqx.combine(jax.lax.stop_gradient(differentiable), static)
    
    def wrap(self, tree: T) -> Self:
        return Freeze(tree)
    

class _TiePlaceholder(eqx.Module):
    """A static, empty marker used to hide parameters from optimizers."""
    pass


class Tie(AbstractUnwrappable):
    """A wrapper that ties subtrees/parameters together.

    Upon initialization, any tied sources are replaced with placeholders.
    Then, during unwrap, values are fetched from the target tree and injected
    into the source tree.
    
    Note that if when tieing and existing Tie, the paths referring
    to the underlying, untied model.

    Attributes:
        tree: The underlying Equinox module or PyTree.
        ties: A static tuple of parameter ties, formatted as 
            `(target_extractor, source_extractor, tie_fn)`.

    Example:
        >>> class Gaussian(eqx.Module):
        ...     mu: jax.Array
        ...     sigma: jax.Array
        ...
        >>> model = Gaussian(mu=jnp.array(1.0), sigma=jnp.array(1.0))
        >>> 
        >>> # Tie sigma to always be 2x mu (`tie_fn` can also be left out for identity)
        >>> tied_model = Tie(
        ...     tree=model,
        ...     target=lambda m: m.sigma,
        ...     source=lambda m: m.mu,
        ...     tie_fn=lambda mu: mu * 2.0
        ... )
        >>>
        >>> # Optimizers will now only see `mu`
        >>> opt_state = optax.sgd(0.1).init(eqx.filter(tied_model, eqx.is_inexact_array))
        >>> 
        >>> # Unwrapping resolves the tie dynamically
        >>> active_model = unwrap(tied_model)
        >>> print(active_model.sigma) # Output: 2.0
    """
    tree: Any
    ties: tuple = eqx.field(static=True)
    
    def __init__(
        self, 
        tree: Any, 
        target: Callable[[Any], Any], 
        source: Callable[[Any], Any], 
        tie_fn: Callable[[Any], Any] = lambda x: x
    ):
        """Initializes the Tie wrapper and strips the target parameter.

        Args:
            tree: The root PyTree or Equinox module to wrap.
            target: A callable (lens) that extracts the parameter to be replaced 
                (e.g., `lambda m: m.layer.weight`).
            source: A callable (lens) that extracts the parameter to draw values 
                from (e.g., `lambda m: m.layer.bias`).
            tie_fn: An optional transformation function applied to the source 
                parameter before injecting it into the target. Defaults to the 
                identity function.
        """
        base_tree = tree.tree if isinstance(tree, Tie) else tree
        stripped_tree = eqx.tree_at(target, base_tree, replace_fn=lambda _x: _TiePlaceholder())
        new_tie = (target, source, tie_fn)
        if isinstance(tree, Tie):
            self.ties = tree.ties + (new_tie,)
        else:
            self.ties = (new_tie,)
            
        self.tree = stripped_tree

    def unwrap(self) -> Any:
        """Evaluates and resolves all parameter ties.

        Iterates through all registered ties, extracts the source values, applies 
        their respective transformation functions, and re-injects them into the 
        active PyTree before passing the tree to the standard `unwrap` function.

        Returns:
            The fully unwrapped PyTree with all target parameters resolved.
        """
        current_tree = self.tree
        for get_target, get_source, tie_fn in self.ties:
            source_val = get_source(current_tree)
            tied_val = tie_fn(source_val)
            current_tree = eqx.tree_at(get_target, current_tree, tied_val)
        return current_tree
    

class Static(AbstractUnwrappable[T], AbstractWrappable[T]):
    """
    Wraps a tree and marks it as static.
    """
    tree: T = eqx.field(static=True)

    def __init__(self, tree: T):
        """
        Args:
            tree: The PyTree to freeze.
        """
        if isinstance(tree, Static):
            tree = tree.tree
        self.tree = tree

    def unwrap(self) -> T:
        return self.tree
    
    def wrap(self, tree: T) -> Self:
        return Static(tree)
    

def as_opaque(tree: Union[T | Static[T]]) -> Union[Freeze, Static]:
    """
    Returns `tree` wrapped in either a `parax.Static` or `parax.Freeze` module, creating one if needed.

    If `tree` is a JAX array or a structured PyTree, it is wrapped in `parax.Freeze`. 
    If `tree` is an unregistered Python object (an opaque leaf e.g. a lambda), it is wrapped in `parax.Static` 
    to safely bypass JAX transformations.

    Args:
        tree: An arbitrary PyTree, array, or Python object.

    Returns:
        A static or freeze version of the tree. If it is already static or freeze, returns it directly.
    """
    if isinstance(tree, Freeze | Static):
        return tree
    
    # Ask JAX how it views this object
    # If JAX can't unpack it, it returns a list containing exactly the object itself.
    leaves, _ = jax.tree_util.tree_flatten(tree)
    is_opaque_leaf = (len(leaves) == 1) and (leaves[0] is tree)
    
    if is_opaque_leaf and not eqx.is_array(tree):
        return Static(tree)
    return Freeze(tree)


class Annotate(AbstractUnwrappable[T], AbstractWrappable[T], AbstractAnnotated[dict]):
    """
    A wrapper to add dictionary metadata to an arbitrary PyTree.
    
    Implements `parax.annotation.AbstractAnnotated`.

    Attributes:
        tree: The wrapped tree.
        metadata: The underlying metadata.
    """
    tree: T
    metadata: dict
    
    def unwrap(self) -> T:
        return self.tree
    
    def wrap(self, other: T) -> Self:
        return Annotate(tree=other, metadata=self.metadata)
    
    
class Bound(AbstractUnwrappable[T], AbstractWrappable[T], AbstractBounded[T]):
    """
    A wrapper to attach bounds to an arbitrary PyTree.
    
    Implements `parax.bounds.AbstractBounded`.
    
    Currently this wrapper does not check to ensure the leaf nodes
    lie within the bounds.

    Attributes:
        tree: The wrapped tree.
        bounds: The tree's bounds as a tuple matching its structure.
    """
    bounds: tuple[T, T]
    tree: T
    
    def unwrap(self) -> T:
        return self.tree
    
    def wrap(self, other: T) -> Self:
        return Annotate(bounds=self.bounds, tree=other)
    
    
class Constrain(AbstractUnwrappable[T], AbstractWrappable[T], AbstractConstrainable[T]):
    """
    A wrapper to attach a `parax.constraints.AbstractConstraint` to an arbitrary PyTree.
    
    Assumes the original PyTree is unconstrained
    i.e. has array-like leaf nodes that are defined
    over the entire real number line.
    
    Implements `parax.constrainable.AbstractConstrainable`.
    
    Currently this wrapper does not check to ensure the leaf nodes
    lie within the constraints.

    Attributes:
        tree: The wrapped tree.
        constraint: The tree's constraint.
    """
    constraint: AbstractConstraint
    
    tree: T
    
    def bounds(self) -> tuple[T, T]:
        return self.constraint.bounds
    
    def constrain(self, constraint: AbstractConstraint) -> Self:
        return Constrain(constraint=constraint, tree=self.tree)
    
    def unwrap(self) -> T:
        return self.tree
    
    def wrap(self, other: T) -> Self:
        return Constrain(constraint=self.constraint, tree=other)
    
    
class Probabilize(AbstractUnwrappable[T], AbstractWrappable[T], AbstractProbabilistic[T]):
    """
    A wrapper to add a probability distribution to an arbitrary PyTree.
    
    Implements `parax.probability.AbstractProbabilistic`.

    Attributes:
        distribution: The tree's associated probability distribution.
        constraint: The tree and probability distribution's constraint. If not explicitly 
            provided during initialization, this is automatically inferred from the 
            distribution using `parax.constraints.infer_distribution_constraint`.
        tree: The wrapped tree.
    """
    distribution: AbstractDistribution
    constraint: AbstractConstraint
    tree: T

    def __init__(
        self, 
        distribution: AbstractDistribution, 
        tree: T, 
        constraint: AbstractConstraint | None = None
    ):
        """
        Args:
            distribution: The probability distribution to associate with the tree.
            tree: The PyTree to be wrapped.
            constraint: An optional explicit constraint. If `None`, it is attempted
                to automatically infer the constraint from the `distribution`.
        """
        self.distribution = distribution
        self.tree = tree
        
        if constraint is None:
            from parax.constraints import infer_distribution_constraint
            self.constraint = infer_distribution_constraint(distribution)
        else:
            self.constraint = constraint

    def bounds(self) -> tuple[T, T]:
        return self.constraint.bounds
    
    def constrain(self, constraint: AbstractConstraint) -> Self:
        return Probabilize(distribution=self.distribution, tree=self.tree, constraint=constraint)
    
    def unwrap(self) -> T:
        return self.tree
    
    def wrap(self, other: T) -> Self:
        return Probabilize(distribution=self.distribution, tree=other, constraint=self.constraint)