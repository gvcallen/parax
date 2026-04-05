from __future__ import annotations
import operator
from typing import Any, Generic, TypeVar, Union, ParamSpec
import equinox as eqx
import jax

OpInputs = ParamSpec("P")
OpOutputs = TypeVar("OpOutputs")

def tree_op(op_fn):
    """Wraps an operator to intelligently map across PyTrees and broadcast scalars."""
    def wrapped(*args):
        if len(args) == 1:
            # Unary operations (e.g., -op)
            return jax.tree_util.tree_map(op_fn, args[0])
        
        elif len(args) == 2:
            # Binary operations (e.g., op + op, op * scalar)
            left, right = args
            
            # In JAX, a single scalar or array has exactly 1 node in its tree structure
            left_is_leaf = jax.tree_util.tree_structure(left).num_nodes == 1
            right_is_leaf = jax.tree_util.tree_structure(right).num_nodes == 1
            
            if left_is_leaf and not right_is_leaf:
                # Left is a scalar/array, broadcast it over the right PyTree
                return jax.tree_util.tree_map(lambda r: op_fn(left, r), right)
            
            elif right_is_leaf and not left_is_leaf:
                # Right is a scalar/array, broadcast it over the left PyTree
                return jax.tree_util.tree_map(lambda l: op_fn(l, right), left)
            
            else:
                # Both are trees (must have matching structures) or both are scalars
                return jax.tree_util.tree_map(op_fn, left, right)
        else:
            raise ValueError("Only unary and binary operators are supported.")
            
    return wrapped

class Operator(eqx.Module, Generic[OpInputs, OpOutputs]):
    r"""
    A composable callable for building delayed computation graphs over PyTrees.
    
    ### Overview
    This class is useful when you want to easily combine mappings over the
    same input/output space. For example, you may have a `Loss` class
    with a number of child classes (MSE, RMSE etc.) that accepts (y_true, y_pred)
    and outputs an error. By simply inheriting from `prx.Operator`,
    you can now combine loss functions into a new loss function using
    addition, subtraction etc.
    
    Semantically, it makes sense to inherit from `prx.Operator`
    if operators on your input space reflection the same operators
    on your output space. For the above example, since "adding loss functions"
    equivalently means "adding loss values", the "Operator" semantics hold.

    ### Mathematical Formulation
    This class constructs an algebra over a field for the space of mappings 
    $f: X \to Y$. If the output space $Y$ is a vector space (e.g., tensors or 
    JAX arrays), then the space of all functions mapping into $Y$ is natively 
    a vector space. 
    
    By overloading standard Python operators, this class implements point-wise 
    operations on these mathematical mappings directly:
    * Addition: $(f + g)(x) = f(x) + g(x)$
    * Scalar Multiplication: $(c \cdot f)(x) = c \cdot f(x)$
    * Element-wise Multiplication: $(f \cdot g)(x) = f(x) \cdot g(x)$

    ### Assumptions
    The combined graph is mathematically valid provided the concrete `__call__` 
    methods return compatible PyTrees whose leaves are arrays or numeric scalars. 
    Operations are automatically wrapped in JAX tree utilities, allowing safe 
    element-wise computation and scalar broadcasting across complex structures.

    ### Example
    ```python
    class Scale(Operator):
        def __call__(self, x): return {"data": x * 2.0}

    class Shift(Operator):
        def __call__(self, x): return {"data": x + 5.0}

    # 1. Build the deferred computation graph
    # Mathematically defines h(x) = Scale(x) + 10 * Shift(x)
    h = Scale() + 10 * Shift()

    # 2. Evaluate the combined mapping
    # Operations map point-wise over the dictionary keys automatically
    y = h(10.0) 
    
    # y == {"data": (10.0 * 2.0) + 10 * (10.0 + 5.0)} 
    # y == {"data": 170.0}
    ```
    """
    
    def __call__(self, *args: OpInputs.args, **kwargs: OpInputs.kwargs) -> OpOutputs:
        raise NotImplementedError("Operator nodes must implement __call__.")

    # --- Arithmetic Operators ---

    def __add__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.add))

    def __sub__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.sub))

    def __mul__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.mul))

    def __truediv__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.truediv))

    def __pow__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.pow))

    # --- Unary Operators ---
    
    def __neg__(self) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Map
        return Map(operator=self, fn=tree_op(operator.neg))

    # --- Reverse Arithmetic (for <scalar> + <Operator>) ---

    def __radd__(self, other: Any) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=other, right=self, fn=tree_op(operator.add))

    def __rsub__(self, other: Any) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=other, right=self, fn=tree_op(operator.sub))

    def __rmul__(self, other: Any) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=other, right=self, fn=tree_op(operator.mul))
        
    def __rtruediv__(self, other: Any) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=other, right=self, fn=tree_op(operator.truediv))

    # --- Comparison Operators ---

    def __gt__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.gt))

    def __lt__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.lt))
        
    def __ge__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.ge))

    def __le__(self, other: Union[Operator[OpInputs, Any], Any]) -> Operator[OpInputs, OpOutputs]:
        from parax.op import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.le))