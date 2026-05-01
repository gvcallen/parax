from typing import Callable, Any, ClassVar

from jaxtyping import PyTree
import optimistix as optx

from parax.optimize.minimize import AbstractMinimizer, MinimizePayload

class OptimistixMinimise(AbstractMinimizer):
    """
    An optimizer that wraps :func:`optimistix.minimise`.
    """
    solver: optx.AbstractMinimiser
    supports_bounds: ClassVar[bool] = False

    def minimize(self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: PyTree = None,
        lower: PyTree | None = None,
        upper: PyTree | None = None,
        has_aux: bool = False,
        max_steps: int = 1024,
        **kwargs
    ) -> tuple[MinimizePayload, PyTree]:
        if lower is not None or upper is not None:
            raise ValueError("Optimistix minimise does not support physical bounds.")

        sol = optx.minimise(
            fn, 
            self.solver, 
            y0, 
            args, 
            has_aux=has_aux, 
            max_steps=max_steps, 
            **kwargs
        )

        opt_y = sol.value
        metrics = sol
        
        if has_aux:
            fn_value, aux = fn(opt_y, args)
        else:
            fn_value = fn(opt_y, args)
            aux = None

        payload = MinimizePayload(y=opt_y, fn_value=fn_value, aux=aux)
        return payload, metrics