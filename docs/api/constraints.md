A constraint describes the space a value lives in, three ways: its physical `bounds`,
a `bijector` from the unconstrained real line (raw space) for unconstrained optimizers,
and a base space (`base_bounds` and `base_bijector`) for bounded optimizers.

**Closedness.** Each bound is closed (the value may sit on it exactly) or open (the value
may only approach it), and every constraint reports which as `closed`, a
`(lower, upper)` pair. `Interval`, `GreaterThan` and `LessThan` are closed by default
and take `closed` to change that. `Positive` and `Negative` are open at 0;
`NonNegative` and `NonPositive` are closed at 0. The infinite side of a half-bounded
or unbounded constraint is open.
`is_outside` respects closedness, and `intersect` keeps the open side where two
constraints share a bound. A distribution's support is closed at an endpoint where its
density is positive and finite, so `Uniform(a, b)` gives `[a, b]`, while `LogNormal`
is open at 0.

**Raw space.** The bijector maps the real line onto the open interior, so a value
exactly on a closed bound has an infinite raw value. Starting a solver from such a
value (by nudging it inward, say) is up to the solver.

**Base space.** The base space is where a bounded optimizer works, and it reaches the
closed bounds exactly. With two finite bounds it is the unit box, mapped affinely onto
the physical bounds, so every parameter presents the same scale whatever its units.
Otherwise it is the physical space itself. For a constraint inferred from a
distribution, the base comes from the support, not the prior: a `Normal` prior gives
the real line with an identity map, while its raw space is still whitened by the prior.

```python
import jax.numpy as jnp
from parax.constraints import Interval, NonNegative, Positive, intersect

assert not Interval(0.0, 1.0).is_outside(jnp.array(0.0))
assert Interval(0.0, 1.0, closed=(False, True)).is_outside(jnp.array(0.0))
assert Positive().is_outside(jnp.array(0.0))
assert not NonNegative().is_outside(jnp.array(0.0))
assert intersect(Interval(0.0, 10.0), Positive()).closed == (False, True)
```

::: parax.constraints.AbstractConstraint

::: parax.constraints.AbstractConstrained

::: parax.constraints.AbstractConstrainable

::: parax.constraints.RealLine

::: parax.constraints.GreaterThan

::: parax.constraints.LessThan

::: parax.constraints.Interval

::: parax.constraints.Positive

::: parax.constraints.NonNegative

::: parax.constraints.Negative

::: parax.constraints.NonPositive

::: parax.constraints.Leafwise

::: parax.constraints.Custom

::: parax.constraints.tree_constraints

::: parax.constraints.tree_leafwise_constraint

::: parax.constraints.tree_constrain

::: parax.constraints.intersect