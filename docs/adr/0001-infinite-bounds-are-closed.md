# 1. Infinite bounds are closed by default

**Status:** accepted, 2026-09-25

## Context

0.11.3 made every bound open or closed, and treated an infinite bound as always open.
So `is_outside` rejected ±∞ for every constraint except an `Interval` given ±∞
explicitly. Nothing chose that; it fell out of the convention. Before 0.11.3,
validation was inclusive and ∞ was accepted everywhere.

∞ is a meaningful value in physical models: a perfect conductor, a lossless resonator,
an open circuit. Users write these as `jnp.inf` and expect them to validate.

## Decision

**An infinite bound is closed unless asked otherwise.** `RealLine` is [−∞, ∞],
`Positive` is (0, ∞], `GreaterThan(a)` is [a, ∞], and likewise for `LessThan`,
`Negative`, `NonNegative`, `NonPositive`, and `Custom` without `closed`.

**`closed` means the same on `Interval`, `GreaterThan` and `LessThan`:** one bool for
both ends, or a `(lower, upper)` pair. So `GreaterThan(1.0, closed=False)` is (1, ∞),
and `GreaterThan(1.0, closed=(True, False))` is [1, ∞). The named half-lines and
`RealLine` keep fixed closedness; an open infinite end is spelled with `GreaterThan`,
`LessThan` or `Interval`.

**An `Interval` may have infinite ends.** Where an end is infinite, it behaves like the
matching half-line or real line (bijector and base space), keeping its own `closed`.

**A distribution's support stays open at ∞.** A prior puts no mass there, and a
sampler never produces it. This is the one place infinite bounds default to open.

**Closedness follows the bounds through a bijector.** So `Transformed(RealLine(),
Sigmoid())` is [0, 1]; for (0, 1), transform an open `Interval(-inf, inf, closed=False)`.

**`intersect` keeps closedness at an infinite bound**, as at a finite one: open wins
where both constraints share it. It returns a named class only when that class's
closedness matches.

**NaN is outside every constraint.**

Closedness stays one flag per bound; per-element closedness is #8.

## Consequences

- A value at ∞ validates but has an infinite raw value and no usable gradient, as on
  any closed bound. It suits a fixed parameter, not a free one's starting point.
- A field where ∞ is nonsense (a length, a frequency) no longer has validation
  rejecting it. It needs a finite upper bound or its own check.
- A consumer that treats every closed bound as a reachable, evaluable point must skip
  non-finite ones.
