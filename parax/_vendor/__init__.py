"""
Vendored copies of distreqx classes, used as fallbacks.

These are copied verbatim from distreqx (https://github.com/lockwo/distreqx,
Apache License 2.0), with their relative imports rewritten to import the
abstract base classes from the installed distreqx.

They exist so that parax works against any distreqx: the latest release, the
unreleased upstream main, or gvcallen's fork. `parax.bijectors` and
`parax.distributions` prefer the installed distreqx's implementation and fall
back to these copies when it does not provide one.

Do not import from this package directly; use `parax.bijectors` and
`parax.distributions` instead.
"""
