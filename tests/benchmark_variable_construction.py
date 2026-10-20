"""CPU constructor timings; run with `python -P tests/benchmark_variable_construction.py`."""

import statistics
import time

import jax

import parax.distributions as dists
from parax.constraints import Interval
from parax.variables import Constrained, Random


def measure(factory, *, eager=False, repetitions=100):
    with jax.disable_jit(eager):
        start = time.perf_counter()
        factory().raw_value.block_until_ready()
        first = (time.perf_counter() - start) * 1000
        samples = []
        for _ in range(repetitions):
            start = time.perf_counter()
            factory().raw_value.block_until_ready()
            samples.append((time.perf_counter() - start) * 1000)
    return first, statistics.median(samples)


def main():
    cases = {
        "Random(TruncatedNormal)": lambda: Random(dists.TruncatedNormal(1.0, 0.2, 0.0, 2.0)),
        "Random(Uniform, value=0)": lambda: Random(dists.Uniform(0.0, 50.0), value=0.0),
        "Constrained(Interval)": lambda: Constrained(Interval(0.0, 10.0), value=1.0),
    }
    print("case | eager first | compiled first | eager warm | compiled warm (ms)")
    for name, factory in cases.items():
        jax.clear_caches()
        eager_first, eager_warm = measure(factory, eager=True)
        jax.clear_caches()
        compiled_first, compiled_warm = measure(factory)
        print(f"{name} | {eager_first:.2f} | {compiled_first:.2f} | "
              f"{eager_warm:.3f} | {compiled_warm:.3f}")


if __name__ == "__main__":
    main()
