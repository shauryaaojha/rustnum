#!/usr/bin/env python3
"""Comparable NumPy microbenchmarks; prints a Markdown table of median timings."""

from __future__ import annotations

import statistics
import timeit

import numpy as np


REPEATS = 7


def median_ms(statement: str, setup: dict[str, object], number: int = 1) -> float:
    timer = timeit.Timer(statement, globals=setup)
    samples = timer.repeat(repeat=REPEATS, number=number)
    return statistics.median(samples) * 1_000 / number


def main() -> None:
    rng = np.random.default_rng(42)
    million = rng.standard_normal(1_000_000)
    ten_million = rng.standard_normal(10_000_000)
    matrix = rng.standard_normal((1_000, 1_000))
    sort_input = rng.standard_normal(1_000_000)
    sort_1e7_input = rng.standard_normal(10_000_000)
    inv_input = rng.standard_normal((200, 200)) + np.eye(200) * 250
    svd_input = rng.standard_normal((100, 100))

    cases: list[tuple[str, str, dict[str, object]]] = [
        ("add (1e6 f64, contiguous)", "a + b", {"a": million, "b": million}),
        ("add broadcast ([1000,1000] + [1000])", "a + b", {"a": matrix, "b": million[:1000]}),
        ("sum (1e7 f64)", "a.sum()", {"a": ten_million}),
        ("mean_axis(0) [1000,1000]", "a.mean(axis=0)", {"a": matrix}),
        ("transpose().to_contiguous() [1000,1000]", "np.ascontiguousarray(a.T)", {"a": matrix, "np": np}),
        ("exp (1e6 f64)", "np.exp(a)", {"a": million, "np": np}),
        ("exp (1e6 f64) simd kernel", "np.exp(a)", {"a": million, "np": np}),
        ("tanh (1e6 f64) simd kernel", "np.tanh(a)", {"a": million, "np": np}),
        ("standard_normal (4096x4096)", "rng.standard_normal((4096, 4096))", {"rng": rng}),
        ("sort (1e6 f64)", "np.sort(a)", {"a": sort_input, "np": np}),
        ("sort (1e7 f64)", "np.sort(a)", {"a": sort_1e7_input, "np": np}),
        ("inv (200x200)", "np.linalg.inv(a)", {"a": inv_input, "np": np}),
        ("svd (100x100)", "np.linalg.svd(a, full_matrices=False)", {"a": svd_input, "np": np}),
    ]
    for n in (256, 512, 1024):
        left = rng.standard_normal((n, n))
        right = rng.standard_normal((n, n))
        cases.append((f"matmul ({n}x{n} f64)", "a @ b", {"a": left, "b": right}))

    print("| benchmark | NumPy median (ms) |")
    print("| --- | ---: |")
    for name, statement, namespace in cases:
        print(f"| {name} | {median_ms(statement, namespace):.3f} |")


if __name__ == "__main__":
    main()
