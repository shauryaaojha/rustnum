"""Heavy NumPy matrix-multiplication workload matching heavy_matmul.rs.

Run from PowerShell:

    python heavy_matmul.py

Tune the workload with:

    $env:SIZE="2048"
    $env:ITERATIONS="6"
    python heavy_matmul.py
"""

import os
import time

import numpy as np


def read_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error

    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


size = read_positive_int("SIZE", 8192)
iterations = read_positive_int("ITERATIONS", 12)
scale = size**0.5

rng = np.random.default_rng(42)
a = rng.standard_normal((size, size))
b = rng.standard_normal((size, size))

print(f"workload: {size}x{size}, {iterations} iterations ({iterations * 2} matmuls)")

start = time.perf_counter()
for _ in range(iterations):
    a = np.tanh((a @ b) / scale)
    b = np.tanh((b @ a) / scale)
elapsed = time.perf_counter() - start

print(f"elapsed: {elapsed:.3f}s")
print(f"checksums: a.mean() = {a.mean():.9f}, b.mean() = {b.mean():.9f}")
