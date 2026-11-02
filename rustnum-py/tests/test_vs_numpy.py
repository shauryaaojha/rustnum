"""Randomised comparison of rustnum_py against NumPy (skipped when numpy is missing)."""
import numpy as np
import pytest

import rustnum_py as rn

rng = np.random.default_rng(0)


def close(got, want, rtol=1e-9, atol=1e-12):
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, (got.shape, want.shape)
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


@pytest.mark.parametrize("shape", [(7,), (3, 4), (2, 3, 4)])
def test_reductions_all_axes(shape):
    x = rng.standard_normal(shape)
    r = rn.array(x)
    for name in ["sum", "mean", "std", "var", "min", "max", "prod"]:
        close(getattr(r, name)(), getattr(x, name)())
        for ax in range(len(shape)):
            close(getattr(r, name)(axis=ax), getattr(x, name)(axis=ax))
    close(r.argmax(), x.argmax())
    close(r.cumsum(), x.cumsum())


def test_binary_ops_with_broadcasting():
    a = rng.standard_normal((4, 1, 3))
    b = rng.standard_normal((5, 1))
    ra, rb = rn.array(a), rn.array(b)
    close(ra + rb, a + b)
    close(ra - rb, a - b)
    close(ra * rb, a * b)
    close(ra / rb, a / b)
    close(2.0 - ra, 2.0 - a)
    close(ra < rb, a < b)


def test_matmul_random():
    a = rng.standard_normal((5, 6))
    b = rng.standard_normal((6, 3))
    close(rn.array(a) @ rn.array(b), a @ b, rtol=1e-12)
    a3 = rng.standard_normal((2, 5, 6))
    close(rn.matmul(rn.array(a3), rn.array(b)), a3 @ b, rtol=1e-12)


def test_linalg_random():
    a = rng.standard_normal((5, 5)) + 5 * np.eye(5)
    b = rng.standard_normal(5)
    close(rn.linalg.solve(rn.array(a), rn.array(b)), np.linalg.solve(a, b), rtol=1e-9)
    close(rn.linalg.inv(rn.array(a)), np.linalg.inv(a), rtol=1e-9)
    assert abs(rn.linalg.det(rn.array(a)) - np.linalg.det(a)) < 1e-6 * abs(np.linalg.det(a))
    close(rn.linalg.norm(rn.array(a)), np.linalg.norm(a))
    spd = a @ a.T
    close(rn.linalg.cholesky(rn.array(spd)), np.linalg.cholesky(spd), rtol=1e-9)


def test_float32_stays_float32():
    x = rng.standard_normal((3, 3)).astype(np.float32)
    r = rn.array(x)
    assert np.asarray(r).dtype == np.float32
    assert np.asarray(r + r).dtype == np.float32
    close(r.sum(), x.sum(), rtol=1e-5)


def test_slicing_matches():
    x = rng.standard_normal((6, 5))
    r = rn.array(x)
    for key in [(slice(1, 5), slice(None, None, 2)), (slice(None), 2), (-1,), (slice(None, None, -1), slice(1, 3))]:
        close(r[key], x[key])
