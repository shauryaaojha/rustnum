"""Tests for the rustnum Python bindings. Ground truth comes from NumPy.

Run after `maturin develop` (or `pip install` of the built wheel):

    python -m pytest rustnum-py/tests/test_basic.py
"""

import math

import numpy as np
import pytest

import rustnum_py as rn


def same(got, want, rtol=1e-12, atol=1e-12):
    got = np.asarray(got)
    want = np.asarray(want)
    assert got.shape == want.shape, (got.shape, want.shape)
    assert got.dtype == want.dtype, (got.dtype, want.dtype)
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


def test_creation_matches_numpy():
    same(rn.zeros((2, 3)), np.zeros((2, 3)))
    same(rn.ones(4), np.ones(4))
    same(rn.full((2, 2), 7), np.full((2, 2), 7))
    same(rn.full((2,), 2.5), np.full((2,), 2.5))
    same(rn.eye(3), np.eye(3))
    same(rn.arange(5), np.arange(5))
    same(rn.arange(1, 10, 3), np.arange(1, 10, 3))
    same(rn.arange(0.0, 1.0, 0.25), np.arange(0.0, 1.0, 0.25))
    same(rn.linspace(0, 1, 5), np.linspace(0, 1, 5))
    same(rn.zeros(3, dtype="float32"), np.zeros(3, dtype=np.float32))
    same(rn.ones(2, dtype="bool"), np.ones(2, dtype=bool))


def test_array_from_python_objects():
    same(rn.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]]))
    same(rn.array([1.5, 2]), np.array([1.5, 2]))
    same(rn.array([True, False]), np.array([True, False]))
    same(rn.array([1, 2], dtype="float32"), np.array([1, 2], dtype=np.float32))
    same(rn.asarray(np.arange(6.0).reshape(2, 3)), np.arange(6.0).reshape(2, 3))
    with pytest.raises(ValueError):
        rn.array([[1, 2], [3]])


def test_metadata():
    a = rn.zeros((2, 3, 4))
    assert a.shape == (2, 3, 4)
    assert a.ndim == 3
    assert a.size == 24
    assert a.dtype == "float64"
    assert len(a) == 2
    assert a.T.shape == (4, 3, 2)


def test_arithmetic_and_promotion():
    a = rn.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[1.0, 2.0], [3.0, 4.0]])
    same(a + a, b + b)
    same(a - 1, b - 1)
    same(2 * a, 2 * b)
    same(1 / a, 1 / b)
    same(a / a, b / b)
    same(a**2, b**2)
    same(-a, -b)
    same(abs(rn.array([-1.5, 2.0])), np.abs(np.array([-1.5, 2.0])))
    same(rn.array([1, 2, 3]) + 1, np.array([1, 2, 3]) + 1)
    same(rn.array([1, 2, 3]) + 1.5, np.array([1, 2, 3]) + 1.5)
    same(rn.array([1, 2, 3]) / 2, np.array([1, 2, 3]) / 2)
    same(rn.array([1.0], dtype="float32") + 1.5, np.array([1.0], dtype=np.float32) + 1.5)
    same(rn.array([1, 2]) + rn.array([0.5, 0.5]), np.array([1, 2]) + np.array([0.5, 0.5]))


def test_broadcasting():
    a = rn.array([[1.0], [2.0], [3.0]])
    b = rn.array([10.0, 20.0])
    same(a + b, np.array([[1.0], [2.0], [3.0]]) + np.array([10.0, 20.0]))


def test_comparisons_and_logic():
    a = rn.array([1, 5, 3])
    same(a > 2, np.array([1, 5, 3]) > 2)
    same(a == 5, np.array([1, 5, 3]) == 5)
    same((a > 2) & (a < 5), (np.array([1, 5, 3]) > 2) & (np.array([1, 5, 3]) < 5))
    same(~(a > 2), ~(np.array([1, 5, 3]) > 2))


def test_matmul_and_dot():
    x = np.arange(6.0).reshape(2, 3)
    y = np.arange(12.0).reshape(3, 4)
    same(rn.array(x) @ rn.array(y), x @ y)
    same(rn.matmul(rn.array(x), rn.array(y)), x @ y)
    same(rn.dot(rn.array(x), rn.array(y)), x.dot(y))
    v = np.array([1.0, 2.0, 3.0])
    assert rn.dot(rn.array(v), rn.array(v)).tolist() == 14.0
    same(rn.array(x.astype(np.float32)) @ rn.array(y.astype(np.float32)), (x @ y).astype(np.float32), rtol=1e-6)


def test_reductions():
    data = np.arange(24.0).reshape(2, 3, 4) * 0.5 - 3
    a = rn.array(data)
    assert a.sum() == pytest.approx(data.sum())
    assert a.mean() == pytest.approx(data.mean())
    assert a.std() == pytest.approx(data.std())
    assert a.var() == pytest.approx(data.var())
    assert a.min() == data.min()
    assert a.max() == data.max()
    assert a.argmax() == data.argmax()
    assert a.argmin() == data.argmin()
    for axis in range(3):
        same(a.sum(axis=axis), data.sum(axis=axis))
        same(a.mean(axis=axis), data.mean(axis=axis))
        same(a.std(axis=axis), data.std(axis=axis))
        same(a.max(axis=axis), data.max(axis=axis))
        same(a.argmin(axis=axis), data.argmin(axis=axis))
    ints = rn.array([1, 2, 3, 4])
    assert ints.sum() == 10 and isinstance(ints.sum(), int)
    assert ints.mean() == 2.5
    assert rn.array([True, False, True]).sum() == 2
    assert rn.array([True, False]).any() and not rn.array([True, False]).all()
    same(rn.array([1.0, 2.0, 3.0]).cumsum(), np.cumsum([1.0, 2.0, 3.0]))


def test_indexing():
    data = np.arange(24.0).reshape(2, 3, 4)
    a = rn.array(data)
    assert a[1, 2, 3] == data[1, 2, 3]
    same(a[1], data[1])
    same(a[:, 1], data[:, 1])
    same(a[::-1, 1:, ::2], data[::-1, 1:, ::2])
    same(a[..., 0], data[..., 0])
    same(a[None, 0], data[None, 0])
    same(a[-1], data[-1])
    with pytest.raises(IndexError):
        a[2]
    with pytest.raises(IndexError):
        a[0, 3]


def test_fancy_indexing_matches_numpy():
    data = np.arange(60.0).reshape(3, 4, 5)
    a = rn.array(data)
    same(a[[0, 2], [1, 3]], data[[0, 2], [1, 3]])
    same(a[[0, 2], :, [1, 4]], data[[0, 2], :, [1, 4]])
    same(a[:, [1, 3], [0, 4]], data[:, [1, 3], [0, 4]])
    same(a[0, :, [1, 2]], data[0, :, [1, 2]])
    same(a[np.array([[0], [2]]), np.array([[0, 1, 3]])], data[np.array([[0], [2]]), np.array([[0, 1, 3]])])
    same(a[:, np.array([True, False, True, True])], data[:, np.array([True, False, True, True])])
    same(a[[-1, 0], [-1, -2]], data[[-1, 0], [-1, -2]])


def test_setitem():
    data = np.zeros((3, 4))
    a = rn.zeros((3, 4))
    a[1] = 5.0
    data[1] = 5.0
    a[:, 2] = rn.array([1.0, 2.0, 3.0])
    data[:, 2] = [1.0, 2.0, 3.0]
    a[0, 0] = 9
    data[0, 0] = 9
    a[[0, 2], [1, 3]] = -1.0
    data[[0, 2], [1, 3]] = -1.0
    same(a, data)


def test_shape_manipulation():
    data = np.arange(12.0)
    a = rn.array(data)
    same(a.reshape(3, 4), data.reshape(3, 4))
    same(a.reshape((2, 6)), data.reshape(2, 6))
    same(a.reshape(-1, 6), data.reshape(-1, 6))
    same(a.reshape(3, 4).T, data.reshape(3, 4).T)
    same(a.reshape(3, 4).transpose(), data.reshape(3, 4).T)
    same(a.reshape(3, 4).ravel(), data)
    same(a.astype("int64"), data.astype(np.int64))
    assert a.reshape(3, 4).tolist() == data.reshape(3, 4).tolist()
    same(rn.concatenate([a, a]), np.concatenate([data, data]))
    same(rn.stack([a, a], axis=1), np.stack([data, data], axis=1))


def test_elementwise_functions():
    x = np.linspace(0.1, 2.0, 7)
    a = rn.array(x)
    for name in ["exp", "log", "sqrt", "sin", "cos", "tan", "tanh", "floor", "ceil"]:
        same(getattr(rn, name)(a), getattr(np, name)(x), rtol=1e-12)
    same(rn.maximum(a, 1.0), np.maximum(x, 1.0))
    same(rn.minimum(a, rn.array(x[::-1].copy())), np.minimum(x, x[::-1]))
    same(rn.where(a > 1.0, a, 0.0), np.where(x > 1.0, x, 0.0))
    same(a.clip(0.5, 1.5), np.clip(x, 0.5, 1.5))


def test_linalg():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((4, 4)) + 4 * np.eye(4)
    b = rng.standard_normal(4)
    same(rn.linalg.inv(rn.array(m)), np.linalg.inv(m), rtol=1e-10, atol=1e-10)
    same(rn.linalg.solve(rn.array(m), rn.array(b)), np.linalg.solve(m, b), rtol=1e-10, atol=1e-10)
    assert rn.linalg.det(rn.array(m)) == pytest.approx(np.linalg.det(m))
    assert rn.linalg.norm(rn.array(m)) == pytest.approx(np.linalg.norm(m))
    u, s, vt = rn.linalg.svd(rn.array(m))
    np.testing.assert_allclose(np.asarray(u) @ np.diag(np.asarray(s)) @ np.asarray(vt), m, atol=1e-10)
    spd = m @ m.T
    w, v = rn.linalg.eigh(rn.array(spd))
    np.testing.assert_allclose(np.asarray(w), np.linalg.eigvalsh(spd), rtol=1e-10)
    q, r = rn.linalg.qr(rn.array(m))
    np.testing.assert_allclose(np.asarray(q) @ np.asarray(r), m, atol=1e-10)
    same(rn.linalg.cholesky(rn.array(spd)), np.linalg.cholesky(spd), rtol=1e-10, atol=1e-10)
    with pytest.raises(ValueError):
        rn.linalg.inv(rn.array([[1.0, 2.0], [2.0, 4.0]]))


def test_random():
    rn.random.seed(1)
    a = rn.random.rand(1000)
    assert a.shape == (1000,)
    assert 0.0 <= a.min() and a.max() < 1.0
    rn.random.seed(1)
    same(rn.random.rand(1000), a)
    n = rn.random.randn(20000)
    assert abs(n.mean()) < 0.05 and abs(n.std() - 1.0) < 0.05
    r = rn.random.randint(0, 10, size=100)
    assert r.dtype == "int64" and r.min() >= 0 and r.max() < 10
    assert rn.random.rand(2, 3).shape == (2, 3)
    assert rn.random.normal(5.0, 0.1, size=(10,)).shape == (10,)


def test_numpy_asarray_is_zero_copy():
    a = rn.array(np.arange(1000.0))
    v = np.asarray(a)
    assert v.dtype == np.float64 and v.shape == (1000,)
    assert not v.flags.owndata and not v.flags.writeable
    assert v.ctypes.data == np.asarray(a).ctypes.data  # the same memory every time
    np.testing.assert_array_equal(v, np.arange(1000.0))
    for dtype, np_dtype in [("float32", np.float32), ("int64", np.int64), ("bool", np.bool_)]:
        x = rn.ones((2, 3), dtype=dtype)
        y = np.asarray(x)
        assert y.dtype == np_dtype and y.shape == (2, 3) and not y.flags.owndata
    t = rn.array(np.arange(6.0).reshape(2, 3)).T  # a transposed view is made contiguous first
    np.testing.assert_array_equal(np.asarray(t), np.arange(6.0).reshape(2, 3).T)
    del a
    np.testing.assert_array_equal(v, np.arange(1000.0))  # the view keeps the array alive


def test_repr_and_str():
    a = rn.array([[1.0, 2.5], [3.0, 4.0]])
    assert str(a) == str(np.array([[1.0, 2.5], [3.0, 4.0]]))
    assert "dtype=float64" in repr(a)


def test_scalars_and_truthiness():
    a = rn.array([2.5])
    assert float(a) == 2.5
    assert bool(a)
    with pytest.raises(ValueError):
        bool(rn.array([1, 2]))
    assert math.isnan(float(rn.array([float("nan")])))
