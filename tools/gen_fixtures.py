#!/usr/bin/env python3
"""Generate NumPy parity test fixtures and manifest.json for rustnum."""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "rustnum" / "tests" / "fixtures"

def main():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    cases = []

    def save_case(name, op, inputs, params, output, dtype, rtol=1e-5, atol=1e-7):
        input_files = []
        for i, inp in enumerate(inputs):
            fname = f"{name}_in{i}.npy"
            np.save(FIXTURE_DIR / fname, inp)
            input_files.append(fname)

        out_fname = f"{name}_out.npy"
        np.save(FIXTURE_DIR / out_fname, output)

        cases.append({
            "name": name,
            "op": op,
            "inputs": input_files,
            "params": params,
            "output": out_fname,
            "dtype": dtype,
            "rtol": rtol,
            "atol": atol,
        })

    # =========================================================================
    # 1. Creation ops
    # =========================================================================
    # arange
    out = np.arange(0.0, 10.0, 1.0, dtype=np.float64)
    save_case("arange_basic", "arange", [], {"start": 0.0, "stop": 10.0, "step": 1.0}, out, "float64")

    out = np.arange(0.5, 5.0, 0.5, dtype=np.float64)
    save_case("arange_step_float", "arange", [], {"start": 0.5, "stop": 5.0, "step": 0.5}, out, "float64")

    # linspace
    out = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    save_case("linspace_basic", "linspace", [], {"start": 0.0, "stop": 1.0, "num": 11}, out, "float64")

    out = np.linspace(-5.0, 5.0, 21, dtype=np.float64)
    save_case("linspace_negative", "linspace", [], {"start": -5.0, "stop": 5.0, "num": 21}, out, "float64")

    # eye
    out = np.eye(3, dtype=np.float64)
    save_case("eye_3", "eye", [], {"n": 3}, out, "float64")

    out = np.eye(5, dtype=np.float64)
    save_case("eye_5", "eye", [], {"n": 5}, out, "float64")

    # tri
    out = np.tri(3, dtype=np.float64)
    save_case("tri_3", "tri", [], {"n": 3}, out, "float64")

    out = np.tri(4, dtype=np.float64)
    save_case("tri_4", "tri", [], {"n": 4}, out, "float64")

    # =========================================================================
    # 2. Broadcasting arithmetic (+, -, *, /, %)
    # =========================================================================
    a_3x1 = np.arange(1.0, 4.0, dtype=np.float64).reshape(3, 1)
    b_1x4 = np.arange(1.0, 5.0, dtype=np.float64).reshape(1, 4)
    save_case("add_broadcast_3x1_1x4", "add", [a_3x1, b_1x4], {}, a_3x1 + b_1x4, "float64")
    save_case("subtract_broadcast_3x1_1x4", "subtract", [a_3x1, b_1x4], {}, a_3x1 - b_1x4, "float64")
    save_case("divide_broadcast_3x1_1x4", "divide", [a_3x1, b_1x4], {}, a_3x1 / b_1x4, "float64")
    save_case("modulo_broadcast_3x1_1x4", "modulo", [a_3x1, b_1x4], {}, a_3x1 % b_1x4, "float64")

    a_2x3x4 = np.arange(1.0, 25.0, dtype=np.float64).reshape(2, 3, 4)
    b_4 = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    save_case("multiply_broadcast_2x3x4_4", "multiply", [a_2x3x4, b_4], {}, a_2x3x4 * b_4, "float64")

    a_2x3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b_2x3 = np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=np.float64)
    save_case("add_same_shape", "add", [a_2x3, b_2x3], {}, a_2x3 + b_2x3, "float64")
    save_case("subtract_same_shape", "subtract", [a_2x3, b_2x3], {}, a_2x3 - b_2x3, "float64")
    save_case("multiply_same_shape", "multiply", [a_2x3, b_2x3], {}, a_2x3 * b_2x3, "float64")
    save_case("divide_same_shape", "divide", [a_2x3, b_2x3], {}, a_2x3 / b_2x3, "float64")
    save_case("modulo_same_shape", "modulo", [a_2x3, b_2x3], {}, a_2x3 % b_2x3, "float64")

    # =========================================================================
    # 3. Ufuncs
    # =========================================================================
    u_vals = np.array([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=np.float64)
    u_pos = np.array([0.1, 0.5, 1.0, 2.71828, 10.0], dtype=np.float64)
    u_trig = np.array([-np.pi / 2, -np.pi / 4, 0.0, np.pi / 4, np.pi / 2], dtype=np.float64)

    save_case("exp_1d", "exp", [u_vals], {}, np.exp(u_vals), "float64")
    save_case("log_pos", "log", [u_pos], {}, np.log(u_pos), "float64")
    save_case("sqrt_pos", "sqrt", [u_pos], {}, np.sqrt(u_pos), "float64")
    save_case("tanh_vals", "tanh", [u_vals], {}, np.tanh(u_vals), "float64")
    save_case("sin_vals", "sin", [u_trig], {}, np.sin(u_trig), "float64")
    save_case("clip_vals", "clip", [u_vals], {"lo": -1.0, "hi": 1.5}, np.clip(u_vals, -1.0, 1.5), "float64")
    save_case("floor_vals", "floor", [u_vals], {}, np.floor(u_vals), "float64")

    # Round half to even
    round_test = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    save_case("round_half_to_even", "round", [round_test], {}, np.round(round_test), "float64")

    # 2D ufunc
    m_2x3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    save_case("exp_2d", "exp", [m_2x3], {}, np.exp(m_2x3), "float64")

    # =========================================================================
    # 4. Reductions (whole array)
    # =========================================================================
    red_in = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], dtype=np.float64)
    save_case("sum_whole", "sum", [red_in], {}, np.array(np.sum(red_in)), "float64")
    save_case("mean_whole", "mean", [red_in], {}, np.array(np.mean(red_in)), "float64")
    save_case("std_whole", "std", [red_in], {}, np.array(np.std(red_in)), "float64")
    save_case("var_whole", "var", [red_in], {}, np.array(np.var(red_in)), "float64")
    save_case("min_whole", "min", [red_in], {}, np.array(np.min(red_in)), "float64")
    save_case("max_whole", "max", [red_in], {}, np.array(np.max(red_in)), "float64")
    save_case("argmax_whole", "argmax", [red_in], {}, np.int64(np.argmax(red_in)), "int64")
    save_case("cumsum_whole", "cumsum", [red_in], {}, np.cumsum(red_in), "float64")
    save_case("median_whole", "median", [red_in], {}, np.array(np.median(red_in)), "float64")
    save_case("percentile_50", "percentile", [red_in], {"q": 50.0}, np.array(np.percentile(red_in, 50.0)), "float64")
    save_case("percentile_75", "percentile", [red_in], {"q": 75.0}, np.array(np.percentile(red_in, 75.0)), "float64")

    # =========================================================================
    # 5. Reductions on each axis of a 3-D array (2, 3, 4)
    # =========================================================================
    arr_3d = np.arange(1.0, 25.0, dtype=np.float64).reshape(2, 3, 4)
    save_case("sum_axis0", "sum_axis", [arr_3d], {"axis": 0}, np.sum(arr_3d, axis=0), "float64")
    save_case("sum_axis1", "sum_axis", [arr_3d], {"axis": 1}, np.sum(arr_3d, axis=1), "float64")
    save_case("sum_axis2", "sum_axis", [arr_3d], {"axis": 2}, np.sum(arr_3d, axis=2), "float64")

    save_case("mean_axis0", "mean_axis", [arr_3d], {"axis": 0}, np.mean(arr_3d, axis=0), "float64")
    save_case("mean_axis1", "mean_axis", [arr_3d], {"axis": 1}, np.mean(arr_3d, axis=1), "float64")
    save_case("mean_axis2", "mean_axis", [arr_3d], {"axis": 2}, np.mean(arr_3d, axis=2), "float64")

    save_case("min_axis0", "min_axis", [arr_3d], {"axis": 0}, np.min(arr_3d, axis=0), "float64")
    save_case("min_axis1", "min_axis", [arr_3d], {"axis": 1}, np.min(arr_3d, axis=1), "float64")
    save_case("min_axis2", "min_axis", [arr_3d], {"axis": 2}, np.min(arr_3d, axis=2), "float64")

    save_case("max_axis0", "max_axis", [arr_3d], {"axis": 0}, np.max(arr_3d, axis=0), "float64")
    save_case("max_axis1", "max_axis", [arr_3d], {"axis": 1}, np.max(arr_3d, axis=1), "float64")
    save_case("max_axis2", "max_axis", [arr_3d], {"axis": 2}, np.max(arr_3d, axis=2), "float64")

    # =========================================================================
    # 6. Manipulation
    # =========================================================================
    save_case("reshape_1d_to_2d", "reshape", [np.arange(6.0)], {"shape": [2, 3]}, np.arange(6.0).reshape(2, 3), "float64")
    save_case("reshape_2d_to_3d", "reshape", [np.arange(12.0).reshape(3, 4)], {"shape": [2, 2, 3]}, np.arange(12.0).reshape(2, 2, 3), "float64")
    save_case("transpose_2d", "transpose", [np.arange(6.0).reshape(2, 3)], {}, np.arange(6.0).reshape(2, 3).T, "float64")
    save_case("swapaxes_3d", "swapaxes", [arr_3d], {"axis1": 0, "axis2": 2}, np.swapaxes(arr_3d, 0, 2), "float64")

    c1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    c2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    save_case("concatenate_axis0", "concatenate", [c1, c2], {"axis": 0}, np.concatenate([c1, c2], axis=0), "float64")
    save_case("concatenate_axis1", "concatenate", [c1, c2], {"axis": 1}, np.concatenate([c1, c2], axis=1), "float64")

    save_case("stack_axis0", "stack", [c1, c2], {"axis": 0}, np.stack([c1, c2], axis=0), "float64")
    save_case("stack_axis1", "stack", [c1, c2], {"axis": 1}, np.stack([c1, c2], axis=1), "float64")

    save_case("flip_axis0", "flip", [c1], {"axis": 0}, np.flip(c1, axis=0), "float64")
    save_case("flip_axis1", "flip", [c1], {"axis": 1}, np.flip(c1, axis=1), "float64")

    save_case("roll_axis0", "roll", [c1], {"shift": 1, "axis": 0}, np.roll(c1, shift=1, axis=0), "float64")
    save_case("roll_axis1", "roll", [c1], {"shift": -1, "axis": 1}, np.roll(c1, shift=-1, axis=1), "float64")

    save_case("repeat_axis0", "repeat", [c1], {"n": 2, "axis": 0}, np.repeat(c1, 2, axis=0), "float64")
    save_case("repeat_axis1", "repeat", [c1], {"n": 3, "axis": 1}, np.repeat(c1, 3, axis=1), "float64")

    save_case("tile_2d", "tile", [c1], {"reps": [2, 3]}, np.tile(c1, (2, 3)), "float64")

    unsorted = np.array([4.0, 2.0, 5.0, 1.0, 3.0], dtype=np.float64)
    save_case("sort_1d", "sort", [unsorted], {}, np.sort(unsorted), "float64")
    save_case("argsort_1d", "argsort", [unsorted], {}, np.argsort(unsorted).astype(np.int64), "int64")

    has_dups = np.array([3.0, 1.0, 2.0, 1.0, 3.0, 4.0, 2.0], dtype=np.float64)
    save_case("unique_1d", "unique", [has_dups], {}, np.unique(has_dups), "float64")

    cond = np.array([True, False, True, False, True], dtype=bool)
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    save_case("where_1d", "where", [cond, x, y], {}, np.where(cond, x, y), "float64")

    # =========================================================================
    # 7. Comparisons & Astype
    # =========================================================================
    cmp_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    cmp_b = np.array([[1.0, 0.0], [3.0, 5.0]], dtype=np.float64)
    save_case("equal_2d", "equal", [cmp_a, cmp_b], {}, (cmp_a == cmp_b), "bool")
    save_case("less_2d", "less", [cmp_a, cmp_b], {}, (cmp_a < cmp_b), "bool")

    ast_f = np.array([1.2, 3.8, -2.4, 5.0], dtype=np.float64)
    save_case("astype_f64_to_i64", "astype_f64_to_i64", [ast_f], {}, ast_f.astype(np.int64), "int64")

    ast_i = np.array([1, -4, 9, 0], dtype=np.int64)
    save_case("astype_i64_to_f64", "astype_i64_to_f64", [ast_i], {}, ast_i.astype(np.float64), "float64")

    # =========================================================================
    # 8. Slicing
    # =========================================================================
    sl_in = np.arange(10.0, dtype=np.float64)
    save_case("slice_step_neg", "slice", [sl_in], {"step": -1}, sl_in[::-1], "float64")

    # =========================================================================
    # 9. Linear Algebra (fixtures generated for future parity verification)
    # =========================================================================
    m1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    m2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    save_case("matmul_2d", "matmul", [m1, m2], {}, m1 @ m2, "float64")

    b_m1 = rng.standard_normal((2, 2, 2))
    b_m2 = rng.standard_normal((2, 2, 2))
    save_case("matmul_batched", "matmul_batched", [b_m1, b_m2], {}, b_m1 @ b_m2, "float64")

    v1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    v2 = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    save_case("dot_vectors", "dot", [v1, v2], {}, np.array(np.dot(v1, v2)), "float64")

    save_case("det_2x2", "det", [m1], {}, np.array(np.linalg.det(m1)), "float64")
    m3 = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], dtype=np.float64)
    save_case("det_3x3", "det", [m3], {}, np.array(np.linalg.det(m3)), "float64")

    save_case("inv_2x2", "inv", [m1], {}, np.linalg.inv(m1), "float64")
    save_case("inv_3x3", "inv", [m3], {}, np.linalg.inv(m3), "float64")

    rhs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    save_case("solve_system", "solve", [m3, rhs], {}, np.linalg.solve(m3, rhs), "float64")

    save_case("norm_matrix", "norm", [m1], {}, np.array(np.linalg.norm(m1)), "float64")
    save_case("norm_vector", "norm", [v1], {}, np.array(np.linalg.norm(v1)), "float64")

    # Symmetric positive definite for cholesky
    spd = np.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]], dtype=np.float64)
    save_case("cholesky_3x3", "cholesky", [spd], {}, np.linalg.cholesky(spd), "float64")

    q, r = np.linalg.qr(m3)
    save_case("qr_q_3x3", "qr_q", [m3], {}, q, "float64")
    save_case("qr_r_3x3", "qr_r", [m3], {}, r, "float64")

    sym = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    w, _ = np.linalg.eigh(sym)
    save_case("eigh_eigenvalues", "eigh_eigenvalues", [sym], {}, w, "float64")

    u, s, vt = np.linalg.svd(m1)
    save_case("svd_singular", "svd_singular", [m1], {}, s, "float64")

    m_rect = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    save_case("pinv_rect", "pinv", [m_rect], {}, np.linalg.pinv(m_rect), "float64")

    rank_mat = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    save_case("matrix_rank_mat", "matrix_rank", [rank_mat], {}, np.int64(np.linalg.matrix_rank(rank_mat)), "int64")

    save_case("outer_product", "outer", [v1, v2], {}, np.outer(v1, v2), "float64")
    save_case("trace_matrix", "trace", [m3], {}, np.array(np.trace(m3)), "float64")

    # =========================================================================
    # 10. Complex dtype (stage 1 of v0.2)
    # =========================================================================
    complex_cases(save_case, rng)

    # Write manifest.json
    manifest_path = FIXTURE_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    print(f"Generated {len(cases)} parity fixture cases in {FIXTURE_DIR}")


def complex_cases(save_case, rng):
    """complex128 / complex64 parity cases (ops prefixed `c_`)."""
    z = np.array([1 + 2j, -0.5 + 0.25j, 3 - 1j, 0.75 + 4j, -2 + 0j, 1.5 - 1.5j])
    za = np.array([[1 + 2j], [3 - 1j], [-0.5 + 0.5j]])
    zb = np.array([[2 - 1j, 0.5 + 0.5j, -1 + 0j, 0 + 3j]])
    zr = np.array([[2.0, -0.5, 4.0, 0.25]])
    tol = dict(rtol=1e-12, atol=1e-13)
    for name, f in [("add", np.add), ("sub", np.subtract), ("mul", np.multiply), ("div", np.divide)]:
        save_case(f"c_{name}_broadcast", f"c_{name}", [za, zb], {}, f(za, zb), "complex128", **tol)
    save_case("c_mul_real", "c_mul", [za, zr], {}, za * zr, "complex128", **tol)
    save_case("c_real_div_complex", "c_div", [zr, za], {}, zr / za, "complex128", **tol)
    save_case("c_add_real_scalar_row", "c_add", [zb, zr], {}, zb + zr, "complex128", **tol)
    save_case("c_neg", "c_neg", [z], {}, -z, "complex128", **tol)
    for name, f in [("exp", np.exp), ("log", np.log), ("sqrt", np.sqrt), ("sin", np.sin), ("cos", np.cos),
                    ("tan", np.tan), ("sinh", np.sinh), ("cosh", np.cosh), ("tanh", np.tanh),
                    ("square", np.square), ("conj", np.conj)]:
        save_case(f"c_{name}", f"c_{name}", [z], {}, f(z), "complex128", **tol)
    save_case("c_abs", "c_abs", [z], {}, np.abs(z), "float64", **tol)
    save_case("c_angle", "c_angle", [z], {}, np.angle(z), "float64", **tol)
    save_case("c_real", "c_real", [z], {}, z.real, "float64", **tol)
    save_case("c_imag", "c_imag", [z], {}, z.imag, "float64", **tol)
    w = np.array([0.5 + 1j, 2 + 0j, -1 + 1j, 1.5 - 0.5j, 3 + 0j, 0.25 + 0.25j])
    save_case("c_power", "c_power", [z, w], {}, np.power(z, w), "complex128", rtol=1e-11, atol=1e-12)
    save_case("c_power_int", "c_power", [z, np.full(6, 3 + 0j)], {}, np.power(z, 3), "complex128", **tol)
    m = z.reshape(2, 3)
    save_case("c_sum", "c_sum", [z], {}, np.array(z.sum()), "complex128", **tol)
    save_case("c_prod", "c_prod", [z], {}, np.array(z.prod()), "complex128", **tol)
    save_case("c_mean", "c_mean", [z], {}, np.array(z.mean()), "complex128", **tol)
    save_case("c_sum_axis0", "c_sum_axis", [m], {"axis": 0}, m.sum(axis=0), "complex128", **tol)
    save_case("c_sum_axis1", "c_sum_axis", [m], {"axis": 1}, m.sum(axis=1), "complex128", **tol)
    save_case("c_mean_axis1", "c_mean_axis", [m], {"axis": 1}, m.mean(axis=1), "complex128", **tol)
    save_case("c_cumsum", "c_cumsum", [z], {}, np.cumsum(z), "complex128", **tol)
    save_case("c_max", "c_max", [z], {}, np.array(z.max()), "complex128", **tol)
    save_case("c_min", "c_min", [z], {}, np.array(z.min()), "complex128", **tol)
    save_case("c_sort", "c_sort", [z], {}, np.sort(z), "complex128", **tol)
    save_case("c_argsort", "c_argsort", [z], {}, np.argsort(z).astype(np.int64), "int64")
    a34 = (rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4)))
    b42 = (rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2)))
    save_case("c_matmul", "c_matmul", [a34, b42], {}, a34 @ b42, "complex128", rtol=1e-12, atol=1e-12)
    v1 = rng.standard_normal(5) + 1j * rng.standard_normal(5)
    v2 = rng.standard_normal(5) + 1j * rng.standard_normal(5)
    save_case("c_dot", "c_dot", [v1, v2], {}, np.array(np.dot(v1, v2)), "complex128", rtol=1e-12, atol=1e-12)
    save_case("c_vdot", "c_vdot", [v1, v2], {}, np.array(np.vdot(v1, v2)), "complex128", rtol=1e-12, atol=1e-12)
    save_case("c_matmul_vec", "c_matmul", [a34, v1[:4]], {}, a34 @ v1[:4], "complex128", rtol=1e-12, atol=1e-12)
    save_case("c_fft", "c_fft", [z], {}, np.fft.fft(z), "complex128", rtol=1e-11, atol=1e-12)
    save_case("c_ifft", "c_ifft", [z], {}, np.fft.ifft(z), "complex128", rtol=1e-11, atol=1e-12)
    xr = rng.standard_normal(7)
    save_case("c_fft_real_input", "c_fft", [xr], {}, np.fft.fft(xr), "complex128", rtol=1e-11, atol=1e-12)
    save_case("c_rfft", "c_rfft", [xr], {}, np.fft.rfft(xr), "complex128", rtol=1e-11, atol=1e-12)
    h = np.fft.rfft(xr)
    save_case("c_irfft_odd", "c_irfft", [h], {"n": 7}, np.fft.irfft(h, 7), "float64", rtol=1e-11, atol=1e-12)
    save_case("c_irfft_default", "c_irfft", [h], {}, np.fft.irfft(h), "float64", rtol=1e-11, atol=1e-12)
    m2 = rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))
    save_case("c_fft2", "c_fft2", [m2], {}, np.fft.fft2(m2), "complex128", rtol=1e-11, atol=1e-12)
    save_case("c_ifft2", "c_ifft2", [m2], {}, np.fft.ifft2(m2), "complex128", rtol=1e-11, atol=1e-12)
    z64 = z.astype(np.complex64)
    save_case("c64_fft", "c64_fft", [z64], {}, np.fft.fft(z64), "complex64", rtol=2e-5, atol=2e-5)
    save_case("c64_mul", "c64_mul", [z64, z64[::-1].copy()], {}, z64 * z64[::-1], "complex64", rtol=2e-6, atol=1e-6)
    save_case("c_astype_c128_to_c64", "c_to_c64", [z], {}, z.astype(np.complex64), "complex64", rtol=1e-7, atol=1e-7)
    save_case("c_astype_f64_to_c128", "c_from_real", [xr], {}, xr.astype(np.complex128), "complex128", **tol)
    save_case("c_isnan", "c_isnan", [np.array([np.nan + 1j, 1 + 0j, 2 + np.nan * 1j])], {},
              np.isnan(np.array([np.nan + 1j, 1 + 0j, 2 + np.nan * 1j])), "bool")
    strings = {
        "mixed": z,
        "round": np.array([1.5 + 2j, 3 - 4j]),
        "sci": np.array([1e-5 + 1j, 1 + 1e5j]),
        "matrix": np.array([[1 + 1j, 2 - 2j], [0.5j, -0.0 - 0j]]),
        "special": np.array([np.nan + 1j, 1 + np.inf * 1j, complex(1, -np.nan)]),
        "c64": z.astype(np.complex64),
        "third": np.array([0.1 + 0.2j, 1 / 3 + 2 / 3j]),
    }
    for k, arr in strings.items():
        save_case(f"c_str_{k}", "c_str", [arr], {"str": str(arr)}, np.array(0), "str")


if __name__ == "__main__":
    main()
