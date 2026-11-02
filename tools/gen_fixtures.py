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

    # =========================================================================
    # 11. float16 / bfloat16 (stage 2 of v0.2)
    # =========================================================================
    half_cases(save_case, rng)

    # =========================================================================
    # 12. Advanced indexing (stage 3 of v0.2)
    # =========================================================================
    fancy_cases(save_case, rng)

    # =========================================================================
    # 13. Masked arrays (stage 4 of v0.2)
    # =========================================================================
    ma_cases(save_case, rng)

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


def _bf16_reference_bits(v):
    """Exact round-to-nearest-even of a Python float to bfloat16 bits, using rational arithmetic."""
    import math
    from fractions import Fraction
    if math.isnan(v):
        return 0x7FC0
    sign = 0x8000 if math.copysign(1.0, v) < 0 else 0
    a = abs(v)
    if math.isinf(a):
        return sign | 0x7F80
    if a == 0.0:
        return sign
    _, ex = math.frexp(a)
    e = max(ex - 1, -126)
    unit = Fraction(2) ** (e - 7)
    q = Fraction(a) / unit
    n = q.numerator // q.denominator
    r = q - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2 == 1):
        n += 1
    val = n * unit
    if val >= Fraction(2) ** 128:
        return sign | 0x7F80
    f32_bits = int(np.float32(float(val)).view(np.uint32))
    return sign | (f32_bits >> 16)


def half_cases(save_case, rng):
    """float16 / bfloat16 parity cases (ops prefixed `h_` for float16, `bh_` for bfloat16)."""
    import ml_dtypes

    bf16 = ml_dtypes.bfloat16
    r = np.random.default_rng(7)

    # --- exhaustive: every float16 bit pattern -------------------------------------------
    h_all = np.arange(65536, dtype=np.uint16).view(np.float16)
    as32 = h_all.astype(np.float32)
    nan = np.isnan(as32)
    bits32 = as32.view(np.uint32).astype(np.int64)
    bits32[nan] = 0x7FC00000
    save_case("h_all_to_f32", "h_to_f32_bits", [h_all], {}, bits32, "int64")
    back = as32.astype(np.float16).view(np.uint16).astype(np.int64)
    back[nan] = 0x7E00
    save_case("h_all_roundtrip", "h_roundtrip_bits", [h_all], {}, back, "int64")
    as64 = h_all.astype(np.float64)
    save_case("h_all_to_f64", "h_to_f64", [h_all], {}, as64, "float64", rtol=0.0, atol=0.0)

    # --- float32 -> float16: random values, every tie, specials ---------------------------
    n = 400_000
    e = r.integers(96, 150, n).astype(np.uint32)
    m = r.integers(0, 1 << 23, n).astype(np.uint32)
    s = r.integers(0, 2, n).astype(np.uint32)
    rand_bits = (s << 31) | (e << 23) | m
    pos = np.arange(0x7C00, dtype=np.uint16).view(np.float16)
    nxt = np.nextafter(pos, np.float16(np.inf))
    mid = ((pos.astype(np.float64) + nxt.astype(np.float64)) / 2).astype(np.float32)
    mb = mid.view(np.uint32)
    ties = np.concatenate([mb, mb + 1, mb - 1, mb | 0x80000000, (mb + 1) | 0x80000000])
    specials = np.array(
        [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0x00000001,
         0x33000000, 0x33000001, 0x477FE000, 0x477FEFFF, 0x477FF000, 0x47800000, 0x38800000,
         0x387FFFFF, 0x7F7FFFFF], dtype=np.uint32)
    x32 = np.concatenate([rand_bits, ties, specials]).astype(np.uint32).view(np.float32)
    with np.errstate(all="ignore"):
        e16 = x32.astype(np.float16).view(np.uint16).astype(np.int64)
    e16[np.isnan(x32)] = 0x7E00
    save_case("h_f32_to_f16_bits", "h_f32_to_f16_bits", [x32], {}, e16, "int64")

    # --- float64 -> float16 (NumPy rounds once) --------------------------------------------
    n = 200_000
    e = r.integers(990, 1040, n).astype(np.uint64)
    m = r.integers(0, 1 << 52, n, dtype=np.uint64)
    s = r.integers(0, 2, n).astype(np.uint64)
    rand64 = (s << np.uint64(63)) | (e << np.uint64(52)) | m
    mid64 = (pos.astype(np.float64) + nxt.astype(np.float64)) / 2
    mb64 = mid64.view(np.uint64)
    ties64 = np.concatenate([mb64, mb64 + np.uint64(1), mb64 - np.uint64(1)])
    x64 = np.concatenate([rand64, ties64]).astype(np.uint64).view(np.float64)
    with np.errstate(all="ignore"):
        e16 = x64.astype(np.float16).view(np.uint16).astype(np.int64)
    e16[np.isnan(x64)] = 0x7E00
    save_case("h_f64_to_f16_bits", "h_f64_to_f16_bits", [x64], {}, e16, "int64")

    # --- arithmetic and functions ------------------------------------------------------------
    a = (r.standard_normal((6, 5)) * 30).astype(np.float16)
    b = (r.standard_normal((1, 5)) * 7).astype(np.float16)
    b[0, 2] = 0
    with np.errstate(all="ignore"):
        for name, f in [("add", np.add), ("sub", np.subtract), ("mul", np.multiply), ("div", np.divide)]:
            save_case(f"h_{name}", f"h_{name}", [a, b], {}, f(a, b), "float16", rtol=0.0, atol=0.0)
        pos_in = np.abs(a) + np.float16(0.01)
        small = (a / np.float16(8)).astype(np.float16)
        for name, f, src in [("exp", np.exp, small), ("log", np.log, pos_in), ("sqrt", np.sqrt, pos_in),
                             ("sin", np.sin, a), ("cos", np.cos, a), ("tanh", np.tanh, small),
                             ("floor", np.floor, a), ("round", np.round, a)]:
            save_case(f"h_{name}", f"h_{name}", [src], {}, f(src), "float16", rtol=2.5e-3, atol=1e-4)
        save_case("h_abs", "h_abs", [a], {}, np.abs(a), "float16", rtol=0.0, atol=0.0)
        save_case("h_neg", "h_neg", [a], {}, -a, "float16", rtol=0.0, atol=0.0)
        long = (r.standard_normal(10000) + 3).astype(np.float16)
        ones = np.ones(10000, dtype=np.float16)
        save_case("h_sum_long", "h_sum", [long], {}, np.array(long.sum()), "float16", rtol=0.0, atol=0.0)
        save_case("h_sum_ones", "h_sum", [ones], {}, np.array(ones.sum()), "float16", rtol=0.0, atol=0.0)
        save_case("h_mean", "h_mean", [long], {}, np.array(long.mean()), "float16", rtol=1e-3, atol=0.0)
        save_case("h_std", "h_std", [long], {}, np.array(long.std()), "float16", rtol=1e-3, atol=0.0)
        save_case("h_var", "h_var", [long], {}, np.array(long.var()), "float16", rtol=1e-3, atol=0.0)
        save_case("h_sum_axis0", "h_sum_axis", [a], {"axis": 0}, a.sum(axis=0), "float16", rtol=1e-3, atol=0.0)
        save_case("h_max", "h_max", [a], {}, np.array(a.max()), "float16", rtol=0.0, atol=0.0)
        save_case("h_cumsum", "h_cumsum", [a.ravel()[:10].copy()], {}, np.cumsum(a.ravel()[:10]), "float16",
                  rtol=0.0, atol=0.0)
        save_case("h_sort", "h_sort", [a.ravel()], {}, np.sort(a.ravel()), "float16", rtol=0.0, atol=0.0)
        save_case("h_astype_i64", "h_to_i64", [a], {}, a.astype(np.int64), "int64")
        save_case("h_from_f64", "h_from_f64", [a.astype(np.float64) / 3], {}, (a.astype(np.float64) / 3).astype(np.float16),
                  "float16", rtol=0.0, atol=0.0)
        special = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 65504, 6.1e-5, 5.96e-8, 0.1, 1 / 3], dtype=np.float16)
        save_case("h_isnan", "h_isnan", [special], {}, np.isnan(special), "bool")
    strings = {
        "simple": np.array([0.1, 1.5, 3.0], dtype=np.float16),
        "sci": np.array([0.1, 1.5, 65504, 1e-5], dtype=np.float16),
        "matrix": np.array([[1, 2.5], [-3, 4]], dtype=np.float16),
        "special": np.array([np.nan, np.inf, -np.inf, 0.5], dtype=np.float16),
        "third": np.array([1 / 3, 2 / 3, 100.0], dtype=np.float16),
    }
    for k, arr in strings.items():
        save_case(f"h_str_{k}", "h_str", [arr], {"str": str(arr)}, np.array(0), "str")

    # --- bfloat16 (ground truth from ml_dtypes; inputs and outputs as int64 bit patterns) -----
    bits16 = np.arange(65536, dtype=np.uint16)
    b_all = bits16.view(bf16)
    f32 = b_all.astype(np.float32)
    nan = np.isnan(f32)
    exp_bits = f32.view(np.uint32).astype(np.int64)
    exp_bits[nan] = 0x7FC00000
    save_case("bh_all_to_f32", "bh_to_f32_bits", [bits16.astype(np.int64)], {}, exp_bits, "int64")
    xs = x32[:300_000]
    with np.errstate(all="ignore"):
        eb = xs.astype(bf16).view(np.uint16).astype(np.int64)
    eb[np.isnan(xs)] = 0x7FC0
    save_case("bh_f32_to_bf16_bits", "bh_f32_to_bf16_bits", [xs], {}, eb, "int64")
    sample64 = np.concatenate([x64[:30000], r.standard_normal(10000) * 1e3, np.array([1.0 + 2**-8 + 2**-40, 1.0 + 2**-8, 3e38, 3.5e38, 1e-40, 1e-45])])
    eb64 = np.array([_bf16_reference_bits(float(v)) for v in sample64], dtype=np.int64)
    save_case("bh_f64_to_bf16_bits", "bh_f64_to_bf16_bits", [sample64], {}, eb64, "int64")
    ba = (r.standard_normal((4, 5)) * 20).astype(bf16)
    bb = (r.standard_normal((1, 5)) * 5).astype(bf16)
    with np.errstate(all="ignore"):
        for name, f in [("add", np.add), ("sub", np.subtract), ("mul", np.multiply), ("div", np.divide)]:
            out = f(ba, bb).view(np.uint16).astype(np.int64)
            save_case(f"bh_{name}", f"bh_{name}", [ba.view(np.uint16).astype(np.int64), bb.view(np.uint16).astype(np.int64)],
                      {}, out, "int64")
        out = np.exp((ba / bf16(8)).astype(bf16)).view(np.uint16).astype(np.int64)
        save_case("bh_exp", "bh_exp", [(ba / bf16(8)).astype(bf16).view(np.uint16).astype(np.int64)], {}, out, "int64")


def fancy_cases(save_case, rng):
    """Advanced (integer-array) indexing parity cases (ops prefixed `fx_`)."""
    a = np.arange(60.0).reshape(3, 4, 5)
    i64 = lambda v: np.array(v, dtype=np.int64)

    def ARR(k):
        return {"t": "arr", "i": k}

    def INT(v):
        return {"t": "int", "v": v}

    def SL(start=None, stop=None, step=1):
        return {"t": "slice", "start": start, "stop": stop, "step": step}

    NEW = {"t": "new"}

    # (name, index spec in JSON, python index tuple, extra arrays referenced by ARR(k), k >= 1)
    idx0, idx1 = i64([0, 2]), i64([1, 3])
    cases = [
        ("pair", [ARR(1), ARR(2)], lambda x, e: x[e[0], e[1]], [idx0, idx1]),
        ("axis0", [ARR(1)], lambda x, e: x[e[0]], [idx0]),
        ("axis1", [SL(), ARR(1)], lambda x, e: x[:, e[0]], [i64([1, 3, 3])]),
        ("split", [ARR(1), SL(), ARR(2)], lambda x, e: x[e[0], :, e[1]], [idx0, i64([1, 4])]),
        ("adjacent", [SL(), ARR(1), ARR(2)], lambda x, e: x[:, e[0], e[1]], [idx1, i64([0, 4])]),
        ("int_split", [INT(0), SL(), ARR(1)], lambda x, e: x[0, :, e[0]], [i64([1, 2])]),
        ("int_adjacent", [SL(), INT(1), ARR(1)], lambda x, e: x[:, 1, e[0]], [i64([0, 2, 4])]),
        ("broadcast2d", [ARR(1), ARR(2)], lambda x, e: x[e[0], e[1]], [i64([[0], [2]]), i64([[0, 1, 3]])]),
        ("negative", [ARR(1), ARR(2)], lambda x, e: x[e[0], e[1]], [i64([-1, 0]), i64([-1, -2])]),
        ("slices_mixed", [SL(1, None), ARR(1), SL(2, 4)], lambda x, e: x[1:, e[0], 2:4], [i64([0, 3])]),
        ("newaxis_between", [ARR(1), NEW, ARR(2)], lambda x, e: x[e[0], None, e[1]], [idx0, idx1]),
        ("steps", [SL(None, None, 2), ARR(1), SL(None, None, -1)], lambda x, e: x[::2, e[0], ::-1], [i64([3, 1, 1])]),
        ("index2d", [SL(), ARR(1)], lambda x, e: x[:, e[0]], [i64([[0, 1], [2, 3]])]),
        ("diagonal", [ARR(1), ARR(2), ARR(3)], lambda x, e: x[e[0], e[1], e[2]], [i64([0, 1, 2]), i64([0, 1, 2]), i64([0, 1, 2])]),
        ("empty", [ARR(1), SL(), INT(0)], lambda x, e: x[e[0], :, 0], [i64([])]),
        ("three_split", [ARR(1), INT(2), ARR(2)], lambda x, e: x[e[0], 2, e[1]], [idx0, idx1]),
        ("scalar_broadcast", [ARR(1), ARR(2)], lambda x, e: x[e[0], e[1]], [i64([[1, 2]]), i64([3])]),
    ]
    for name, spec, fn, extra in cases:
        out = fn(a, extra)
        save_case(f"fx_get_{name}", "fx_get", [a] + extra, {"index": spec}, out, "float64", rtol=0.0, atol=0.0)

    # a 1-d boolean mask inside an index tuple
    mask = np.array([True, False, True, True])
    save_case("fx_get_mask", "fx_get", [a, mask], {"index": [SL(), {"t": "mask", "i": 1}]}, a[:, mask], "float64",
              rtol=0.0, atol=0.0)

    # assignment: values are the last input
    assigns = [
        ("pair", [ARR(1), ARR(2)], lambda x, e, v: x.__setitem__((e[0], e[1]), v), [idx0, idx1], np.array([100.0, 200, 300, 400, 500])),
        ("duplicates", [SL(), ARR(1), INT(1)], lambda x, e, v: x.__setitem__((slice(None), e[0], 1), v),
         [i64([0, 0])], np.arange(6.0).reshape(3, 2) + 1000),
        ("split_scalar", [ARR(1), SL(), ARR(2)], lambda x, e, v: x.__setitem__((e[0], slice(None), e[1]), v),
         [idx0, i64([1, 4])], np.array(-1.0)),
        ("split_values", [ARR(1), SL(), ARR(2)], lambda x, e, v: x.__setitem__((e[0], slice(None), e[1]), v),
         [idx0, i64([1, 4])], np.arange(8.0).reshape(2, 4) * 10),
        ("negative", [ARR(1)], lambda x, e, v: x.__setitem__(e[0], v), [i64([-1, 1])], np.array([5.0, 6, 7, 8, 9])),
    ]
    for name, spec, fn, extra, vals in assigns:
        b = a.copy()
        fn(b, extra, vals)
        save_case(f"fx_set_{name}", "fx_set", [a] + extra + [vals], {"index": spec, "n_arrays": len(extra)}, b, "float64",
                  rtol=0.0, atol=0.0)

    # take_along_axis / put_along_axis
    m = np.array([[10.0, 30, 20, 5], [60, 40, 50, 1], [7, 9, 8, 6]])
    order = np.argsort(m, axis=1)
    save_case("fx_take_along_axis1", "fx_take_along", [m, order.astype(np.int64)], {"axis": 1},
              np.take_along_axis(m, order, axis=1), "float64", rtol=0.0, atol=0.0)
    order0 = np.argsort(m, axis=0)
    save_case("fx_take_along_axis0", "fx_take_along", [m, order0.astype(np.int64)], {"axis": 0},
              np.take_along_axis(m, order0, axis=0), "float64", rtol=0.0, atol=0.0)
    mx = np.argmax(m, axis=1, keepdims=True).astype(np.int64)
    save_case("fx_take_along_argmax", "fx_take_along", [m, mx], {"axis": -1},
              np.take_along_axis(m, mx, axis=-1), "float64", rtol=0.0, atol=0.0)
    bi = np.array([[0], [2], [3]], dtype=np.int64)
    save_case("fx_take_along_broadcast", "fx_take_along", [m, bi], {"axis": 1},
              np.take_along_axis(m, bi, axis=1), "float64", rtol=0.0, atol=0.0)
    p = m.copy()
    pi = np.array([[1], [0], [3]], dtype=np.int64)
    np.put_along_axis(p, pi, 99.0, axis=1)
    save_case("fx_put_along_scalar", "fx_put_along", [m, pi, np.array(99.0)], {"axis": 1}, p, "float64", rtol=0.0, atol=0.0)
    p = m.copy()
    pv = np.array([[-1.0, -2], [-3, -4], [-5, -6]])
    pi2 = np.array([[0, 3], [1, 2], [3, 0]], dtype=np.int64)
    np.put_along_axis(p, pi2, pv, axis=1)
    save_case("fx_put_along_values", "fx_put_along", [m, pi2, pv], {"axis": 1}, p, "float64", rtol=0.0, atol=0.0)

    # put
    q = np.arange(10.0)
    np.put(q, [1, 3, -1, 5], [100.0, 200.0])
    save_case("fx_put_cycle", "fx_put", [np.arange(10.0), i64([1, 3, -1, 5]), np.array([100.0, 200.0])], {}, q, "float64",
              rtol=0.0, atol=0.0)
    q2 = np.zeros((2, 3))
    np.put(q2, [[0, 5], [2, 4]], [1.0, 2.0, 3.0, 4.0])
    save_case("fx_put_2d", "fx_put", [np.zeros((2, 3)), i64([[0, 5], [2, 4]]), np.array([1.0, 2.0, 3.0, 4.0])], {}, q2,
              "float64", rtol=0.0, atol=0.0)

    # choose
    sel = i64([[0, 1, 2], [2, 1, 0]])
    ch = [np.arange(6.0).reshape(2, 3), np.arange(6.0).reshape(2, 3) * 10, np.array([7.0, 8.0, 9.0])]
    save_case("fx_choose", "fx_choose", [sel] + ch, {}, np.choose(sel, ch), "float64", rtol=0.0, atol=0.0)

    # ravel_multi_index / unravel_index
    r, c, d = i64([0, 1, 2, 2]), i64([1, 2, 3, 0]), i64([0, 4, 1, 4])
    save_case("fx_ravel", "fx_ravel", [r, c, d], {"dims": [3, 4, 5]}, np.ravel_multi_index((r, c, d), (3, 4, 5)).astype(np.int64),
              "int64")
    flat = i64([[0, 7], [59, 33]])
    un = np.stack(np.unravel_index(flat, (3, 4, 5))).astype(np.int64)
    save_case("fx_unravel", "fx_unravel", [flat], {"dims": [3, 4, 5]}, un, "int64")

    # ix_
    rows, cols = i64([0, 2]), i64([1, 3, 4])
    b2 = np.arange(20.0).reshape(4, 5)
    save_case("fx_ix", "fx_ix", [b2, rows, cols], {}, b2[np.ix_(rows, cols)], "float64", rtol=0.0, atol=0.0)
    b3 = np.arange(60.0).reshape(3, 4, 5)
    z = i64([2, 0])
    save_case("fx_ix_3", "fx_ix", [b3, z, i64([1, 2, 3]), i64([4, 0])], {}, b3[np.ix_(z, i64([1, 2, 3]), i64([4, 0]))],
              "float64", rtol=0.0, atol=0.0)


def ma_cases(save_case, rng):
    """numpy.ma parity cases (ops prefixed `ma_`).

    Every case is saved three times (`part` = filled / mask / data) so the masked result is compared
    value-by-value, mask-by-mask and, where NumPy defines it, under the mask as well.
    """
    FILL_F, FILL_I = -999.0, -999

    def sample(shape, seed, integer=False):
        r = np.random.default_rng(seed)
        if integer:
            return r.integers(-9, 10, shape).astype(np.int64)
        return np.round(r.standard_normal(shape) * 3, 3)

    x = sample((4, 5), 1)
    xm = np.zeros((4, 5), dtype=bool)
    xm[0, 1] = xm[2, 3] = xm[3, 0] = xm[3, 4] = True
    y = sample((4, 5), 2)
    y[1, 1] = 0.0
    y[2, 2] = 0.0
    ym = np.zeros((4, 5), dtype=bool)
    ym[1, 3] = ym[2, 2] = True
    row = sample((1, 5), 3)
    rowm = np.array([[False, True, False, False, False]])
    X = np.ma.array(x, mask=xm)
    Y = np.ma.array(y, mask=ym)
    R = np.ma.array(row, mask=rowm)

    def parts(res, integer=False):
        r = np.ma.asanyarray(res)
        fill = FILL_I if integer else FILL_F
        dtype = r.dtype
        filled = np.ma.filled(r, np.array(fill).astype(dtype))
        return filled, np.ma.getmaskarray(r), np.ma.getdata(r)

    def case(name, op, operands, params, res, raw=True, integer=False, rtol=1e-12, atol=1e-12):
        """operands: masked arrays / ndarrays; masks saved as extra input files."""
        files, nomask = [], []
        for k, o in enumerate(operands):
            if isinstance(o, np.ma.MaskedArray):
                files += [np.ma.getdata(o), np.ma.getmaskarray(o)]
                if o.mask is np.ma.nomask:
                    nomask.append(k)
            else:
                files += [np.asarray(o), np.zeros(np.shape(o), dtype=bool)]
                nomask.append(k)
        filled, mask, data = parts(res, integer)
        base = dict(params)
        base["nomask"] = nomask
        base["n_operands"] = len(operands)
        for part, arr in [("filled", filled), ("mask", mask)] + ([("data", data)] if raw else []):
            dtype = "bool" if part == "mask" else ("int64" if arr.dtype.kind == "i" else "float64")
            save_case(f"ma_{name}_{part}", f"ma_{op}", files, {**base, "part": part}, arr, dtype, rtol=rtol, atol=atol)

    # arithmetic with mask propagation (data under the mask is the first operand's, like NumPy)
    with np.errstate(all="ignore"):
        case("add", "add", [X, Y], {}, X + Y)
        case("sub", "sub", [X, Y], {}, X - Y)
        case("mul", "mul", [X, Y], {}, X * Y)
        case("div", "div", [X, Y], {}, X / Y)
        case("add_broadcast", "add", [X, R], {}, X + R)
        case("div_broadcast", "div", [X, R], {}, X / R)
        case("add_scalar", "add_scalar", [X], {"value": 2.5}, X + 2.5)
        case("mul_scalar", "mul_scalar", [X], {"value": -2.0}, X * -2.0)
        case("rdiv_scalar", "rdiv_scalar", [X], {"value": 1.0}, 1.0 / X)
        case("neg", "neg", [X], {}, -X)
        case("abs", "abs", [X], {}, np.ma.abs(X))
        case("square", "square", [X], {}, np.ma.power(X, 2), raw=False)
        # unary functions and their domains
        for name, f in [("exp", np.ma.exp), ("sin", np.ma.sin), ("cos", np.ma.cos), ("tanh", np.ma.tanh),
                        ("sqrt", np.ma.sqrt), ("log", np.ma.log), ("log2", np.ma.log2), ("log10", np.ma.log10)]:
            case(name, name, [X], {}, f(X))
        unit = np.ma.array(np.array([[-1.5, -1.0, -0.25, 0.0, 0.5], [1.0, 1.5, 0.75, -0.75, 2.0]]),
                           mask=np.array([[False, False, False, True, False], [False, False, False, False, False]]))
        case("arcsin", "arcsin", [unit], {}, np.ma.arcsin(unit))
        case("arccos", "arccos", [unit], {}, np.ma.arccos(unit))
        # no mask at all: the nomask fast path
        pa, pb = np.arange(6.0).reshape(2, 3), np.arange(6.0).reshape(2, 3) + 1
        case("nomask_add", "add", [pa, pb], {}, np.ma.array(pa) + np.ma.array(pb))
        case("nomask_sum", "sum", [pa], {}, np.ma.array(pa).sum())
        # reductions
        for axis in [None, 0, 1]:
            tag = "all" if axis is None else f"ax{axis}"
            kw = {} if axis is None else {"axis": axis}
            params = {} if axis is None else {"axis": axis}
            case(f"sum_{tag}", "sum", [X], params, X.sum(**kw), raw=False)
            case(f"prod_{tag}", "prod", [X], params, X.prod(**kw), raw=False, rtol=1e-10)
            case(f"mean_{tag}", "mean", [X], params, X.mean(**kw), raw=False)
            case(f"std_{tag}", "std", [X], params, X.std(**kw), raw=False, rtol=1e-10)
            case(f"var_{tag}", "var", [X], params, X.var(**kw), raw=False, rtol=1e-10)
            case(f"min_{tag}", "min", [X], params, X.min(**kw), raw=False)
            case(f"max_{tag}", "max", [X], params, X.max(**kw), raw=False)
            case(f"argmin_{tag}", "argmin", [X], params, np.ma.asarray(X.argmin(**kw)), raw=False, integer=True)
            case(f"argmax_{tag}", "argmax", [X], params, np.ma.asarray(X.argmax(**kw)), raw=False, integer=True)
            case(f"count_{tag}", "count", [X], params, np.ma.asarray(X.count(**kw)), raw=False, integer=True)
        case("var_ddof1", "var_ddof", [X], {"ddof": 1}, X.var(ddof=1), raw=False, rtol=1e-10)
        case("var_ddof1_ax1", "var_ddof", [X], {"ddof": 1, "axis": 1}, X.var(axis=1, ddof=1), raw=False, rtol=1e-10)
        allm = np.ma.array(x[:2], mask=True)
        case("sum_allmasked", "sum", [allm], {}, allm.sum(), raw=False)
        case("mean_allmasked", "mean", [allm], {}, allm.mean(), raw=False)
        half = np.ma.array(x[:3], mask=np.array([[True] * 5, [False] * 5, [False] * 5]))
        case("sum_lane_masked", "sum", [half], {"axis": 1}, half.sum(axis=1), raw=False)
        case("max_lane_masked", "max", [half], {"axis": 1}, half.max(axis=1), raw=False)
        case("mean_lane_masked", "mean", [half], {"axis": 1}, half.mean(axis=1), raw=False)
        case("std_lane_masked", "std", [half], {"axis": 1}, half.std(axis=1), raw=False, rtol=1e-10)
        # integer data
        xi = sample((3, 4), 5, integer=True)
        mi = np.zeros((3, 4), dtype=bool)
        mi[0, 0] = mi[1, 2] = True
        XI = np.ma.array(xi, mask=mi)
        case("int_sum", "sum", [XI], {}, XI.sum(), raw=False, integer=True)
        case("int_sum_ax0", "sum", [XI], {"axis": 0}, XI.sum(axis=0), raw=False, integer=True)
        case("int_min", "min", [XI], {}, XI.min(), raw=False, integer=True)
        case("int_max_ax1", "max", [XI], {"axis": 1}, XI.max(axis=1), raw=False, integer=True)
        case("int_add", "add", [XI, XI], {}, XI + XI, integer=True)
        case("int_mul_scalar", "mul_scalar", [XI], {"value": 3}, XI * 3, integer=True)
        # constructors
        plain = sample((3, 4), 7)
        pl = plain.copy()
        pl[0, 1], pl[2, 2], pl[1, 3] = np.nan, np.inf, -np.inf
        case("masked_invalid", "masked_invalid", [pl], {}, np.ma.masked_invalid(pl))
        eq = np.array([[1.0, 2.0, 1.0], [3.0, 1.0, 2.0]])
        for name, f, value in [("equal", np.ma.masked_equal, 1.0), ("greater", np.ma.masked_greater, 1.5),
                               ("greater_equal", np.ma.masked_greater_equal, 2.0), ("less", np.ma.masked_less, 2.0),
                               ("less_equal", np.ma.masked_less_equal, 2.0), ("not_equal", np.ma.masked_not_equal, 1.0)]:
            case(f"masked_{name}", f"masked_{name}", [eq], {"value": value}, f(eq, value))
        case("masked_inside", "masked_inside", [eq], {"v1": 1.5, "v2": 2.5}, np.ma.masked_inside(eq, 1.5, 2.5))
        case("masked_outside", "masked_outside", [eq], {"v1": 1.5, "v2": 2.5}, np.ma.masked_outside(eq, 1.5, 2.5))
        case("masked_values", "masked_values", [eq + np.array([[1e-9, 0, 0], [0, 1e-9, 0]])],
             {"value": 1.0, "rtol": 1e-5, "atol": 1e-8}, np.ma.masked_values(eq + np.array([[1e-9, 0, 0], [0, 1e-9, 0]]), 1.0, rtol=1e-5, atol=1e-8))
        chained = np.ma.masked_greater(np.ma.masked_less(eq, 1.5), 2.5)
        case("chained", "chained", [eq], {}, chained)
        cond = np.array([[True, False, False], [False, True, False]])
        case("masked_where", "masked_where", [eq, cond], {}, np.ma.masked_where(cond, eq))
        # compressed / filled / indexing
        case("compressed", "compressed", [X], {}, np.ma.asarray(X.compressed()), raw=False)
        case("slice", "slice", [X], {}, X[1:3, ::2])
        case("reshape", "reshape", [X], {"shape": [5, 4]}, X.reshape(5, 4))
        case("transpose", "transpose", [X], {}, X.T)
        case("ravel", "ravel", [X], {}, X.ravel())
        # display
        strings = {
            "float": np.ma.array([1.0, 2.5, 3.0, 22.25], mask=[0, 1, 0, 0]),
            "matrix": np.ma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mask=[[0, 0, 1], [1, 0, 0]]),
            "int": np.ma.array([[1, 22], [333, 4]], mask=[[0, 1], [0, 0]]),
            "bool": np.ma.array([True, False, True], mask=[0, 0, 1]),
            "exp": np.ma.array([1e-5, 1e16, 0.1, 100.0], mask=[0, 0, 1, 0]),
            "nomask": np.ma.array([1.5, 2.5]),
            "special": np.ma.array([np.nan, np.inf, -0.0, 1 / 3], mask=[0, 0, 0, 0]),
        }
        for k, arr in strings.items():
            save_case(f"ma_str_{k}", "ma_str", [np.ma.getdata(arr), np.ma.getmaskarray(arr)],
                      {"str": str(arr), "nomask": [] if arr.mask is not np.ma.nomask else [0], "n_operands": 1},
                      np.array(0), "str")


if __name__ == "__main__":
    main()
