"""Generate rustnum/src/misc/tests_{math,window,complex}.rs from real NumPy (run once)."""
import itertools
import math
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
np.seterr(all="ignore")

OUT = sys.argv[1]
nan, inf = float("nan"), float("inf")

HEADER = """//! NumPy parity tests for the scalar math kernels. GENERATED ONCE by gen_tests.py with NumPy @VER@;
//! the tables below are literal NumPy output (not computed by the code under test).
#![allow(clippy::approx_constant, clippy::excessive_precision, clippy::type_complexity)]

use super::math_kernels::*;
use super::tests_util::{check, Report};
""".replace("@VER@", np.__version__)


def lit(v, ty="f64"):
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = float(v) if ty == "f64" else v
    if ty == "f32":
        v32 = np.float32(v)
        if np.isnan(v32):
            return "f32::NAN"
        if np.isinf(v32):
            return "f32::INFINITY" if v32 > 0 else "f32::NEG_INFINITY"
        s = str(v32)
        return s if any(c in s for c in ".e") else s + ".0"
    if math.isnan(v):
        return "f64::NAN"
    if math.isinf(v):
        return "f64::INFINITY" if v > 0 else "f64::NEG_INFINITY"
    return repr(v)


def table(name, ty_decl, rows, tys):
    """rows: list of tuples; tys: per-column literal type."""
    out = [f"    const {name}: &[{ty_decl}] = &["]
    line = []
    for r in rows:
        cells = ", ".join(lit(c, t) for c, t in zip(r, tys))
        line.append(f"({cells}),")
    out += ["        " + l for l in line]
    out.append("    ];")
    return "\n".join(out)


DT = {"f64": np.float64, "f32": np.float32}

# ------------------------------------------------------------------------------------------
# math kernels
# ------------------------------------------------------------------------------------------
GRID = [nan, inf, -inf, 0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.0, 4.5, -7.25,
        8.0, 8.000001, 100.0, 123456.789, 1e16, 0.1, -0.1, 1e-300, -1e-300, 5e-324, -5e-324,
        2.2250738585072014e-308, 1e300, -1e300, 1.7976931348623157e308, -1.7976931348623157e308]
GRID32 = [nan, inf, -inf, 0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.0, 4.5, -7.25,
          8.0, 100.0, 123456.789, 1e16, 0.1, -0.1, 1e-30, -1e-30, 1e-45, -1e-45, 1.1754944e-38,
          1e30, -1e30, 3.4028235e38, -3.4028235e38, 89.0, 88.0]
PAIR = [nan, inf, -inf, 0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 5.5, -5.5, 0.3, -0.3,
        7.0, 1e300, -1e300, 5e-324, -5e-324, 1e-10, 3.0, -3.0]

m = []  # math test file body


def unary(fname, rust, ty, grid, f, rtol, atol, extra=""):
    a = np.array(grid, dtype=DT[ty])
    res = f(a)
    rows = list(zip(a.tolist(), res.tolist()))
    m.append(f"""
#[test]
fn {fname}_{ty}() {{
{table("CASES", f"({ty}, {ty})", rows, [ty, ty])}
    for &(x, want) in CASES {{
        check("{rust}", &format!("{{x:?}}"), {extra or rust}(x), want, {rtol}, {atol});
    }}
}}""")


def binary(fname, rust, ty, grid, f, rtol, atol):
    pairs = list(itertools.product(grid, grid))
    x = np.array([p[0] for p in pairs], dtype=DT[ty])
    y = np.array([p[1] for p in pairs], dtype=DT[ty])
    res = f(x, y)
    rows = list(zip(x.tolist(), y.tolist(), res.tolist()))
    m.append(f"""
#[test]
fn {fname}_{ty}() {{
{table("CASES", f"({ty}, {ty}, {ty})", rows, [ty, ty, ty])}
    for &(x, y, want) in CASES {{
        check("{rust}", &format!("{{x:?}}, {{y:?}}"), {rust}(x, y), want, {rtol}, {atol});
    }}
}}""")


M_IMPORT = "use super::math_kernels::*;"
for ty in ("f64", "f32"):
    g = GRID if ty == "f64" else GRID32
    rt = "1e-12" if ty == "f64" else "2e-5"
    unary("sinc", "sinc", ty, g, np.sinc, rt, "0.0")
    unary("i0", "i0", ty, g, np.i0, rt, "0.0")
    unary("deg2rad", "deg2rad", ty, g, np.deg2rad, "0.0", "0.0")
    unary("rad2deg", "rad2deg", ty, g, np.rad2deg, "0.0", "0.0")
    unary("rint", "rint", ty, g + [0.5, 1.5, 2.5, -0.4, -0.6, 4503599627370497.0, 8388609.0], np.rint, "0.0", "0.0")
    unary("fix", "fix", ty, g + [-0.4, 0.4, 2.7, -2.7], np.fix, "0.0", "0.0")
    unary("fabs", "fabs", ty, g, np.fabs, "0.0", "0.0")
    unary("positive", "positive", ty, g, np.positive, "0.0", "0.0")
    unary("negative", "negative", ty, g, np.negative, "0.0", "0.0")
    unary("spacing", "spacing", ty, g, np.spacing, "0.0", "0.0")

# signbit (bool), also with a negative NaN
a = np.array(GRID + [-nan])
a[-1] = np.float64(np.frombuffer(np.uint64(0xFFF8000000000000).tobytes(), dtype=np.float64)[0])
rows = list(zip(a.tolist(), np.signbit(a).tolist()))
neg_nan_idx = len(rows) - 1
m.append(f"""
#[test]
fn signbit_f64() {{
{table("CASES", "(f64, bool)", rows[:-1], ["f64", "bool"])}
    for &(x, want) in CASES {{
        assert_eq!(signbit(x), want, "signbit({{x:?}})");
    }}
    // a NaN with the sign bit set: NumPy says {rows[-1][1]}
    assert!({'' if rows[-1][1] else '!'}signbit(f64::from_bits(0xFFF8_0000_0000_0000)));
    assert!(!signbit(f32::NAN));
    assert!(signbit(-0.0f32));
}}""")

# binary float kernels
for ty in ("f64", "f32"):
    g = PAIR if ty == "f64" else [v for v in PAIR if abs(v) < 1e30 and (v == 0 or abs(v) > 1e-30)] + [1e30, -1e30, 1e-40, 1.0000001]
    rt = "1e-13" if ty == "f64" else "1e-5"
    binary("fmod", "fmod", ty, g, np.fmod, "0.0", "0.0")
    binary("remainder", "remainder", ty, g, np.remainder, "0.0", "0.0")
    binary("floor_divide", "floor_divide", ty, g, np.floor_divide, "0.0", "0.0")
    binary("copysign", "copysign", ty, g, np.copysign, "0.0", "0.0")
    binary("nextafter", "nextafter", ty, g, np.nextafter, "0.0", "0.0")
    binary("heaviside", "heaviside", ty, g, np.heaviside, "0.0", "0.0")
    binary("logaddexp", "logaddexp", ty, g, np.logaddexp, rt, "0.0")
    binary("logaddexp2", "logaddexp2", ty, g, np.logaddexp2, rt, "0.0")

# divmod: both outputs
for ty in ("f64", "f32"):
    g = PAIR if ty == "f64" else [v for v in PAIR if abs(v) < 1e30 and (v == 0 or abs(v) > 1e-30)]
    pairs = list(itertools.product(g, g))
    x = np.array([p[0] for p in pairs], dtype=DT[ty])
    y = np.array([p[1] for p in pairs], dtype=DT[ty])
    q, r = np.divmod(x, y)
    rows = list(zip(x.tolist(), y.tolist(), q.tolist(), r.tolist()))
    m.append(f"""
#[test]
fn divmod_{ty}() {{
{table("CASES", f"({ty}, {ty}, {ty}, {ty})", rows, [ty] * 4)}
    for &(x, y, wq, wr) in CASES {{
        let (q, r) = divmod(x, y);
        check("divmod.0", &format!("{{x:?}}, {{y:?}}"), q, wq, 0.0, 0.0);
        check("divmod.1", &format!("{{x:?}}, {{y:?}}"), r, wr, 0.0, 0.0);
    }}
}}""")

# float_power (always f64)
g = [nan, inf, -inf, 0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, 10.0, 1e-3, 1e200, 300.0, -3.5, 0.1]
pairs = list(itertools.product(g, g))
x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
rows = list(zip(x.tolist(), y.tolist(), np.float_power(x, y).tolist()))
m.append(f"""
#[test]
fn float_power_f64() {{
{table("CASES", "(f64, f64, f64)", rows, ["f64"] * 3)}
    for &(x, y, want) in CASES {{
        check("float_power", &format!("{{x:?}}, {{y:?}}"), float_power(x, y), want, 1e-13, 0.0);
    }}
}}""")
# float_power of f32 inputs returns f64
x32 = np.array([2.0, 0.1, -3.0, 1e10, 7.5], dtype=np.float32)
y32 = np.array([0.5, 3.0, 3.0, 2.0, -1.0], dtype=np.float32)
rows = list(zip(x32.tolist(), y32.tolist(), np.float_power(x32, y32).tolist()))
assert np.float_power(x32, y32).dtype == np.float64
m.append(f"""
#[test]
fn float_power_f32_inputs_give_f64() {{
{table("CASES", "(f32, f32, f64)", [(np.float32(a), np.float32(b), c) for a, b, c in rows], ["f32", "f32", "f64"])}
    for &(x, y, want) in CASES {{
        check("float_power", &format!("{{x:?}}, {{y:?}}"), float_power(x, y), want, 1e-13, 0.0);
    }}
}}""")

# modf
for ty in ("f64", "f32"):
    g = GRID if ty == "f64" else GRID32
    a = np.array(g + [-3.0, 3.75, -0.25], dtype=DT[ty])
    fr, ip = np.modf(a)
    rows = list(zip(a.tolist(), fr.tolist(), ip.tolist()))
    m.append(f"""
#[test]
fn modf_{ty}() {{
{table("CASES", f"({ty}, {ty}, {ty})", rows, [ty] * 3)}
    for &(x, wf, wi) in CASES {{
        let (f, i) = modf(x);
        check("modf.frac", &format!("{{x:?}}"), f, wf, 0.0, 0.0);
        check("modf.int", &format!("{{x:?}}"), i, wi, 0.0, 0.0);
    }}
}}""")

# frexp
for ty in ("f64", "f32"):
    g = (GRID if ty == "f64" else GRID32) + [3.0e-320, -3.0e-320, 1e-310] if ty == "f64" else GRID32 + [1e-40, -1e-40, 5e-39]
    a = np.array(g, dtype=DT[ty])
    mm, ee = np.frexp(a)
    assert ee.dtype == np.int32
    rows = list(zip(a.tolist(), mm.tolist(), ee.tolist()))
    m.append(f"""
#[test]
fn frexp_{ty}() {{
{table("CASES", f"({ty}, {ty}, i32)", rows, [ty, ty, "i"])}
    for &(x, wm, we) in CASES {{
        let (mm, e) = frexp(x);
        check("frexp.m", &format!("{{x:?}}"), mm, wm, 0.0, 0.0);
        if x.is_finite() {{
            // NumPy on Windows returns -1 for inf/nan, glibc returns 0: unspecified, not compared
            assert_eq!(e, we, "frexp({{x:?}}).e");
        }}
    }}
}}""")

# ldexp
for ty in ("f64", "f32"):
    if ty == "f64":
        xs = [1.0, -1.0, 0.75, 3.0, 1e-300, 5e-324, 1.7976931348623157e308, 0.0, -0.0, nan, inf, -inf, 1.2345678901234567, 3e-320]
        ns = [-2200, -1100, -1075, -1074, -1023, -1022, -60, -1, 0, 1, 10, 52, 1023, 1024, 1100, 2100, 4000, 2**31 - 1, -(2**31)]
    else:
        xs = [1.0, -1.0, 0.75, 3.0, 1e-30, 1e-45, 3.4028235e38, 0.0, -0.0, nan, inf, -inf, 1.2345678, 3e-40]
        ns = [-400, -200, -150, -149, -127, -126, -60, -1, 0, 1, 10, 23, 127, 128, 200, 400, 2**31 - 1, -(2**31)]
    pairs = list(itertools.product(xs, ns))
    x = np.array([p[0] for p in pairs], dtype=DT[ty])
    n = np.array([p[1] for p in pairs], dtype=np.int32)
    res = np.ldexp(x, n)
    assert res.dtype == DT[ty], res.dtype
    rows = [(a, b, c) for a, b, c in zip(x.tolist(), n.tolist(), res.tolist())]
    m.append(f"""
#[test]
fn ldexp_{ty}() {{
{table("CASES", f"({ty}, i32, {ty})", rows, [ty, "i", ty])}
    for &(x, n, want) in CASES {{
        check("ldexp", &format!("{{x:?}}, {{n}}"), ldexp(x, n), want, 0.0, 0.0);
    }}
}}""")

# gcd / lcm
ints = [0, 1, -1, 2, -2, 6, -6, 12, 18, -18, 35, 1071, 462, 2**31, 2**40 + 6, 2**62, -(2**62), 3**30, 10**18, -(2**63), 2**63 - 1, 7, 0]
pairs = list(itertools.product(ints, ints))
x = np.array([p[0] for p in pairs], dtype=np.int64); y = np.array([p[1] for p in pairs], dtype=np.int64)
rows = list(zip(x.tolist(), y.tolist(), np.gcd(x, y).tolist(), np.lcm(x, y).tolist()))
body = ["    const CASES: &[(i64, i64, i64, i64)] = &["]
for r in rows:
    body.append("        (" + ", ".join(f"{v}_i64" if v != -(2**63) else "i64::MIN" for v in r) + "),")
body.append("    ];")
m.append(f"""
#[test]
fn gcd_lcm_i64() {{
{chr(10).join(body)}
    for &(a, b, wg, wl) in CASES {{
        assert_eq!(gcd(a, b), wg, "gcd({{a}}, {{b}})");
        assert_eq!(lcm(a, b), wl, "lcm({{a}}, {{b}})");
    }}
}}""")

# unwrap
unwrap_cases = [
    ("default", np.linspace(0, 4 * np.pi, 25) % (2 * np.pi), None, 2 * np.pi),
    ("phase", np.array([0, 0.78539816, 1.57079633, 5.49778714, 6.28318531]), None, 2 * np.pi),
    ("period4", np.array([0.0, 1, 2, -1, 0]), None, 4.0),
    ("period6", np.array([1.0, 2, 3, 4, 5, 6, 1, 2, 3]), None, 6.0),
    ("period4b", np.array([2.0, 3, 4, 5, 2, 3, 4, 5]), None, 4.0),
    ("deg", np.mod(np.linspace(0, 720, 19), 360) - 180, None, 360.0),
    ("discont_small", np.array([0.0, 3.0, 6.0, 0.5, 3.5]), 1.0, 2 * np.pi),
    ("discont_big", np.array([0.0, 3.0, 6.0, 0.5, 3.5]), 5.0, 2 * np.pi),
    ("nan_inside", np.array([0.0, 1.0, np.nan, 7.0, 8.0]), None, 2 * np.pi),
    ("inf", np.array([0.0, np.inf, 1.0]), None, 2 * np.pi),
    ("boundary", np.array([0.0, np.pi, 2 * np.pi, 0.0, -np.pi]), None, 2 * np.pi),
    ("single", np.array([3.0]), None, 2 * np.pi),
    ("empty", np.array([]), None, 2 * np.pi),
    ("neg_zero", np.array([-0.0, 0.0, -0.0]), None, 2 * np.pi),
]
parts = []
for name, p, d, per in unwrap_cases:
    kw = {} if d is None else {"discont": d}
    res = np.unwrap(p, period=per, **kw)
    dl = "None" if d is None else f"Some({lit(d)})"
    parts.append(f"""    {{
        let p: &[f64] = &[{", ".join(lit(v) for v in p)}];
        let want: &[f64] = &[{", ".join(lit(v) for v in res)}];
        let got = unwrap(p, {dl}, {lit(per)});
        assert_eq!(got.len(), want.len(), "unwrap {name}");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {{
            check("unwrap {name}", &format!("index {{i}}"), g, w, 1e-12, 0.0);
        }}
    }}""")
m.append("\n#[test]\nfn unwrap_f64() {\n" + "\n".join(parts) + "\n}")
# unwrap f32
p32 = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float32)
res = np.unwrap(p32)
assert res.dtype == np.float32, res.dtype
m.append(f"""
#[test]
fn unwrap_f32() {{
    let p: &[f32] = &[{", ".join(lit(v, "f32") for v in p32)}];
    let want: &[f32] = &[{", ".join(lit(v, "f32") for v in res)}];
    let got = unwrap(p, None, core::f32::consts::TAU);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {{
        check("unwrap f32", &format!("index {{i}}"), g, w, 1e-5, 1e-6);
    }}
}}""")

# nan_to_num
for ty in ("f64", "f32"):
    g = GRID if ty == "f64" else GRID32
    a = np.array(g, dtype=DT[ty])
    r1 = np.nan_to_num(a)
    r2 = np.nan_to_num(a, nan=-1.0, posinf=5.0, neginf=-5.0)
    r3 = np.nan_to_num(a, nan=2.0, posinf=None, neginf=-9.0)
    rows = list(zip(a.tolist(), r1.tolist(), r2.tolist(), r3.tolist()))
    m.append(f"""
#[test]
fn nan_to_num_{ty}() {{
{table("CASES", f"({ty}, {ty}, {ty}, {ty})", rows, [ty] * 4)}
    for &(x, d, w2, w3) in CASES {{
        let s = format!("{{x:?}}");
        check("nan_to_num default", &s, nan_to_num(x, 0.0, None, None), d, 0.0, 0.0);
        check("nan_to_num explicit", &s, nan_to_num(x, -1.0, Some(5.0), Some(-5.0)), w2, 0.0, 0.0);
        check("nan_to_num partial", &s, nan_to_num(x, 2.0, None, Some(-9.0)), w3, 0.0, 0.0);
    }}
}}""")

# ediff1d
ed = []
for name, a, end, beg in [
    ("ints", np.array([1, 2, 4, 7, 11], dtype=np.int64), [], []),
    ("ints_end_begin", np.array([1, 2, 4, 7, 11], dtype=np.int64), [99], [-5, -6]),
    ("floats", np.array([1.5, 1.0, 3.25, -2.0]), [], []),
    ("floats_end", np.array([1.5, 1.0, 3.25, -2.0]), [0.5, 0.25], []),
    ("single", np.array([4.0]), [], []),
    ("empty", np.array([]), [1.0], [2.0]),
]:
    kw = {}
    kw["to_end"] = np.array(end, dtype=a.dtype) if end else None
    kw["to_begin"] = np.array(beg, dtype=a.dtype) if beg else None
    res = np.ediff1d(a, **kw)
    ty = "i64" if a.dtype == np.int64 else "f64"
    f = (lambda v: str(int(v))) if ty == "i64" else lit
    ed.append(f"""    {{
        let a: &[{ty}] = &[{", ".join(f(v) for v in a)}];
        let end: &[{ty}] = &[{", ".join(f(v) for v in end)}];
        let beg: &[{ty}] = &[{", ".join(f(v) for v in beg)}];
        let want: &[{ty}] = &[{", ".join(f(v) for v in res)}];
        assert_eq!(ediff1d(a, end, beg), want, "ediff1d {name}");
    }}""")
m.append("\n#[test]\nfn ediff1d_cases() {\n" + "\n".join(ed) + "\n}")

# angle (real input uses im = 0)
for ty in ("f64", "f32"):
    g = [nan, inf, -inf, 0.0, -0.0, 1.0, -1.0, 2.5, -3.0, 1e-300] if ty == "f64" else [nan, inf, -inf, 0.0, -0.0, 1.0, -1.0, 2.5, -3.0, 1e-30]
    pts = list(itertools.product(g, g))
    z = np.array([complex(a, b) for a, b in pts], dtype=np.complex128 if ty == "f64" else np.complex64)
    ang = np.angle(z)
    angd = np.angle(z, deg=True)
    rows = [(a, b, c, d) for (a, b), c, d in zip(pts, ang.tolist(), angd.tolist())]
    m.append(f"""
#[test]
fn angle_{ty}() {{
{table("CASES", f"({ty}, {ty}, {ty}, {ty})", rows, [ty] * 4)}
    for &(re, im, w, wd) in CASES {{
        let s = format!("{{re:?}}, {{im:?}}");
        check("angle", &s, angle(re, im), w, 1e-6, 0.0);
        check("angle_deg", &s, angle_deg(re, im), wd, 1e-6, 0.0);
    }}
}}""")
# angle of real input
a = np.array([0.0, -0.0, 1.0, -1.0, nan])
ang = np.angle(a)
rows = list(zip(a.tolist(), ang.tolist()))
m.append(f"""
#[test]
fn angle_real_input_uses_positive_zero_imag() {{
{table("CASES", "(f64, f64)", rows, ["f64", "f64"])}
    for &(x, want) in CASES {{
        check("angle(real)", &format!("{{x:?}}"), angle(x, 0.0), want, 1e-15, 0.0);
    }}
}}""")

# real_if_close
ric = []
for ty in ("f64", "f32"):
    for imag, tol in [([1e-14, 5e-15], 100), ([1e-13, 0.0], 100), ([1e-13, 0.0], 1000), ([0.0, -0.0], 100),
                      ([], 100), ([nan], 100), ([1e-5], 1e-4), ([1e-5], 1e-6), ([2.2e-14], 100), ([2.3e-14], 100),
                      ([1e-6], 100), ([1e-6], 10000), ([inf], 100), ([2e-5], 1000000), ([1e-3], 1.0), ([0.5], 2.0)]:
        z = np.array([complex(1.0, v) for v in imag], dtype=np.complex128 if ty == "f64" else np.complex64)
        res = np.real_if_close(z, tol=tol)
        ok = res.dtype.kind == "f"
        ric.append(f"        (vec![{', '.join(lit(v, ty) for v in imag)}], {lit(float(tol), ty)}, {lit(ok)}),")
        pass
    m.append(f"""
#[test]
fn real_if_close_{ty}() {{
    let cases: Vec<(Vec<{ty}>, {ty}, bool)> = vec![
{chr(10).join(ric)}
    ];
    for (imag, tol, want) in cases {{
        assert_eq!(real_if_close_ok(&imag, tol), want, "real_if_close({{imag:?}}, tol={{tol:?}})");
    }}
}}""")
    ric = []

# ------------------------------------------------------------------------------------------
# windows
# ------------------------------------------------------------------------------------------
w = []
funcs = [("hanning", np.hanning), ("hamming", np.hamming), ("blackman", np.blackman), ("bartlett", np.bartlett)]
for name, f in funcs:
    for M in (0, 1, 2, 3, 4, 5, 10, 11, 51):
        res = f(M)
        w.append(f"""
#[test]
fn {name}_{M}() {{
    let want: &[f64] = &[{", ".join(lit(v) for v in res)}];
    let got = {name}({M});
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {{
        close(g, w, "{name}({M})", i);
    }}
}}""")
for beta in (0.0, 5.0, 8.6, 14.0, -3.0, 0.5):
    for M in (0, 1, 2, 3, 12, 51):
        res = np.kaiser(M, beta)
        tag = str(beta).replace(".", "_").replace("-", "neg")
        w.append(f"""
#[test]
fn kaiser_{M}_beta_{tag}() {{
    let want: &[f64] = &[{", ".join(lit(v) for v in res)}];
    let got = kaiser({M}, {lit(beta)});
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {{
        close(g, w, "kaiser({M}, {beta})", i);
    }}
}}""")
# big windows: sample indices
idx = [0, 1, 2, 249, 250, 499, 500, 501, 750, 998, 999, 1000]
samples = {}
for name, f in funcs:
    samples[name] = [(i, f(1001)[i]) for i in idx]
samples["kaiser"] = [(i, np.kaiser(1001, 8.6)[i]) for i in idx]
for name, rows in samples.items():
    call = f"{name}(1001)" if name != "kaiser" else "kaiser(1001, 8.6)"
    w.append(f"""
#[test]
fn {name}_1001_samples() {{
    const SAMPLES: &[(usize, f64)] = &[{", ".join(f"({i}, {lit(v)})" for i, v in rows)}];
    let got = {call};
    assert_eq!(got.len(), 1001);
    for &(i, want) in SAMPLES {{
        close(got[i], want, "{call}", i);
    }}
}}""")
window_file = """//! NumPy parity tests for the window functions (GENERATED ONCE with NumPy {ver}).
#![allow(clippy::approx_constant, clippy::excessive_precision)]

use super::window_kernels::*;

fn close(got: f64, want: f64, what: &str, i: usize) {{
    let tol = 1e-14 + 1e-13 * want.abs();
    assert!((got - want).abs() <= tol, "{{what}}[{{i}}]: got {{got:?}}, NumPy says {{want:?}}");
}}
""".format(ver=np.__version__) + "\n".join(w) + "\n"

# extra windows semantic tests (hand-checked facts, not numerics)
window_file += """
#[test]
fn windows_are_symmetric_and_peak_at_one() {
    for f in [hanning, hamming, blackman, bartlett] {
        let w = f(33);
        for i in 0..w.len() {
            assert!((w[i] - w[w.len() - 1 - i]).abs() < 1e-15);
        }
        assert!((w[16] - 1.0).abs() < 1e-15);
    }
}
"""

# ------------------------------------------------------------------------------------------
# complex
# ------------------------------------------------------------------------------------------
c = []
V = [nan, inf, -inf, 0.0, -0.0, 1.0, -1.0, 0.5, -2.0, 3.0]
SPECIAL = [complex(a, b) for a in V for b in V]
NICE = [1 + 2j, -3 + 0.5j, 1e-5 + 1e-5j, 25 + 0j, 30j, 22.5 - 3j, 10 + 1e5j, 700 + 1j, 710 + 1j, 750 + 2j,
        -750 + 2j, 1e300 + 1e300j, -1e300 - 1e300j, 1e-300 + 1e-300j, 5e-324 + 0j, -4 + 0j, -4 - 0j,
        complex(-4, 0.0), complex(-4, -0.0), 0.5 + 1e-10j, 1 + 1e-10j, 1 - 1e-10j, 0.999999 + 0.001j, 3 + 4j,
        -1 + 1e-300j, 2 - 2j, 8.5 + 0.1j, -9 + 40j, 12 + 200j, -0.3 - 0.4j, 1e8 + 1e8j, 1.7e308 + 1.7e308j,
        -1.7e308 + 1j, 100 + 0.5j, -100 - 0.5j, 0.1 + 0.1j, 709.5 + 0.5j, 709.9 + 1.5707963267948966j,
        -0.0 + 2j, 0.0 - 2j, complex(-0.0, -2.0)]
ZS = SPECIAL + NICE


def cvals(zs, ty):
    dt = np.complex128 if ty == "f64" else np.complex64
    return np.array(zs, dtype=dt)


def ctable(name, rows, ty):
    out = [f"    const {name}: &[(({ty}, {ty}), ({ty}, {ty}))] = &["]
    for (zr, zi), (wr, wi) in rows:
        out.append(f"        (({lit(zr, ty)}, {lit(zi, ty)}), ({lit(wr, ty)}, {lit(wi, ty)})),")
    out.append("    ];")
    return "\n".join(out)


DIVERGENT = {
    "exp": [710 + 1j, 709.9 + 1.5707963267948966j],
    "sinh": [710 + 1j, 709.9 + 1.5707963267948966j],
    "cosh": [710 + 1j, 709.9 + 1.5707963267948966j],
    "sqrt": [1.7e308 + 1.7e308j, -1.7e308 + 1j],
}


def cunary(fname, method, f, ty, zs, rtol):
    zs = [z for z in zs if z not in DIVERGENT.get(fname, [])]
    z = cvals(zs, ty)
    res = f(z)
    rows = [((a.real.item(), a.imag.item()), (b.real.item(), b.imag.item())) for a, b in zip(z, res)]
    c.append(f"""
#[test]
fn complex_{fname}_{ty}() {{
{ctable("CASES", rows, ty)}
    for &((re, im), want) in CASES {{
        let z = Complex::new(re, im);
        check_c("{fname}", &format!("{{z:?}}"), z.{method}(), want, {rtol}, !finite_input(z));
    }}
}}""")


for ty, rt in (("f64", "1e-12"), ("f32", "2e-4")):
    zs = ZS if ty == "f64" else [z for z in ZS if abs(z.real) < 1e30 and abs(z.imag) < 1e30 or not (np.isfinite(z.real) and np.isfinite(z.imag))]
    for fname, f in (("exp", np.exp), ("ln", np.log), ("sqrt", np.sqrt), ("sin", np.sin), ("cos", np.cos), ("tan", np.tan),
                     ("sinh", np.sinh), ("cosh", np.cosh), ("tanh", np.tanh)):
        cunary(fname, fname, f, ty, zs, rt)

# abs / angle / conj / norm_sqr / is_* on ZS
for ty in ("f64", "f32"):
    zs = ZS if ty == "f64" else [z for z in ZS if abs(z.real) < 1e30 and abs(z.imag) < 1e30 or not (np.isfinite(z.real) and np.isfinite(z.imag))]
    z = cvals(zs, ty)
    rows = []
    for a, ab, an, cj, isn, isf, isi in zip(z, np.abs(z), np.angle(z), np.conj(z), np.isnan(z), np.isfinite(z), np.isinf(z)):
        rows.append((a.real.item(), a.imag.item(), ab.item(), an.item(), cj.real.item(), cj.imag.item(), bool(isn), bool(isf), bool(isi)))
    c.append(f"""
#[test]
fn complex_abs_arg_conj_flags_{ty}() {{
{table("CASES", f"({ty}, {ty}, {ty}, {ty}, {ty}, {ty}, bool, bool, bool)", rows, [ty] * 6 + ["b"] * 3)}
    for &(re, im, wabs, warg, wcr, wci, wnan, wfin, winf) in CASES {{
        let z = Complex::new(re, im);
        let s = format!("{{z:?}}");
        check("norm", &s, z.norm(), wabs, 1e-6, 0.0);
        check("arg", &s, z.arg(), warg, 1e-6, 0.0);
        let cj = z.conj();
        check("conj.re", &s, cj.re, wcr, 0.0, 0.0);
        check("conj.im", &s, cj.im, wci, 0.0, 0.0);
        assert_eq!(z.is_nan(), wnan, "is_nan {{s}}");
        assert_eq!(z.is_finite(), wfin, "is_finite {{s}}");
        assert_eq!(z.is_infinite(), winf, "is_infinite {{s}}");
    }}
}}""")

# arithmetic on complex pairs
P = [complex(*t) for t in [(nan, 0.0), (inf, 1.0), (-inf, -inf), (0.0, 0.0), (-0.0, 0.0), (0.0, -0.0), (1.0, 2.0), (-3.0, 0.5),
                           (1e300, 1e300), (1e-300, -1e-300), (2.0, 0.0), (0.0, 1.0), (inf, nan), (nan, nan), (5.0, -7.0), (1e308, 1e-308), (-1.0, -0.0)]]
for ty in ("f64", "f32"):
    pp = P if ty == "f64" else [p for p in P if abs(p.real) < 1e30 and abs(p.imag) < 1e30 or not (np.isfinite(p.real) and np.isfinite(p.imag))]
    pairs = [p for p in itertools.product(pp, pp) if p != (1e300 + 1e300j, 1e300 + 1e300j)]
    a = cvals([p[0] for p in pairs], ty)
    b = cvals([p[1] for p in pairs], ty)
    ops = [("add", a + b), ("sub", a - b), ("mul", a * b), ("div", a / b)]
    for name, res in ops:
        rows = [((x.real.item(), x.imag.item()), (y.real.item(), y.imag.item()), (r.real.item(), r.imag.item())) for x, y, r in zip(a, b, res)]
        out = [f"    const CASES: &[(({ty}, {ty}), ({ty}, {ty}), ({ty}, {ty}))] = &["]
        for (xr, xi), (yr, yi), (rr, ri) in rows:
            out.append(f"        (({lit(xr, ty)}, {lit(xi, ty)}), ({lit(yr, ty)}, {lit(yi, ty)}), ({lit(rr, ty)}, {lit(ri, ty)})),")
        out.append("    ];")
        opsym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[name]
        c.append(f"""
#[test]
fn complex_{name}_{ty}() {{
{chr(10).join(out)}
    for &((ar, ai), (br, bi), want) in CASES {{
        let (a, b) = (Complex::new(ar, ai), Complex::new(br, bi));
        check_c("{name}", &format!("{{a:?}}, {{b:?}}"), a {opsym} b, want, 1e-5, false);
        // assign operator gives the same answer
        let mut t = a;
        t {opsym}= b;
        check_c("{name}_assign", &format!("{{a:?}}, {{b:?}}"), t, want, 1e-5, false);
    }}
}}""")

# real-scalar mixed arithmetic (NumPy promotes the scalar to s+0j)
S = [2.0, 0.0, -0.0, inf, nan, -1.5]
Z = [complex(*t) for t in [(1.0, 2.0), (inf, 1.0), (0.0, -0.0), (-0.0, 3.0), (nan, 1.0), (1e300, -1e300), (-2.0, -0.5), (inf, inf)]]
pairs = list(itertools.product(Z, S))
z = np.array([p[0] for p in pairs]); s = np.array([p[1] for p in pairs])
mixed = []
for name, sym, zr, rr in [("z_add_s", "z + s", z + s, None), ("z_sub_s", "z - s", z - s, None), ("z_mul_s", "z * s", z * s, None),
                          ("z_div_s", "z / s", z / s, None), ("s_add_z", "s + z", s + z, None), ("s_sub_z", "s - z", s - z, None),
                          ("s_mul_z", "s * z", s * z, None), ("s_div_z", "s / z", s / z, None)]:
    out = ["    const CASES: &[((f64, f64), f64, (f64, f64))] = &["]
    for (zz, ss), r in zip(pairs, zr):
        out.append(f"        (({lit(zz.real)}, {lit(zz.imag)}), {lit(ss)}, ({lit(r.real.item())}, {lit(r.imag.item())})),")
    out.append("    ];")
    mixed.append(f"""
#[test]
fn complex_{name}() {{
{chr(10).join(out)}
    for &((re, im), s, want) in CASES {{
        let z = Complex::new(re, im);
        check_c("{name}", &format!("{{z:?}}, {{s:?}}"), {sym}, want, 1e-13, false);
    }}
}}""")
c += mixed
# scalar assign ops
c.append("""
#[test]
fn complex_scalar_assign_ops() {
    let mut z = Complex::new(1.0, 2.0);
    z += 3.0;
    assert_eq!(z, Complex::new(4.0, 2.0));
    z -= 1.0;
    assert_eq!(z, Complex::new(3.0, 2.0));
    z *= 2.0;
    assert_eq!(z, Complex::new(6.0, 4.0));
    z /= 2.0;
    assert_eq!(z, Complex::new(3.0, 2.0));
    z *= Complex::i();
    assert_eq!(z, Complex::new(-2.0, 3.0));
    assert_eq!(-z, Complex::new(2.0, -3.0));
    assert_eq!(Complex::<f32>::default(), Complex::new(0.0, 0.0));
    assert_eq!(Complex::from_real(2.5f64), Complex::new(2.5, 0.0));
    assert_eq!(2.0f32 * Complex::new(1.0f32, -1.0), Complex::new(2.0, -2.0));
}

#[test]
fn complex_nan_is_not_equal_to_itself() {
    let z = Complex::new(f64::NAN, 1.0);
    assert_ne!(z, z);
    assert_eq!(Complex::new(0.0, -0.0), Complex::new(-0.0, 0.0));
}

#[test]
fn complex_imaginary_unit_squares_to_minus_one() {
    let i = Complex::<f64>::i();
    assert_eq!(i * i, Complex::new(-1.0, 0.0));
    assert_eq!(Complex::<f32>::i().norm_sqr(), 1.0);
}

#[test]
fn complex_from_polar_roundtrip() {
    let z = Complex::from_polar(2.0f64, core::f64::consts::FRAC_PI_3);
    assert!((z.norm() - 2.0).abs() < 1e-15);
    assert!((z.arg() - core::f64::consts::FRAC_PI_3).abs() < 1e-15);
    assert!((Complex::new(3.0f64, 4.0).norm_sqr() - 25.0).abs() == 0.0);
}""")

# Inputs where NumPy on this platform is wrong or platform dependent: use mpmath (50 digits).
import mpmath
mpmath.mp.dps = 50


def mp_rows(fn, zs):
    rows = []
    for z in zs:
        r = fn(mpmath.mpc(z.real, z.imag))
        rows.append(((z.real, z.imag), (float(r.real), float(r.imag))))
    return rows


for fname, mpf in (("exp", mpmath.exp), ("sinh", mpmath.sinh), ("cosh", mpmath.cosh), ("sqrt", mpmath.sqrt)):
    rows = mp_rows(mpf, DIVERGENT[fname])
    c.append(f"""
#[test]
fn complex_{fname}_overflow_edge_matches_mpmath() {{
    // NumPy 2.3 on Windows returns inf (exp/sinh/cosh: intermediate overflow) or a result that is
    // off by a factor 2 (sqrt: scaled path). These are the exact values from mpmath.
{ctable("CASES", rows, "f64")}
    for &((re, im), want) in CASES {{
        let z = Complex::new(re, im);
        check_c("{fname}", &format!("{{z:?}}"), z.{fname}(), want, 1e-14, false);
    }}
}}""")
c.append("""
#[test]
fn complex_mul_overflow_is_plain_ieee() {
    // (1e300+1e300j)^2 = 0 + 2e600j. Plain IEEE evaluation gives (NaN, inf); NumPy builds with
    // FMA loops give (-inf, inf). Both agree that the imaginary part overflows.
    let z = Complex::new(1e300f64, 1e300);
    let p = z * z;
    assert_eq!(p.im, f64::INFINITY);
    assert!(!p.re.is_finite());
}""")

c.append("""
#[test]
fn complex_branch_cuts_follow_the_sign_of_zero() {
    use core::f64::consts::PI;
    // sqrt: the cut is the negative real axis
    let up = Complex::new(-4.0, 0.0).sqrt();
    let down = Complex::new(-4.0, -0.0).sqrt();
    assert_eq!((up.re, up.im), (0.0, 2.0));
    assert!(down.re == 0.0 && down.im == -2.0);
    // ln: imaginary part is +pi above the cut and -pi below it
    assert_eq!(Complex::new(-1.0, 0.0).ln(), Complex::new(0.0, PI));
    assert_eq!(Complex::new(-1.0, -0.0).ln(), Complex::new(0.0, -PI));
    assert_eq!(Complex::new(0.0, 0.0).ln(), Complex::new(f64::NEG_INFINITY, 0.0));
    assert_eq!(Complex::new(-0.0, 0.0).ln().im, PI);
    // sqrt(-0 + 0j) keeps the zeros; sqrt(0) is 0
    let z = Complex::new(-0.0f64, 0.0).sqrt();
    assert!(z.re == 0.0 && !z.re.is_sign_negative() && z.im == 0.0);
    let w = Complex::new(0.0f64, -0.0).sqrt();
    assert!(w.im.is_sign_negative());
}

#[test]
fn complex_hand_checked_values() {
    use core::f64::consts::{E, FRAC_PI_2, PI};
    let e_ipi = Complex::new(0.0, PI).exp();
    assert!((e_ipi.re + 1.0).abs() < 1e-15 && e_ipi.im.abs() < 2e-16);
    let i_pow_i = Complex::<f64>::i().powc(Complex::i());
    assert!((i_pow_i.re - (-FRAC_PI_2).exp()).abs() < 1e-15 && i_pow_i.im.abs() < 1e-15);
    let z = Complex::new(3.0, 4.0);
    assert_eq!(z.norm(), 5.0);
    assert_eq!(z.sqrt(), Complex::new(2.0, 1.0));
    assert_eq!(z * z.conj(), Complex::new(25.0, 0.0));
    assert_eq!(z / z, Complex::new(1.0, 0.0));
    assert_eq!(z.powf(2.0), Complex::new(-7.0, 24.0));
    assert_eq!(z.powc(Complex::new(0.0, 0.0)), Complex::new(1.0, 0.0));
    let l = Complex::new(E, 0.0).ln();
    assert!((l.re - 1.0).abs() < 1e-15 && l.im == 0.0);
    // sin^2 + cos^2 = 1 for a complex argument
    let a = Complex::new(0.7f64, -1.3);
    let s = a.sin() * a.sin() + a.cos() * a.cos();
    assert!((s.re - 1.0).abs() < 1e-14 && s.im.abs() < 1e-14);
    // cosh^2 - sinh^2 = 1
    let h = a.cosh() * a.cosh() - a.sinh() * a.sinh();
    assert!((h.re - 1.0).abs() < 1e-14 && h.im.abs() < 1e-14);
    // tanh = sinh / cosh
    let t = a.tanh();
    let q = a.sinh() / a.cosh();
    assert!((t.re - q.re).abs() < 1e-14 && (t.im - q.im).abs() < 1e-14);
}
""")

# powers
PW = [(0, 0), (0, 2), (0, -1), (0, 1 + 1j), (0, 0.5), (0, 0.0 + 1e-300j), (1 + 2j, 0), (1 + 2j, 1), (1 + 2j, 2), (1 + 2j, 3), (1 + 2j, 4),
      (1 + 2j, -1), (1 + 2j, -2), (1 + 2j, 5), (1 + 2j, 7), (1 + 2j, 10), (1 + 2j, 99), (1 + 2j, -99), (1 + 2j, 100), (1 + 2j, 101),
      (1 + 2j, 0.5), (1 + 2j, 1.5), (1 + 2j, 0.5 + 0.5j), (2j, 0.5), (-1 + 0j, 0.5), (complex(-1, -0.0), 0.5), (-8 + 0j, 1 / 3),
      (3 + 4j, 2 + 0j), (complex(inf, 0), 2), (complex(inf, 1), 3), (nan, 2), (1 + 1j, nan), (1 + 1j, 1e10), (2 + 0j, 10), (2 + 0j, -10),
      (2 + 0j, 0.5), (0.5 + 0.5j, 2.5), (1e100 + 1e100j, 2), (1e200 + 1e200j, 3), (-2 - 3j, 3 + 2j), (1j, 1j), (np.e, 1j * np.pi), (10 + 0j, 300)]
for ty in ("f64",):
    a = cvals([p[0] for p in PW], ty); b = cvals([p[1] for p in PW], ty)
    res = np.power(a, b)
    out = [f"    const CASES: &[(({ty}, {ty}), ({ty}, {ty}), ({ty}, {ty}))] = &["]
    for x, y, r in zip(a, b, res):
        out.append(f"        (({lit(x.real.item())}, {lit(x.imag.item())}), ({lit(y.real.item())}, {lit(y.imag.item())}), ({lit(r.real.item())}, {lit(r.imag.item())})),")
    out.append("    ];")
    c.append(f"""
#[test]
fn complex_powc_f64() {{
{chr(10).join(out)}
    for &((ar, ai), (br, bi), want) in CASES {{
        let (a, b) = (Complex::new(ar, ai), Complex::new(br, bi));
        check_c("powc", &format!("{{a:?}}, {{b:?}}"), a.powc(b), want, 1e-11, !(finite_input(a) && finite_input(b)));
    }}
}}""")
    # powf: real exponents via np.power(complex, float)
    rp = [p for p in PW if isinstance(p[1], (int, float)) or p[1].imag == 0]
    a = cvals([p[0] for p in rp], ty)
    bf = np.array([complex(p[1]).real for p in rp])
    res = np.power(a, bf)
    out = [f"    const CASES: &[(({ty}, {ty}), {ty}, ({ty}, {ty}))] = &["]
    for x, y, r in zip(a, bf, res):
        out.append(f"        (({lit(x.real.item())}, {lit(x.imag.item())}), {lit(y)}, ({lit(r.real.item())}, {lit(r.imag.item())})),")
    out.append("    ];")
    c.append(f"""
#[test]
fn complex_powf_f64() {{
{chr(10).join(out)}
    for &((ar, ai), p, want) in CASES {{
        let a = Complex::new(ar, ai);
        check_c("powf", &format!("{{a:?}}, {{p:?}}"), a.powf(p), want, 1e-11, !(finite_input(a) && p.is_finite()));
    }}
}}""")
# powc f32 integer powers
pw32 = [(1 + 2j, 2), (1 + 2j, 3), (1 + 2j, 5), (1 + 2j, -2), (0.5 + 0.25j, 0.5), (2 + 0j, 10)]
a = cvals([p[0] for p in pw32], "f32"); b = cvals([p[1] for p in pw32], "f32")
res = np.power(a, b)
out = ["    const CASES: &[((f32, f32), (f32, f32), (f32, f32))] = &["]
for x, y, r in zip(a, b, res):
    out.append(f"        (({lit(x.real.item(), 'f32')}, {lit(x.imag.item(), 'f32')}), ({lit(y.real.item(), 'f32')}, {lit(y.imag.item(), 'f32')}), ({lit(r.real.item(), 'f32')}, {lit(r.imag.item(), 'f32')})),")
out.append("    ];")
c.append(f"""
#[test]
fn complex_powc_f32() {{
{chr(10).join(out)}
    for &((ar, ai), (br, bi), want) in CASES {{
        let (a, b) = (Complex::new(ar, ai), Complex::new(br, bi));
        check_c("powc", &format!("{{a:?}}, {{b:?}}"), a.powc(b), want, 2e-4, false);
    }}
}}""")

# display
disp = [1 + 2j, -0.5 - 1j, 2j, -2j, 0j, complex(-0.0, 1), complex(0, -0.0), complex(1, -0.0), 1e16 + 1j, 1e15 + 1j, 1e-5 + 1e-4j, 0.0001 + 1j, 123456789.123 - 2.5e-7j,
        complex(nan, nan), complex(inf, -inf), complex(0, nan), complex(-inf, 0), complex(nan, 1), 1.5 + 0j, 100 + 100j, 0.1 + 0.2j, 1e22 - 1e-22j, 5e-324 + 1.7976931348623157e308j,
        3 - 4j, -1e300 + 1e-300j, 12345678901234567890.0 + 1j, 2.5 + 3.5j, 1 / 3 + 2 / 3j]
rows = [(z.real, z.imag, str(np.complex128(z))) for z in disp]
c.append("""
#[test]
fn complex_display_f64() {
    const CASES: &[(f64, f64, &str)] = &[
""" + "\n".join(f"        ({lit(r)}, {lit(i)}, {s!r}),".replace("'", '"') for r, i, s in rows) + """
    ];
    for &(re, im, want) in CASES {
        assert_eq!(Complex::new(re, im).to_string(), want, "Display of ({re:?}, {im:?})");
    }
}""")
disp32 = [1 + 2j, -0.5 - 1j, 2j, 0.1 + 0.2j, 1e16 + 1j, 3.4028235e38 - 1e-45j, complex(nan, inf), 1 / 3 + 1j, 16777216 + 0.5j]
rows = [(np.complex64(z).real.item(), np.complex64(z).imag.item(), str(np.complex64(z))) for z in disp32]
c.append("""
#[test]
fn complex_display_f32() {
    const CASES: &[(f32, f32, &str)] = &[
""" + "\n".join(f"        ({lit(r, 'f32')}, {lit(i, 'f32')}, {s!r}),".replace("'", '"') for r, i, s in rows) + """
    ];
    for &(re, im, want) in CASES {
        assert_eq!(Complex::new(re, im).to_string(), want, "Display of ({re:?}, {im:?})");
    }
}""")

# sort order
sort_in = [complex(a, b) for a in (nan, 2.0, -1.0, inf, -inf, 0.0, -0.0) for b in (nan, 1.0, -3.0, inf, 0.0)]
rng = np.random.default_rng(7)
perm = rng.permutation(len(sort_in))
sort_in = [sort_in[i] for i in perm]
arr = np.array(sort_in)
srt = np.sort(arr)
args = np.argsort(arr, kind="stable")
c.append(f"""
#[test]
fn complex_cmp_numpy_matches_np_sort() {{
    const INPUT: &[(f64, f64)] = &[{", ".join(f"({lit(z.real)}, {lit(z.imag)})" for z in sort_in)}];
    const SORTED: &[(f64, f64)] = &[{", ".join(f"({lit(z.real)}, {lit(z.imag)})" for z in srt)}];
    let mut v: Vec<Complex<f64>> = INPUT.iter().map(|&(r, i)| Complex::new(r, i)).collect();
    v.sort_by(|a, b| a.cmp_numpy(b));
    assert_eq!(v.len(), SORTED.len());
    for (k, (z, &(wr, wi))) in v.iter().zip(SORTED).enumerate() {{
        // equal values (0.0 vs -0.0) may come out in either order, NumPy's sort is not stable
        let same = |g: f64, w: f64| (g.is_nan() && w.is_nan()) || g == w;
        assert!(same(z.re, wr) && same(z.im, wi), "position {{k}}: got {{z:?}}, NumPy says ({{wr:?}}, {{wi:?}})");
    }}
}}
""")
# comparison table: np's sort uses <; test pairwise ordering from NumPy's maximum/lexicographic via argsort ranks
pts = [complex(nan, nan), complex(nan, 1), complex(1, nan), complex(1, 1), complex(1, 2), complex(0, 5), complex(inf, -inf), complex(-inf, nan), complex(2, 0), complex(-0.0, 0.0)]
rows = []
for a, b in itertools.product(pts, pts):
    arr = np.array([a, b])
    # NumPy's rank: sort 2 elements; a<b iff sorted order is (a, b) and they are not identical
    s = np.sort(arr)
    def same(x, y):
        return (x.real == y.real or (np.isnan(x.real) and np.isnan(y.real))) and (x.imag == y.imag or (np.isnan(x.imag) and np.isnan(y.imag)))
    if same(a, b):
        o = 0
    else:
        o = -1 if same(s[0], a) else 1
    rows.append((a, b, o))
c.append(f"""
#[test]
fn complex_cmp_numpy_pairs() {{
    const CASES: &[((f64, f64), (f64, f64), i32)] = &[{", ".join(f"(({lit(a.real)}, {lit(a.imag)}), ({lit(b.real)}, {lit(b.imag)}), {o})" for a, b, o in rows)}];
    for &((ar, ai), (br, bi), want) in CASES {{
        let got = Complex::new(ar, ai).cmp_numpy(&Complex::new(br, bi)) as i32;
        assert_eq!(got, want, "cmp_numpy(({{ar:?}}, {{ai:?}}), ({{br:?}}, {{bi:?}}))");
    }}
}}""")

complex_file = """//! NumPy parity tests for `Complex<F>` (GENERATED ONCE with NumPy {ver}).
#![allow(clippy::approx_constant, clippy::excessive_precision, clippy::type_complexity)]

use super::complex::Complex;
use super::tests_util::{{check, check_c, finite_input, Report}};
""".format(ver=np.__version__) + "\n".join(c) + "\n"

math_file = HEADER + "".join(m) + "\n"

import re


def add_report(t):
    return re.sub(r"(#\[test\]\nfn \w+\(\) \{\n)", lambda mo: mo.group(1) + "    let _report = Report;\n", t)


math_file = add_report(math_file)
complex_file = add_report(complex_file)
window_file = add_report(window_file).replace("use super::window_kernels::*;", "use super::tests_util::Report;\nuse super::window_kernels::*;")

open(f"{OUT}/tests_math.rs", "w", newline="\n").write(math_file)
open(f"{OUT}/tests_window.rs", "w", newline="\n").write(window_file)
open(f"{OUT}/tests_complex.rs", "w", newline="\n").write(complex_file)
print("written")
