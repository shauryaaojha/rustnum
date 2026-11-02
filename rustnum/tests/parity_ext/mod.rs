//! Parity cases for the v0.2 dtypes (complex, float16, indexing, masked arrays, datetime,
//! structured). `parity.rs` falls back to [`run`] for ops it does not know and to [`check`] to
//! compare the result with the NumPy output stored next to the inputs.

use super::{Json, exact_bool, param_f64, param_isize, param_usize};
use rustnum as np;
use rustnum::{Array, Complex};
use std::fs;
use std::path::Path;

mod dt;
mod fancy;
mod half;
mod ma;

/// What an extension case computed.
pub enum Out {
    F64(Array<f64>),
    I64(Array<i64>),
    Bool(Array<bool>),
    C128(Array<Complex<f64>>),
    C64(Array<Complex<f32>>),
    F16(Array<rustnum::f16>),
    Str(String),
}

fn bytes(path: &Path) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// Reads the `descr` entry of a `.npy` header.
pub fn descr_of(path: &Path) -> String {
    let b = bytes(path);
    let head = String::from_utf8_lossy(&b[..b.len().min(400)]).to_string();
    let i = head.find("'descr'").expect("descr key");
    let rest = &head[i + 7..];
    let q = rest.find('\'').unwrap();
    let rest = &rest[q + 1..];
    rest[..rest.find('\'').unwrap()].to_string()
}

fn load<T: np::Element>(path: &Path) -> Array<T> {
    np::io::from_npy_bytes::<T>(&bytes(path))
        .unwrap_or_else(|e| panic!("cannot decode {}: {e}", path.display()))
}

/// An input that is real or complex, as stored by NumPy.
#[derive(Clone)]
enum Val {
    R(Array<f64>),
    C(Array<Complex<f64>>),
    C64(Array<Complex<f32>>),
}

fn val(path: &Path) -> Val {
    match descr_of(path).as_str() {
        "<c16" => Val::C(load(path)),
        "<c8" => Val::C64(load(path)),
        _ => Val::R(load(path)),
    }
}

fn vals(dir: &Path, case: &Json) -> Vec<Val> {
    case.get("inputs")
        .as_arr()
        .iter()
        .map(|f| val(&dir.join(f.as_str())))
        .collect()
}

fn c(v: &Val) -> Array<Complex<f64>> {
    match v {
        Val::C(a) => a.clone(),
        _ => panic!("expected complex128 input"),
    }
}

macro_rules! binary {
    ($v:expr, $op:tt) => {
        match (&$v[0], &$v[1]) {
            (Val::C(a), Val::C(b)) => Out::C128(a $op b),
            (Val::C(a), Val::R(b)) => Out::C128(a $op b),
            (Val::R(a), Val::C(b)) => Out::C128(a $op b),
            _ => panic!("unsupported operand combination"),
        }
    };
}

/// Runs an extension op; `None` when `op` belongs to the base runner.
pub fn run(dir: &Path, case: &Json) -> Option<Result<Out, String>> {
    if let Some(r) = half::run(dir, case) {
        return Some(r);
    }
    if let Some(r) = dt::run(dir, case) {
        return Some(r);
    }
    if let Some(r) = fancy::run(dir, case) {
        return Some(r);
    }
    if let Some(r) = ma::run(dir, case) {
        return Some(r);
    }
    let op = case.get("op").as_str();
    if !op.starts_with("c_") && !op.starts_with("c64_") {
        return None;
    }
    let v = vals(dir, case);
    Some(Ok(match op {
        "c_add" => binary!(v, +),
        "c_sub" => binary!(v, -),
        "c_mul" => binary!(v, *),
        "c_div" => binary!(v, /),
        "c_neg" => Out::C128(-&c(&v[0])),
        "c_exp" => Out::C128(c(&v[0]).exp()),
        "c_log" => Out::C128(c(&v[0]).log()),
        "c_sqrt" => Out::C128(c(&v[0]).sqrt()),
        "c_sin" => Out::C128(c(&v[0]).sin()),
        "c_cos" => Out::C128(c(&v[0]).cos()),
        "c_tan" => Out::C128(c(&v[0]).tan()),
        "c_sinh" => Out::C128(c(&v[0]).sinh()),
        "c_cosh" => Out::C128(c(&v[0]).cosh()),
        "c_tanh" => Out::C128(c(&v[0]).tanh()),
        "c_square" => Out::C128(c(&v[0]).square()),
        "c_conj" => Out::C128(c(&v[0]).conj()),
        "c_abs" => Out::F64(np::absolute(&c(&v[0]))),
        "c_angle" => Out::F64(c(&v[0]).angle()),
        "c_real" => Out::F64(c(&v[0]).real()),
        "c_imag" => Out::F64(c(&v[0]).imag()),
        "c_power" => Out::C128(np::power(c(&v[0]), c(&v[1]))),
        "c_sum" => Out::C128(Array::scalar(c(&v[0]).sum())),
        "c_prod" => Out::C128(Array::scalar(c(&v[0]).prod())),
        "c_mean" => Out::C128(Array::scalar(c(&v[0]).mean())),
        "c_sum_axis" => Out::C128(c(&v[0]).sum_axis(param_isize(case, "axis"))),
        "c_mean_axis" => Out::C128(c(&v[0]).mean_axis(param_isize(case, "axis"))),
        "c_cumsum" => Out::C128(c(&v[0]).cumsum()),
        "c_max" => Out::C128(Array::scalar(c(&v[0]).max())),
        "c_min" => Out::C128(Array::scalar(c(&v[0]).min())),
        "c_sort" => Out::C128(c(&v[0]).sort()),
        "c_argsort" => {
            let r = c(&v[0]).argsort();
            Out::I64(r.astype::<i64>())
        }
        "c_matmul" => Out::C128(c(&v[0]).matmul(&c(&v[1]))),
        "c_dot" => Out::C128(c(&v[0]).dot(&c(&v[1]))),
        "c_vdot" => Out::C128(Array::scalar(c(&v[0]).vdot(&c(&v[1])))),
        "c_fft" => match &v[0] {
            Val::C(a) => Out::C128(np::fft::fft(a)),
            Val::R(a) => Out::C128(np::fft::fft(a)),
            Val::C64(_) => unreachable!(),
        },
        "c_ifft" => Out::C128(np::fft::ifft(&c(&v[0]))),
        "c_rfft" => match &v[0] {
            Val::R(a) => Out::C128(np::fft::rfft(a)),
            _ => unreachable!(),
        },
        "c_irfft" => {
            let n = case.get("params").as_obj().iter().any(|(k, _)| k == "n");
            Out::F64(np::fft::irfft(&c(&v[0]), n.then(|| param_usize(case, "n"))))
        }
        "c_fft2" => Out::C128(np::fft::fft2(&c(&v[0]))),
        "c_ifft2" => Out::C128(np::fft::ifft2(&c(&v[0]))),
        "c64_fft" => match &v[0] {
            Val::C64(a) => Out::C64(np::fft::fft(a)),
            _ => unreachable!(),
        },
        "c64_mul" => match (&v[0], &v[1]) {
            (Val::C64(a), Val::C64(b)) => Out::C64(a * b),
            _ => unreachable!(),
        },
        "c_to_c64" => Out::C64(c(&v[0]).astype::<Complex<f32>>()),
        "c_from_real" => match &v[0] {
            Val::R(a) => Out::C128(a.astype::<Complex<f64>>()),
            _ => unreachable!(),
        },
        "c_isnan" => Out::Bool(c(&v[0]).is_nan()),
        "c_str" => Out::Str(display_of(&v[0])),
        _ => return Some(Err(format!("skipped: op `{op}` not wired"))),
    }))
}

fn display_of(v: &Val) -> String {
    match v {
        Val::C(a) => a.to_string(),
        Val::C64(a) => a.to_string(),
        Val::R(a) => a.to_string(),
    }
}

fn close_parts<T: np::Float>(
    a: &Array<Complex<T>>,
    b: &Array<Complex<T>>,
    rtol: f64,
    atol: f64,
) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    let ok = |x: T, y: T| {
        let (x, y) = (x.to_f64(), y.to_f64());
        (x.is_nan() && y.is_nan()) || x == y || (x - y).abs() <= atol + rtol * y.abs()
    };
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| ok(x.re, y.re) && ok(x.im, y.im))
}

/// Element-wise comparison where equal values (infinities included) and NaN pairs match.
fn close_f64(a: &Array<f64>, b: &Array<f64>, rtol: f64, atol: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| {
        (x.is_nan() && y.is_nan()) || x == y || (x - y).abs() <= atol + rtol * y.abs()
    })
}

/// Integer comparison that reports the first differences.
fn exact_i64_verbose(a: &Array<i64>, b: &Array<i64>) -> bool {
    if a.shape() != b.shape() {
        eprintln!("  shape mismatch {:?} vs {:?}", a.shape(), b.shape());
        return false;
    }
    let bad: Vec<(usize, i64, i64)> = a
        .iter()
        .zip(b.iter())
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, (x, y))| (i, x, y))
        .take(5)
        .collect();
    for (i, x, y) in &bad {
        eprintln!("  index {i}: got {x:#x}, want {y:#x}");
    }
    bad.is_empty()
}

/// Compares an extension result with the NumPy output file.
pub fn check(out: Out, case: &Json, output_file: &Path) -> bool {
    let rtol = case.get("rtol").as_f64();
    let atol = case.get("atol").as_f64();
    match out {
        Out::F64(a) => close_f64(&a, &load::<f64>(output_file), rtol, atol),
        Out::I64(a) => exact_i64_verbose(&a, &load::<i64>(output_file)),
        Out::Bool(a) => exact_bool(&a, &load::<bool>(output_file)),
        Out::C128(a) => close_parts(&a, &load::<Complex<f64>>(output_file), rtol, atol),
        Out::C64(a) => close_parts(&a, &load::<Complex<f32>>(output_file), rtol, atol),
        Out::F16(a) => {
            let want = load::<rustnum::f16>(output_file);
            close_f64(&a.astype::<f64>(), &want.astype::<f64>(), rtol, atol)
        }
        Out::Str(s) => {
            let want = case.get("params").get("str").as_str();
            if s != want {
                eprintln!("  string mismatch:\n  got:  {s:?}\n  want: {want:?}");
            }
            s == want
        }
    }
}
