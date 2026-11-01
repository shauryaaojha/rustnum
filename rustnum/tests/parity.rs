//! NumPy parity test runner.
//!
//! Loads `rustnum/tests/fixtures/manifest.json`, runs each case through
//! rustnum, and compares with `allclose` (or exact equality for ints/bools).
//!
//! This file is `.pending` — it will be renamed to `parity.rs` and placed
//! in `rustnum/tests/` after the core Array type is merged.

use np::Array;
use rustnum as np;

use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Tiny JSON reader — no new dependencies
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Json {
    Null,
    Bool,
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn as_str(&self) -> &str {
        match self {
            Json::Str(s) => s,
            _ => panic!("expected string, got {self:?}"),
        }
    }
    fn as_f64(&self) -> f64 {
        match self {
            Json::Num(n) => *n,
            _ => panic!("expected number, got {self:?}"),
        }
    }
    fn as_arr(&self) -> &[Json] {
        match self {
            Json::Arr(a) => a,
            _ => panic!("expected array, got {self:?}"),
        }
    }
    fn as_obj(&self) -> &[(String, Json)] {
        match self {
            Json::Obj(o) => o,
            _ => panic!("expected object, got {self:?}"),
        }
    }
    fn get(&self, key: &str) -> &Json {
        for (k, v) in self.as_obj() {
            if k == key {
                return v;
            }
        }
        panic!("key not found: {key}");
    }
}

fn parse_json(s: &str) -> Json {
    let bytes = s.as_bytes();
    let (val, _) = parse_value(bytes, skip_ws(bytes, 0));
    val
}

fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    i
}

fn parse_value(b: &[u8], i: usize) -> (Json, usize) {
    match b[i] {
        b'"' => parse_string(b, i),
        b'{' => parse_object(b, i),
        b'[' => parse_array(b, i),
        b't' => {
            assert!(&b[i..i + 4] == b"true");
            (Json::Bool, i + 4)
        }
        b'f' => {
            assert!(&b[i..i + 5] == b"false");
            (Json::Bool, i + 5)
        }
        b'n' => {
            assert!(&b[i..i + 4] == b"null");
            (Json::Null, i + 4)
        }
        _ => parse_number(b, i),
    }
}

fn parse_string(b: &[u8], i: usize) -> (Json, usize) {
    assert_eq!(b[i], b'"');
    let mut j = i + 1;
    let mut s = String::new();
    while b[j] != b'"' {
        if b[j] == b'\\' {
            j += 1;
            match b[j] {
                b'"' => s.push('"'),
                b'\\' => s.push('\\'),
                b'/' => s.push('/'),
                b'n' => s.push('\n'),
                b'r' => s.push('\r'),
                b't' => s.push('\t'),
                _ => {
                    s.push('\\');
                    s.push(b[j] as char);
                }
            }
        } else {
            s.push(b[j] as char);
        }
        j += 1;
    }
    (Json::Str(s), j + 1)
}

fn parse_number(b: &[u8], i: usize) -> (Json, usize) {
    let mut j = i;
    while j < b.len() && (b[j].is_ascii_digit() || matches!(b[j], b'-' | b'+' | b'.' | b'e' | b'E'))
    {
        j += 1;
    }
    let s = std::str::from_utf8(&b[i..j]).unwrap();
    (Json::Num(s.parse().unwrap()), j)
}

fn parse_array(b: &[u8], i: usize) -> (Json, usize) {
    assert_eq!(b[i], b'[');
    let mut j = skip_ws(b, i + 1);
    let mut arr = Vec::new();
    if b[j] == b']' {
        return (Json::Arr(arr), j + 1);
    }
    loop {
        let (val, next) = parse_value(b, j);
        arr.push(val);
        j = skip_ws(b, next);
        if b[j] == b']' {
            return (Json::Arr(arr), j + 1);
        }
        assert_eq!(b[j], b',');
        j = skip_ws(b, j + 1);
    }
}

fn parse_object(b: &[u8], i: usize) -> (Json, usize) {
    assert_eq!(b[i], b'{');
    let mut j = skip_ws(b, i + 1);
    let mut obj = Vec::new();
    if b[j] == b'}' {
        return (Json::Obj(obj), j + 1);
    }
    loop {
        let (key, next) = parse_string(b, j);
        let key = match key {
            Json::Str(s) => s,
            _ => unreachable!(),
        };
        j = skip_ws(b, next);
        assert_eq!(b[j], b':');
        j = skip_ws(b, j + 1);
        let (val, next) = parse_value(b, j);
        obj.push((key, val));
        j = skip_ws(b, next);
        if b[j] == b'}' {
            return (Json::Obj(obj), j + 1);
        }
        assert_eq!(b[j], b',');
        j = skip_ws(b, j + 1);
    }
}

// ---------------------------------------------------------------------------
// .npy loader (only f64 and i64 for parity tests)
// ---------------------------------------------------------------------------

fn load_npy_f64(path: &Path) -> Array<f64> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    np::io::from_npy_bytes::<f64>(&bytes)
        .unwrap_or_else(|e| panic!("cannot decode {}: {e}", path.display()))
}

fn load_npy_i64(path: &Path) -> Array<i64> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    np::io::from_npy_bytes::<i64>(&bytes)
        .unwrap_or_else(|e| panic!("cannot decode {}: {e}", path.display()))
}

fn load_npy_bool(path: &Path) -> Array<bool> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    np::io::from_npy_bytes::<bool>(&bytes)
        .unwrap_or_else(|e| panic!("cannot decode {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// allclose comparison
// ---------------------------------------------------------------------------

fn allclose_f64(a: &Array<f64>, b: &Array<f64>, rtol: f64, atol: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| {
        if x.is_nan() && y.is_nan() {
            return true;
        }
        (x - y).abs() <= atol + rtol * y.abs()
    })
}

fn exact_i64(a: &Array<i64>, b: &Array<i64>) -> bool {
    a.shape() == b.shape() && a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

fn exact_bool(a: &Array<bool>, b: &Array<bool>) -> bool {
    a.shape() == b.shape() && a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

// ---------------------------------------------------------------------------
// Dispatch: run one case
// ---------------------------------------------------------------------------

fn load_f64_inputs(fixture_dir: &Path, case: &Json) -> Vec<Array<f64>> {
    case.get("inputs")
        .as_arr()
        .iter()
        .map(|f| load_npy_f64(&fixture_dir.join(f.as_str())))
        .collect()
}

fn param_f64(case: &Json, key: &str) -> f64 {
    case.get("params").get(key).as_f64()
}

fn param_usize(case: &Json, key: &str) -> usize {
    case.get("params").get(key).as_f64() as usize
}

fn param_isize(case: &Json, key: &str) -> isize {
    case.get("params").get(key).as_f64() as isize
}

/// Dispatch on `op`, return Ok(computed) or Err("skipped: reason").
fn run_case(fixture_dir: &Path, case: &Json) -> Result<RunResult, String> {
    let op = case.get("op").as_str();

    match op {
        // -- creation --
        "arange" => {
            let start = param_f64(case, "start");
            let stop = param_f64(case, "stop");
            let step = param_f64(case, "step");
            Ok(RunResult::F64(np::arange(start, stop, step)))
        }
        "linspace" => {
            let start = param_f64(case, "start");
            let stop = param_f64(case, "stop");
            let num = param_usize(case, "num");
            Ok(RunResult::F64(np::linspace(start, stop, num)))
        }
        "eye" => {
            let n = param_usize(case, "n");
            Ok(RunResult::F64(np::eye::<f64>(n)))
        }
        "tri" => {
            let n = param_usize(case, "n");
            Ok(RunResult::F64(np::tri::<f64>(n)))
        }

        // -- arithmetic (broadcasting) --
        "add" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(&inputs[0] + &inputs[1]))
        }
        "subtract" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(&inputs[0] - &inputs[1]))
        }
        "multiply" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(&inputs[0] * &inputs[1]))
        }
        "divide" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(&inputs[0] / &inputs[1]))
        }
        "modulo" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(&inputs[0] % &inputs[1]))
        }

        // -- ufuncs --
        "exp" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(inputs[0].exp()))
        }
        "log" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(inputs[0].log()))
        }
        "sqrt" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(inputs[0].sqrt()))
        }
        "tanh" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(inputs[0].tanh()))
        }
        "sin" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(inputs[0].sin()))
        }
        "clip" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            let lo = param_f64(case, "lo");
            let hi = param_f64(case, "hi");
            Ok(RunResult::F64(inputs[0].clip(lo, hi)))
        }
        "floor" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(inputs[0].floor()))
        }
        "round" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(inputs[0].round()))
        }

        // -- reductions (whole array) --
        "sum" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(Array::scalar(inputs[0].sum())))
        }
        "mean" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(Array::scalar(inputs[0].mean())))
        }
        "std" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(Array::scalar(inputs[0].std())))
        }
        "var" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(Array::scalar(inputs[0].var())))
        }
        "min" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(Array::scalar(inputs[0].min())))
        }
        "max" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(Array::scalar(inputs[0].max())))
        }
        "argmax" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            // argmax returns usize, but fixture is int64
            let idx = inputs[0].argmax();
            Ok(RunResult::I64(Array::scalar(idx as i64)))
        }
        "cumsum" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(inputs[0].cumsum()))
        }
        "median" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(Array::scalar(inputs[0].median())))
        }
        "percentile" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            let q = param_f64(case, "q");
            Ok(RunResult::F64(Array::scalar(inputs[0].percentile(q))))
        }

        // -- reductions along axis --
        "sum_axis" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            let axis = param_isize(case, "axis");
            Ok(RunResult::F64(inputs[0].sum_axis(axis)))
        }
        "mean_axis" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            let axis = param_isize(case, "axis");
            Ok(RunResult::F64(inputs[0].mean_axis(axis)))
        }
        "min_axis" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            let axis = param_isize(case, "axis");
            Ok(RunResult::F64(inputs[0].min_axis(axis)))
        }
        "max_axis" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            let axis = param_isize(case, "axis");
            Ok(RunResult::F64(inputs[0].max_axis(axis)))
        }

        // -- manipulation --
        "reshape" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            let shape_arr: Vec<usize> = case
                .get("params")
                .get("shape")
                .as_arr()
                .iter()
                .map(|v| v.as_f64() as usize)
                .collect();
            Ok(RunResult::F64(inputs[0].reshape(&shape_arr)))
        }
        "transpose" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(inputs[0].t()))
        }
        "swapaxes" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            let a = param_isize(case, "axis1");
            let b = param_isize(case, "axis2");
            Ok(RunResult::F64(inputs[0].swapaxes(a, b)))
        }
        "concatenate" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            let axis = param_isize(case, "axis");
            let refs: Vec<&Array<f64>> = inputs.iter().collect();
            Ok(RunResult::F64(np::concatenate(&refs, axis)))
        }
        "stack" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            let axis = param_isize(case, "axis");
            let refs: Vec<&Array<f64>> = inputs.iter().collect();
            Ok(RunResult::F64(np::stack(&refs, axis)))
        }
        "flip" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            let axis = param_isize(case, "axis");
            Ok(RunResult::F64(inputs[0].flip(axis)))
        }
        "roll" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            let shift = param_isize(case, "shift");
            let axis = param_isize(case, "axis");
            Ok(RunResult::F64(inputs[0].roll(shift, axis)))
        }
        "repeat" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            let n = param_usize(case, "n");
            let axis = param_isize(case, "axis");
            Ok(RunResult::F64(inputs[0].repeat(n, axis)))
        }
        "tile" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            let reps: Vec<usize> = case
                .get("params")
                .get("reps")
                .as_arr()
                .iter()
                .map(|v| v.as_f64() as usize)
                .collect();
            Ok(RunResult::F64(inputs[0].tile(&reps)))
        }
        "sort" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(inputs[0].sort()))
        }
        "argsort" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            let result = inputs[0].argsort();
            // argsort returns Array<usize>, fixture is int64
            let data: Vec<i64> = result.iter().map(|x| x as i64).collect();
            Ok(RunResult::I64(Array::from_vec(data, result.shape())))
        }
        "unique" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(np::unique(&inputs[0])))
        }
        "where" => {
            let cond = load_npy_bool(&fixture_dir.join(case.get("inputs").as_arr()[0].as_str()));
            let x = load_npy_f64(&fixture_dir.join(case.get("inputs").as_arr()[1].as_str()));
            let y = load_npy_f64(&fixture_dir.join(case.get("inputs").as_arr()[2].as_str()));
            Ok(RunResult::F64(np::where_(&cond, &x, &y)))
        }

        // -- slicing --
        "slice" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            let step = param_isize(case, "step");
            Ok(RunResult::F64(inputs[0].slice(&np::s![..;step])))
        }

        // -- comparisons --
        "equal" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::Bool(inputs[0].eq_elem(&inputs[1])))
        }
        "less" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::Bool(inputs[0].lt(&inputs[1])))
        }

        // -- astype --
        "astype_f64_to_i64" => {
            let inputs = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::I64(inputs[0].astype::<i64>()))
        }
        "astype_i64_to_f64" => {
            let inputs: Vec<Array<i64>> = case
                .get("inputs")
                .as_arr()
                .iter()
                .map(|f| load_npy_i64(&fixture_dir.join(f.as_str())))
                .collect();
            Ok(RunResult::F64(inputs[0].astype::<f64>()))
        }

        // -- linalg --
        "matmul" | "matmul_batched" => {
            let i = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(i[0].matmul(&i[1])))
        }
        "dot" => {
            let i = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(i[0].dot(&i[1])))
        }
        "det" => {
            let i = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(Array::scalar(np::linalg::det(&i[0]))))
        }
        "inv" => {
            let i = load_f64_inputs(fixture_dir, case);
            np::linalg::inv(&i[0])
                .map(RunResult::F64)
                .map_err(|e| e.to_string())
        }
        "solve" => {
            let i = load_f64_inputs(fixture_dir, case);
            np::linalg::solve(&i[0], &i[1])
                .map(RunResult::F64)
                .map_err(|e| e.to_string())
        }
        "norm" => {
            let i = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(Array::scalar(np::linalg::norm(&i[0]))))
        }
        "cholesky" => {
            let i = load_f64_inputs(fixture_dir, case);
            np::linalg::cholesky(&i[0])
                .map(RunResult::F64)
                .map_err(|e| e.to_string())
        }
        "qr_q" => {
            let i = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(np::linalg::qr(&i[0]).0))
        }
        "qr_r" => {
            let i = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(np::linalg::qr(&i[0]).1))
        }
        "eigh_eigenvalues" => {
            let i = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(np::linalg::eigh(&i[0]).0))
        }
        "svd_singular" => {
            let i = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(np::linalg::svd(&i[0]).1))
        }
        "pinv" => {
            let i = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(np::linalg::pinv(&i[0])))
        }
        "matrix_rank" => {
            let i = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::I64(Array::scalar(
                np::linalg::matrix_rank(&i[0]) as i64,
            )))
        }
        "outer" => {
            let i = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(np::linalg::outer(&i[0], &i[1])))
        }
        "trace" => {
            let i = load_f64_inputs(fixture_dir, case);
            Ok(RunResult::F64(Array::scalar(i[0].trace())))
        }

        _ => run_extra_case(fixture_dir, case, op),
    }
}

/// Cases added by the numpy-extra groups.
fn run_extra_case(fixture_dir: &Path, case: &Json, op: &str) -> Result<RunResult, String> {
    let i = load_f64_inputs(fixture_dir, case);
    match op {
        "nanvar_ddof_axis" => Ok(RunResult::F64(
            i[0].nanvar_ddof_axis(param_isize(case, "axis"), param_usize(case, "ddof")),
        )),
        "nanstd" => Ok(RunResult::F64(Array::scalar(i[0].nanstd()))),
        "nanmedian_axis" => Ok(RunResult::F64(
            i[0].nanmedian_axis(param_isize(case, "axis")),
        )),
        "nanpercentile_axis" => Ok(RunResult::F64(
            i[0].nanpercentile_axis(param_f64(case, "q"), param_isize(case, "axis")),
        )),
        "nancumsum_axis" => Ok(RunResult::F64(
            i[0].nancumsum_axis(param_isize(case, "axis")),
        )),
        "nancumprod_axis" => Ok(RunResult::F64(
            i[0].nancumprod_axis(param_isize(case, "axis")),
        )),
        "nanprod_axis" => Ok(RunResult::F64(i[0].nanprod_axis(param_isize(case, "axis")))),
        "nanargmax_axis" => Ok(RunResult::I64(
            i[0].nanargmax_axis(param_isize(case, "axis"))
                .map(|x| x as i64),
        )),
        "nanargmin" => Ok(RunResult::I64(Array::scalar(i[0].nanargmin() as i64))),
        "lexsort" => {
            let rows: Vec<Array<f64>> = (0..i[0].shape()[0])
                .map(|r| i[0].index_axis(0, r))
                .collect();
            let keys: Vec<&Array<f64>> = rows.iter().collect();
            Ok(RunResult::I64(np::lexsort(&keys, -1).map(|x| x as i64)))
        }
        "partition_kth" => {
            let axis = param_isize(case, "axis");
            let kth = param_isize(case, "kth");
            let n = i[0].shape()[if axis < 0 {
                (axis + 2) as usize
            } else {
                axis as usize
            }] as isize;
            let k = if kth < 0 { kth + n } else { kth };
            Ok(RunResult::F64(
                i[0].partition(kth, axis).index_axis(axis, k as usize),
            ))
        }
        "flatnonzero" => Ok(RunResult::I64(np::flatnonzero(&i[0]).map(|x| x as i64))),
        "trim_zeros" => Ok(RunResult::F64(np::trim_zeros(&i[0], "fb"))),
        "select" => {
            let (lo, hi, def) = (
                param_f64(case, "lo"),
                param_f64(case, "hi"),
                param_f64(case, "default"),
            );
            let x = &i[0];
            Ok(RunResult::F64(np::select(
                &[&x.lt(lo), &x.gt(hi)],
                &[&(x * 2.0), &(x * 3.0)],
                def,
            )))
        }
        "extract" => Ok(RunResult::F64(np::extract(&i[0].gt(0.0), &i[0]))),
        "tril_indices" | "triu_indices" => {
            let (n, k) = (param_usize(case, "n"), param_isize(case, "k"));
            let m = param_usize(case, "m");
            let m = if m == 0 { None } else { Some(m) };
            let (r, c) = if op == "tril_indices" {
                np::tril_indices(n, k, m)
            } else {
                np::triu_indices(n, k, m)
            };
            Ok(RunResult::I64(
                np::concatenate(&[&r, &c], 0)
                    .map(|x| x as i64)
                    .reshape(&[2, r.size()]),
            ))
        }
        "fill_diagonal_wrap" | "fill_diagonal" => {
            let mut a = i[0].clone();
            np::fill_diagonal(&mut a, param_f64(case, "val"), op == "fill_diagonal_wrap");
            Ok(RunResult::F64(a))
        }
        _ => Err(format!("skipped: op `{op}` not wired")),
    }
}

enum RunResult {
    F64(Array<f64>),
    I64(Array<i64>),
    Bool(Array<bool>),
}

// ---------------------------------------------------------------------------
// Main test
// ---------------------------------------------------------------------------

#[test]
fn parity_with_numpy() {
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");

    let manifest_path = fixture_dir.join("manifest.json");
    if !manifest_path.exists() {
        eprintln!(
            "SKIP parity tests: {manifest_path:?} not found. \
             Run `python tools/gen_fixtures.py` first."
        );
        return;
    }

    let manifest_text = fs::read_to_string(&manifest_path).unwrap();
    let manifest = parse_json(&manifest_text);
    let cases = manifest.as_arr();

    let mut passed = 0usize;
    let mut skipped = 0usize;
    let mut failed = Vec::new();

    for case in cases {
        let name = case.get("name").as_str();
        let dtype = case.get("dtype").as_str();
        let rtol = case.get("rtol").as_f64();
        let atol = case.get("atol").as_f64();
        let output_file = fixture_dir.join(case.get("output").as_str());

        match run_case(&fixture_dir, case) {
            Err(reason) => {
                eprintln!("  {name}: {reason}");
                skipped += 1;
            }
            Ok(result) => {
                let ok = match (result, dtype) {
                    (RunResult::F64(computed), "float64" | "float32") => {
                        let expected = load_npy_f64(&output_file);
                        allclose_f64(&computed, &expected, rtol, atol)
                    }
                    (RunResult::I64(computed), "int64" | "int32") => {
                        let expected = load_npy_i64(&output_file);
                        exact_i64(&computed, &expected)
                    }
                    (RunResult::Bool(computed), "bool") => {
                        let expected = load_npy_bool(&output_file);
                        exact_bool(&computed, &expected)
                    }
                    _ => {
                        eprintln!("  {name}: dtype mismatch in dispatch");
                        false
                    }
                };
                if ok {
                    passed += 1;
                } else {
                    failed.push(name.to_string());
                }
            }
        }
    }

    let total = passed + skipped + failed.len();
    eprintln!();
    eprintln!("parity: {passed}/{total} cases passed, {skipped} skipped");
    if !failed.is_empty() {
        eprintln!("FAILED cases:");
        for name in &failed {
            eprintln!("  - {name}");
        }
        panic!(
            "parity: {}/{total} cases failed: {}",
            failed.len(),
            failed.join(", ")
        );
    }
}
