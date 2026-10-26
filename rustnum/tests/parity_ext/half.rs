//! float16 (`h_*` ops) and bfloat16 (`bh_*` ops) parity cases.

use super::{Json, Out, load};
use rustnum as np;
use rustnum::{Array, bf16, f16};
use std::path::Path;

fn inputs<T: np::Element>(dir: &Path, case: &Json) -> Vec<Array<T>> {
    case.get("inputs")
        .as_arr()
        .iter()
        .map(|f| load::<T>(&dir.join(f.as_str())))
        .collect()
}

fn bits16(a: &Array<f16>) -> Array<i64> {
    a.map(|x| {
        if x.is_nan() {
            0x7e00
        } else {
            i64::from(x.to_bits())
        }
    })
}

fn f32_bits(a: &Array<f32>) -> Array<i64> {
    a.map(|x| {
        if x.is_nan() {
            0x7fc0_0000
        } else {
            i64::from(x.to_bits())
        }
    })
}

fn bf16_bits(a: &Array<bf16>) -> Array<i64> {
    a.map(|x| {
        if x.is_nan() {
            0x7fc0
        } else {
            i64::from(x.to_bits())
        }
    })
}

fn to_bf16(a: &Array<i64>) -> Array<bf16> {
    a.map(|b| bf16::from_bits(b as u16))
}

/// Runs a float16 / bfloat16 op; `None` for ops of other dtypes.
pub fn run(dir: &Path, case: &Json) -> Option<Result<Out, String>> {
    let op = case.get("op").as_str();
    if let Some(name) = op.strip_prefix("bh_") {
        return Some(run_bf16(dir, case, name));
    }
    let name = op.strip_prefix("h_")?;
    Some(run_f16(dir, case, name))
}

fn run_f16(dir: &Path, case: &Json, name: &str) -> Result<Out, String> {
    let axis = |case: &Json| super::param_isize(case, "axis");
    Ok(match name {
        "to_f32_bits" => {
            let a = load::<f16>(&dir.join(case.get("inputs").as_arr()[0].as_str()));
            let bulk = a.astype::<f32>();
            let scalar = a.map(|x| x.to_f32());
            if f32_bits(&bulk) != f32_bits(&scalar) {
                return Err("bulk f16->f32 differs from the scalar path".into());
            }
            Out::I64(f32_bits(&bulk))
        }
        "roundtrip_bits" => {
            let a = load::<f16>(&dir.join(case.get("inputs").as_arr()[0].as_str()));
            Out::I64(bits16(&a.astype::<f32>().astype::<f16>()))
        }
        "to_f64" => {
            let a = load::<f16>(&dir.join(case.get("inputs").as_arr()[0].as_str()));
            Out::F64(a.astype::<f64>())
        }
        "f32_to_f16_bits" => {
            let a = load::<f32>(&dir.join(case.get("inputs").as_arr()[0].as_str()));
            let bulk = a.astype::<f16>();
            let scalar = a.map(f16::from_f32);
            let mut slice_scalar = vec![f16::from_bits(0); a.size()];
            f16::slice_from_f32_scalar(&a.to_vec(), &mut slice_scalar);
            if bits16(&bulk) != bits16(&scalar)
                || bits16(&scalar) != bits16(&Array::from(slice_scalar))
            {
                return Err("bulk f32->f16 differs from the scalar path".into());
            }
            Out::I64(bits16(&bulk))
        }
        "f64_to_f16_bits" => {
            let a = load::<f64>(&dir.join(case.get("inputs").as_arr()[0].as_str()));
            Out::I64(bits16(&a.astype::<f16>()))
        }
        "from_f64" => {
            let a = load::<f64>(&dir.join(case.get("inputs").as_arr()[0].as_str()));
            Out::F16(a.astype::<f16>())
        }
        "to_i64" => {
            let a = load::<f16>(&dir.join(case.get("inputs").as_arr()[0].as_str()));
            Out::I64(a.astype::<i64>())
        }
        "isnan" => {
            let a = load::<f16>(&dir.join(case.get("inputs").as_arr()[0].as_str()));
            Out::Bool(a.is_nan())
        }
        "str" => {
            Out::Str(load::<f16>(&dir.join(case.get("inputs").as_arr()[0].as_str())).to_string())
        }
        _ => {
            let v = inputs::<f16>(dir, case);
            let a = &v[0];
            match name {
                "add" => Out::F16(a + &v[1]),
                "sub" => Out::F16(a - &v[1]),
                "mul" => Out::F16(a * &v[1]),
                "div" => Out::F16(a / &v[1]),
                "exp" => Out::F16(a.exp()),
                "log" => Out::F16(a.log()),
                "sqrt" => Out::F16(a.sqrt()),
                "sin" => Out::F16(a.sin()),
                "cos" => Out::F16(a.cos()),
                "tanh" => Out::F16(a.tanh()),
                "floor" => Out::F16(a.floor()),
                "round" => Out::F16(a.round()),
                "abs" => Out::F16(a.abs()),
                "neg" => Out::F16(-a),
                "sum" => Out::F16(Array::scalar(a.sum())),
                "mean" => Out::F16(Array::scalar(a.mean())),
                "std" => Out::F16(Array::scalar(a.std())),
                "var" => Out::F16(Array::scalar(a.var())),
                "max" => Out::F16(Array::scalar(a.max())),
                "sum_axis" => Out::F16(a.sum_axis(axis(case))),
                "cumsum" => Out::F16(a.cumsum()),
                "sort" => Out::F16(a.sort()),
                _ => return Err(format!("skipped: op `h_{name}` not wired")),
            }
        }
    })
}

fn run_bf16(dir: &Path, case: &Json, name: &str) -> Result<Out, String> {
    let first = dir.join(case.get("inputs").as_arr()[0].as_str());
    Ok(match name {
        "to_f32_bits" => {
            let a = to_bf16(&load::<i64>(&first));
            Out::I64(f32_bits(&a.astype::<f32>()))
        }
        "f32_to_bf16_bits" => {
            let a = load::<f32>(&first);
            let bulk = a.astype::<bf16>();
            let scalar = a.map(bf16::from_f32);
            if bf16_bits(&bulk) != bf16_bits(&scalar) {
                return Err("bulk f32->bf16 differs from the scalar path".into());
            }
            Out::I64(bf16_bits(&bulk))
        }
        "f64_to_bf16_bits" => Out::I64(bf16_bits(&load::<f64>(&first).astype::<bf16>())),
        "str" => Out::Str(to_bf16(&load::<i64>(&first)).to_string()),
        "add" | "sub" | "mul" | "div" | "exp" => {
            let a = to_bf16(&load::<i64>(&first));
            let b = case
                .get("inputs")
                .as_arr()
                .get(1)
                .map(|f| to_bf16(&load::<i64>(&dir.join(f.as_str()))));
            let r = match (name, &b) {
                ("add", Some(b)) => &a + b,
                ("sub", Some(b)) => &a - b,
                ("mul", Some(b)) => &a * b,
                ("div", Some(b)) => &a / b,
                ("exp", None) => a.exp(),
                _ => return Err(format!("bad operands for bh_{name}")),
            };
            Out::I64(bf16_bits(&r))
        }
        _ => return Err(format!("skipped: op `bh_{name}` not wired")),
    })
}
