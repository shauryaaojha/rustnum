//! numpy.ma parity cases (`ma_*` ops). Each case is run three times by the manifest (`part` =
//! filled / mask / data).

use super::{Json, Out, descr_of, load};
use rustnum as np;
use rustnum::ma::{self, MaskedArray};
use rustnum::{Array, Float, Num};
use std::path::Path;

trait Emit: Num {
    const FILL: Self;
    fn out(a: Array<Self>) -> Out;
}
impl Emit for f64 {
    const FILL: f64 = -999.0;
    fn out(a: Array<f64>) -> Out {
        Out::F64(a)
    }
}
impl Emit for i64 {
    const FILL: i64 = -999;
    fn out(a: Array<i64>) -> Out {
        Out::I64(a)
    }
}

fn has(case: &Json, key: &str) -> bool {
    case.get("params").as_obj().iter().any(|(k, _)| k == key)
}

fn part_of(case: &Json) -> String {
    if has(case, "part") {
        case.get("params").get("part").as_str().to_string()
    } else {
        "filled".to_string()
    }
}

fn axis(case: &Json) -> Option<isize> {
    has(case, "axis").then(|| super::param_isize(case, "axis"))
}

fn operand<T: Emit>(dir: &Path, case: &Json, k: usize) -> MaskedArray<T> {
    let files = case.get("inputs").as_arr();
    let data = load::<T>(&dir.join(files[2 * k].as_str()));
    let nomask = case
        .get("params")
        .get("nomask")
        .as_arr()
        .iter()
        .any(|v| v.as_f64() as usize == k);
    let mask = (!nomask).then(|| load::<bool>(&dir.join(files[2 * k + 1].as_str())));
    ma::masked_array(data, mask)
}

fn emit<T: Emit>(res: &MaskedArray<T>, part: &str) -> Out {
    match part {
        "filled" => T::out(res.filled(T::FILL)),
        "mask" => Out::Bool(ma::getmask(res)),
        _ => T::out(ma::getdata(res)),
    }
}

/// A 0-d masked array from an optional scalar (`None` = `np.ma.masked`).
fn scalar<T: Emit>(v: Option<T>) -> MaskedArray<T> {
    ma::masked_array(
        Array::scalar(v.unwrap_or_default()),
        Some(Array::scalar(v.is_none())),
    )
}

fn index_result<T: Emit>(a: Array<usize>) -> MaskedArray<i64> {
    let _ = std::marker::PhantomData::<T>;
    ma::masked_array(a.astype::<i64>(), None)
}

fn run_num<T: Emit>(dir: &Path, case: &Json, name: &str) -> Option<Result<Out, String>> {
    let part = part_of(case);
    let x = operand::<T>(dir, case, 0);
    let value = || T::from_f64(super::param_f64(case, "value"));
    let ax = axis(case);
    let res: MaskedArray<T> = match name {
        "add" | "sub" | "mul" | "div" => {
            let y = operand::<T>(dir, case, 1);
            match name {
                "add" => &x + &y,
                "sub" => &x - &y,
                "mul" => &x * &y,
                _ => &x / &y,
            }
        }
        "add_scalar" => &x + value(),
        "mul_scalar" => &x * value(),
        "abs" => x.abs(),
        "square" => x.square(),
        "sum" => match ax {
            None => scalar(x.sum()),
            Some(a) => x.sum_axis(a),
        },
        "prod" => match ax {
            None => scalar(x.prod()),
            Some(a) => x.prod_axis(a),
        },
        "min" => match ax {
            None => scalar(x.min()),
            Some(a) => x.min_axis(a),
        },
        "max" => match ax {
            None => scalar(x.max()),
            Some(a) => x.max_axis(a),
        },
        "argmin" => {
            let r = match ax {
                None => ma::masked_array(Array::scalar(x.argmin() as i64), None),
                Some(a) => index_result::<T>(x.argmin_axis(a)),
            };
            return Some(Ok(emit(&r, &part)));
        }
        "argmax" => {
            let r = match ax {
                None => ma::masked_array(Array::scalar(x.argmax() as i64), None),
                Some(a) => index_result::<T>(x.argmax_axis(a)),
            };
            return Some(Ok(emit(&r, &part)));
        }
        "count" => {
            let r = match ax {
                None => ma::masked_array(Array::scalar(x.count() as i64), None),
                Some(a) => index_result::<T>(x.count_axis(a)),
            };
            return Some(Ok(emit(&r, &part)));
        }
        "compressed" => {
            let c = ma::compressed(&x);
            ma::masked_array(c, None)
        }
        "slice" => x.slice(&np::s![1..3, ..;2]),
        "reshape" => {
            let shape: Vec<usize> = case
                .get("params")
                .get("shape")
                .as_arr()
                .iter()
                .map(|v| v.as_f64() as usize)
                .collect();
            x.reshape(&shape)
        }
        "transpose" => x.t(),
        "ravel" => x.ravel(),
        "masked_equal" => ma::masked_equal(x.data().clone(), value()),
        "masked_greater" => ma::masked_greater(x.data(), value()),
        "masked_greater_equal" => ma::masked_greater_equal(x.data(), value()),
        "masked_less" => ma::masked_less(x.data(), value()),
        "masked_less_equal" => ma::masked_less_equal(x.data(), value()),
        "masked_not_equal" => ma::masked_not_equal(x.data(), value()),
        "masked_inside" => ma::masked_inside(
            x.data(),
            T::from_f64(super::param_f64(case, "v1")),
            T::from_f64(super::param_f64(case, "v2")),
        ),
        "masked_outside" => ma::masked_outside(
            x.data(),
            T::from_f64(super::param_f64(case, "v1")),
            T::from_f64(super::param_f64(case, "v2")),
        ),
        "chained" => ma::masked_greater(
            ma::masked_less(x.data(), T::from_f64(1.5)),
            T::from_f64(2.5),
        ),
        _ => return None,
    };
    let _ = &part;
    Some(Ok(emit(&res, &part)))
}

fn run_float(dir: &Path, case: &Json, name: &str) -> Option<Result<Out, String>> {
    let part = part_of(case);
    let x = operand::<f64>(dir, case, 0);
    let ax = axis(case);
    let res: MaskedArray<f64> = match name {
        "rdiv_scalar" => super::param_f64(case, "value") / &x,
        "neg" => -&x,
        "exp" => x.exp(),
        "sin" => x.sin(),
        "cos" => x.cos(),
        "tanh" => x.tanh(),
        "sqrt" => x.sqrt(),
        "log" => x.log(),
        "log2" => x.log2(),
        "log10" => x.log10(),
        "arcsin" => x.arcsin(),
        "arccos" => x.arccos(),
        "mean" => match ax {
            None => scalar(x.mean()),
            Some(a) => x.mean_axis(a),
        },
        "std" => match ax {
            None => scalar(x.std()),
            Some(a) => x.std_axis(a),
        },
        "var" => match ax {
            None => scalar(x.var()),
            Some(a) => x.var_axis(a),
        },
        "var_ddof" => {
            let ddof = super::param_usize(case, "ddof");
            match ax {
                None => scalar(x.var_ddof(ddof)),
                Some(a) => x.var_ddof_axis(a, ddof),
            }
        }
        "masked_invalid" => ma::masked_invalid(x.data()),
        "masked_values" => ma::masked_values(
            x.data(),
            super::param_f64(case, "value"),
            super::param_f64(case, "rtol"),
            super::param_f64(case, "atol"),
        ),
        "masked_where" => {
            let files = case.get("inputs").as_arr();
            let cond = load::<bool>(&dir.join(files[2].as_str()));
            ma::masked_where(&cond, x.data())
        }
        "str" => return Some(Ok(Out::Str(x.to_string()))),
        _ => return None,
    };
    let _ = <f64 as Float>::PI;
    Some(Ok(emit(&res, &part)))
}

/// Runs a masked-array op; `None` for other ops.
pub fn run(dir: &Path, case: &Json) -> Option<Result<Out, String>> {
    let op = case.get("op").as_str();
    let name = op.strip_prefix("ma_")?;
    let first = dir.join(case.get("inputs").as_arr()[0].as_str());
    let integer = descr_of(&first) == "<i8";
    if name == "str" {
        let text = |k: usize| dir.join(case.get("inputs").as_arr()[k].as_str());
        let nomask = has(case, "nomask") && !case.get("params").get("nomask").as_arr().is_empty();
        let mask = (!nomask).then(|| load::<bool>(&text(1)));
        let s = match descr_of(&first).as_str() {
            "<i8" => ma::masked_array(load::<i64>(&first), mask).to_string(),
            "|b1" => ma::masked_array(load::<bool>(&first), mask).to_string(),
            _ => ma::masked_array(load::<f64>(&first), mask).to_string(),
        };
        return Some(Ok(Out::Str(s)));
    }
    let r = if integer {
        run_num::<i64>(dir, case, name)
    } else {
        run_num::<f64>(dir, case, name).or_else(|| run_float(dir, case, name))
    };
    Some(r.unwrap_or_else(|| Err(format!("skipped: op `{op}` not wired"))))
}
