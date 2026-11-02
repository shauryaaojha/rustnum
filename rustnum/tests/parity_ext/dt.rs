//! datetime64 (`dt_*` ops) and timedelta64 (`td_*` ops) parity cases. The unit is `params.unit`.
//! Results are compared as raw `i64` counts (`NaT` is `i64::MIN`), booleans, floats or strings.

use super::{Json, Out, load};
use rustnum as np;
use rustnum::datetime::{D, M, TimeUnit, W, Y, h, m, ms, ns, s, us};
use rustnum::{Array, DateTime64, TimeDelta64};
use std::path::Path;

macro_rules! by_unit {
    (@arms $unit:expr, $call:ident, ($($pre:ty),*), ($($arg:expr),*)) => {
        match $unit {
            "Y" => $call::<$($pre,)* Y>($($arg),*),
            "M" => $call::<$($pre,)* M>($($arg),*),
            "W" => $call::<$($pre,)* W>($($arg),*),
            "D" => $call::<$($pre,)* D>($($arg),*),
            "h" => $call::<$($pre,)* h>($($arg),*),
            "m" => $call::<$($pre,)* m>($($arg),*),
            "s" => $call::<$($pre,)* s>($($arg),*),
            "ms" => $call::<$($pre,)* ms>($($arg),*),
            "us" => $call::<$($pre,)* us>($($arg),*),
            "ns" => $call::<$($pre,)* ns>($($arg),*),
            other => panic!("unknown unit {other}"),
        }
    };
    ($unit:expr, $call:ident($($arg:expr),*)) => {
        by_unit!(@arms $unit, $call, (), ($($arg),*))
    };
    ($unit:expr, $call:ident<$pre:ty>($($arg:expr),*)) => {
        by_unit!(@arms $unit, $call, ($pre), ($($arg),*))
    };
}

fn param_str<'a>(case: &'a Json, key: &str) -> &'a str {
    case.get("params").get(key).as_str()
}

fn param_i64(case: &Json, key: &str) -> i64 {
    case.get("params").get(key).as_f64() as i64
}

fn raw_dt<U: TimeUnit>(a: &Array<DateTime64<U>>) -> Out {
    Out::I64(a.map(|x| x.value()))
}

fn raw_td<U: TimeUnit>(a: &Array<TimeDelta64<U>>) -> Out {
    Out::I64(a.map(|x| x.value()))
}

fn td_scalar<U: TimeUnit>(x: TimeDelta64<U>) -> Out {
    Out::I64(Array::scalar(x.value()))
}

fn file(case: &Json, dir: &Path, i: usize) -> std::path::PathBuf {
    dir.join(case.get("inputs").as_arr()[i].as_str())
}

/// Runs a datetime / timedelta op; `None` for ops of other dtypes.
pub fn run(dir: &Path, case: &Json) -> Option<Result<Out, String>> {
    let op = case.get("op").as_str();
    if !(op.starts_with("dt_") || op.starts_with("td_")) {
        return None;
    }
    if op == "dt_astype" || op == "td_astype" {
        let from = param_str(case, "from");
        let to = param_str(case, "unit");
        return Some(astype_dispatch(dir, case, from, to));
    }
    let unit = param_str(case, "unit");
    Some(by_unit!(unit, run_unit(dir, case, op)))
}

fn astype_dispatch(dir: &Path, case: &Json, from: &str, to: &str) -> Result<Out, String> {
    macro_rules! inner {
        ($f:ty) => {
            by_unit!(to, astype_unit<$f>(dir, case))
        };
    }
    Ok(match from {
        "Y" => inner!(Y),
        "M" => inner!(M),
        "W" => inner!(W),
        "D" => inner!(D),
        "h" => inner!(h),
        "m" => inner!(m),
        "s" => inner!(s),
        "ms" => inner!(ms),
        "us" => inner!(us),
        "ns" => inner!(ns),
        other => panic!("unknown unit {other}"),
    })
}

fn astype_unit<F: TimeUnit, T: TimeUnit>(dir: &Path, case: &Json) -> Out {
    let op = case.get("op").as_str();
    if op == "dt_astype" {
        let a = load::<DateTime64<F>>(&file(case, dir, 0));
        raw_dt(&a.astype::<DateTime64<T>>())
    } else {
        let a = load::<TimeDelta64<F>>(&file(case, dir, 0));
        raw_td(&a.astype::<TimeDelta64<T>>())
    }
}

fn texts<U: TimeUnit>(a: &Array<DateTime64<U>>) -> String {
    np::datetime_as_string(a).join("|")
}

fn run_unit<U: TimeUnit>(dir: &Path, case: &Json, op: &str) -> Result<Out, String> {
    let f0 = || file(case, dir, 0);
    let dts = |i: usize| load::<DateTime64<U>>(&file(case, dir, i));
    let tds = |i: usize| load::<TimeDelta64<U>>(&file(case, dir, i));
    Ok(match op {
        "dt_load" => raw_dt(&dts(0)),
        "td_load" => raw_td(&tds(0)),
        "dt_str" => Out::Str(dts(0).to_string()),
        "dt_repr" => Out::Str(format!("{:?}", dts(0))),
        "td_str" => Out::Str(tds(0).to_string()),
        "td_repr" => Out::Str(format!("{:?}", tds(0))),
        "dt_as_string" => Out::Str(texts(&dts(0))),
        "dt_cast_month_str" => Out::Str(texts(&dts(0).astype::<DateTime64<M>>())),
        "dt_isnat" => Out::Bool(np::isnat(&dts(0))),
        "dt_parse" => {
            let strings: Vec<DateTime64<U>> = case
                .get("params")
                .get("strings")
                .as_arr()
                .iter()
                .map(|j| {
                    DateTime64::<U>::parse(j.as_str())
                        .unwrap_or_else(|e| panic!("parse {:?}: {e}", j.as_str()))
                })
                .collect();
            raw_dt(&Array::from(strings))
        }
        "dt_sub" => raw_td(&(&dts(0) - &dts(1))),
        "dt_add_td" => raw_dt(&(&dts(0) + &tds(1))),
        "td_add_dt" => raw_dt(&(&tds(0) + &dts(1))),
        "dt_sub_td" => raw_dt(&(&dts(0) - &tds(1))),
        "td_add" => raw_td(&(&tds(0) + &tds(1))),
        "td_sub" => raw_td(&(&tds(0) - &tds(1))),
        "td_neg" => raw_td(&-&tds(0)),
        "td_mul_int" => raw_td(&(&tds(0) * param_i64(case, "k"))),
        "td_div_int" => raw_td(&(&tds(0) / param_i64(case, "k"))),
        "td_div_td" => Out::F64(&tds(0) / &tds(1)),
        "td_floordiv" => Out::I64(tds(0).floor_div(tds(1))),
        "td_mod" => raw_td(&(&tds(0) % &tds(1))),
        "dt_eq" => Out::Bool(dts(0).eq_elem(dts(1))),
        "dt_ne" => Out::Bool(dts(0).ne_elem(dts(1))),
        "dt_lt" => Out::Bool(dts(0).lt(dts(1))),
        "dt_le" => Out::Bool(dts(0).le(dts(1))),
        "dt_gt" => Out::Bool(dts(0).gt(dts(1))),
        "dt_ge" => Out::Bool(dts(0).ge(dts(1))),
        "td_eq" => Out::Bool(tds(0).eq_elem(tds(1))),
        "td_ne" => Out::Bool(tds(0).ne_elem(tds(1))),
        "td_lt" => Out::Bool(tds(0).lt(tds(1))),
        "td_le" => Out::Bool(tds(0).le(tds(1))),
        "td_gt" => Out::Bool(tds(0).gt(tds(1))),
        "td_ge" => Out::Bool(tds(0).ge(tds(1))),
        "td_sum" => td_scalar(tds(0).sum()),
        "td_mean" => td_scalar(tds(0).mean()),
        "td_cumsum" => raw_td(&tds(0).cumsum()),
        "dt_sort" => raw_dt(&dts(0).sort()),
        "td_sort" => raw_td(&tds(0).sort()),
        "dt_argsort" => Out::I64(dts(0).argsort().astype::<i64>()),
        "td_argsort" => Out::I64(tds(0).argsort().astype::<i64>()),
        "dt_from_i64" => {
            let a = load::<i64>(&f0());
            raw_dt(&a.astype::<DateTime64<U>>())
        }
        "td_from_i64" => {
            let a = load::<i64>(&f0());
            raw_td(&a.astype::<TimeDelta64<U>>())
        }
        "td_from_f64" => {
            let a = load::<f64>(&f0());
            raw_td(&a.astype::<TimeDelta64<U>>())
        }
        "dt_from_f64" => {
            let a = load::<f64>(&f0());
            raw_dt(&a.astype::<DateTime64<U>>())
        }
        "dt_to_i64" => Out::I64(dts(0).astype::<i64>()),
        "td_to_f64" => Out::F64(tds(0).astype::<f64>()),
        "dt_arange" => {
            let p = |k: &str| {
                DateTime64::<U>::parse(param_str(case, k)).unwrap_or_else(|e| panic!("{k}: {e}"))
            };
            let step = TimeDelta64::<U>::new(param_i64(case, "step"));
            raw_dt(&Array::<DateTime64<U>>::arange(p("start"), p("stop"), step))
        }
        "td_arange" => {
            let v = |k: &str| TimeDelta64::<U>::new(param_i64(case, k));
            raw_td(&Array::<TimeDelta64<U>>::arange(
                v("start"),
                v("stop"),
                v("step"),
            ))
        }
        _ => return Err(format!("skipped: op `{op}` not wired")),
    })
}
