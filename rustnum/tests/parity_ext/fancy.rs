//! Advanced-indexing parity cases (`fx_*` ops).

use super::{Json, Out, load};
use rustnum as np;
use rustnum::{Array, SliceArg};
use std::path::Path;

fn file(dir: &Path, case: &Json, k: usize) -> std::path::PathBuf {
    dir.join(case.get("inputs").as_arr()[k].as_str())
}

fn opt(v: &Json) -> Option<isize> {
    match v {
        Json::Num(n) => Some(*n as isize),
        _ => None,
    }
}

/// Builds the index tuple described by the JSON `index` list.
fn index_tuple(dir: &Path, case: &Json) -> Vec<SliceArg> {
    case.get("params")
        .get("index")
        .as_arr()
        .iter()
        .map(|e| match e.get("t").as_str() {
            "arr" => SliceArg::from(&load::<i64>(&file(dir, case, e.get("i").as_f64() as usize))),
            "mask" => SliceArg::from(&load::<bool>(&file(
                dir,
                case,
                e.get("i").as_f64() as usize,
            ))),
            "int" => SliceArg::Index(e.get("v").as_f64() as isize),
            "new" => SliceArg::NewAxis,
            "slice" => SliceArg::Range {
                start: opt(e.get("start")),
                end: opt(e.get("stop")),
                step: e.get("step").as_f64() as isize,
            },
            other => panic!("unknown index entry {other}"),
        })
        .collect()
}

/// Runs an advanced-indexing op; `None` for other ops.
pub fn run(dir: &Path, case: &Json) -> Option<Result<Out, String>> {
    let op = case.get("op").as_str();
    if !op.starts_with("fx_") {
        return None;
    }
    let n_inputs = case.get("inputs").as_arr().len();
    let axis = |case: &Json| super::param_isize(case, "axis");
    Some(Ok(match op {
        "fx_get" => {
            let a = load::<f64>(&file(dir, case, 0));
            Out::F64(a.slice(&index_tuple(dir, case)))
        }
        "fx_set" => {
            let mut a = load::<f64>(&file(dir, case, 0));
            let values = load::<f64>(&file(dir, case, n_inputs - 1));
            a.slice_assign(&index_tuple(dir, case), values);
            Out::F64(a)
        }
        "fx_take_along" => {
            let a = load::<f64>(&file(dir, case, 0));
            let i = load::<i64>(&file(dir, case, 1));
            Out::F64(np::take_along_axis(&a, &i, axis(case)))
        }
        "fx_put_along" => {
            let mut a = load::<f64>(&file(dir, case, 0));
            let i = load::<i64>(&file(dir, case, 1));
            let v = load::<f64>(&file(dir, case, 2));
            np::put_along_axis(&mut a, &i, v, axis(case));
            Out::F64(a)
        }
        "fx_put" => {
            let mut a = load::<f64>(&file(dir, case, 0));
            np::put(
                &mut a,
                load::<i64>(&file(dir, case, 1)),
                load::<f64>(&file(dir, case, 2)),
            );
            Out::F64(a)
        }
        "fx_choose" => {
            let sel = load::<i64>(&file(dir, case, 0));
            let choices: Vec<Array<f64>> = (1..n_inputs)
                .map(|k| load::<f64>(&file(dir, case, k)))
                .collect();
            let refs: Vec<&Array<f64>> = choices.iter().collect();
            Out::F64(np::choose(&sel, &refs))
        }
        "fx_ravel" => {
            let parts: Vec<Array<i64>> = (0..n_inputs)
                .map(|k| load::<i64>(&file(dir, case, k)))
                .collect();
            let refs: Vec<&Array<i64>> = parts.iter().collect();
            let dims: Vec<usize> = case
                .get("params")
                .get("dims")
                .as_arr()
                .iter()
                .map(|v| v.as_f64() as usize)
                .collect();
            Out::I64(np::ravel_multi_index(&refs, &dims))
        }
        "fx_unravel" => {
            let flat = load::<i64>(&file(dir, case, 0));
            let dims: Vec<usize> = case
                .get("params")
                .get("dims")
                .as_arr()
                .iter()
                .map(|v| v.as_f64() as usize)
                .collect();
            let cols = np::unravel_index(&flat, &dims);
            let refs: Vec<&Array<i64>> = cols.iter().collect();
            Out::I64(np::stack(&refs, 0))
        }
        "fx_ix" => {
            let a = load::<f64>(&file(dir, case, 0));
            let seqs: Vec<Vec<i64>> = (1..n_inputs)
                .map(|k| load::<i64>(&file(dir, case, k)).to_vec())
                .collect();
            let refs: Vec<&[i64]> = seqs.iter().map(Vec::as_slice).collect();
            let mesh = np::ix_(&refs);
            let tuple: Vec<SliceArg> = mesh.iter().map(SliceArg::from).collect();
            Out::F64(a.slice(&tuple))
        }
        _ => return Some(Err(format!("skipped: op `{op}` not wired"))),
    }))
}
