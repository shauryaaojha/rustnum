//! Structured dtype (`rec_*` ops) parity cases: field access, sorting by field order, display and
//! the exact bytes of the `.npy` file NumPy writes.

use super::{Json, Out, load};
use rustnum as np;
use rustnum::Array;
use std::path::Path;

np::record! {
    struct P { id: i32, w: f64, ok: bool }
}

np::record! {
    struct Q { p: f32, q: u8 }
}

np::record! {
    struct N { a: i32, n: Q, c: np::Complex<f32> }
}

fn file_bytes<T: np::Element>(a: &Array<T>) -> Out {
    let bytes = np::io::to_npy_bytes(a);
    // compare the data after the header, whose spare padding differs from NumPy's
    let header = 10 + usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    let bytes = bytes[header..].to_vec();
    Out::I64(Array::from(
        bytes.into_iter().map(i64::from).collect::<Vec<_>>(),
    ))
}

/// Runs a structured-dtype op; `None` for ops of other dtypes.
pub fn run(dir: &Path, case: &Json) -> Option<Result<Out, String>> {
    let op = case.get("op").as_str();
    if !op.starts_with("rec_") {
        return None;
    }
    let input = dir.join(case.get("inputs").as_arr()[0].as_str());
    if op.starts_with("rec_nested_") {
        let b = load::<N>(&input);
        return Some(Ok(match op {
            "rec_nested_bytes" => file_bytes(&b),
            "rec_nested_str" => Out::Str(b.to_string()),
            _ => return Some(Err(format!("skipped: op `{op}` not wired"))),
        }));
    }
    let a = load::<P>(&input);
    let order: Vec<String> = case
        .get("params")
        .as_obj()
        .iter()
        .find(|(k, _)| k == "order")
        .map(|(_, v)| v.as_arr().iter().map(|j| j.as_str().to_string()).collect())
        .unwrap_or_default();
    let order: Vec<&str> = order.iter().map(String::as_str).collect();
    Some(Ok(match op {
        "rec_field_id" => Out::I64(a.field::<i32>("id").astype::<i64>()),
        "rec_field_w" => Out::F64(a.field::<f64>("w")),
        "rec_field_ok" => Out::Bool(a.field::<bool>("ok")),
        "rec_file_bytes" => file_bytes(&a),
        "rec_str" => Out::Str(a.to_string()),
        "rec_repr" => Out::Str(format!("{a:?}")),
        "rec_argsort" => Out::I64(a.argsort_order(&order).astype::<i64>()),
        "rec_sort_id" => Out::I64(a.sort_order(&order).field::<i32>("id").astype::<i64>()),
        _ => return Some(Err(format!("skipped: op `{op}` not wired"))),
    }))
}
