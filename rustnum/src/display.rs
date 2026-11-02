use crate::{Array, Element};
use std::{
    any::Any,
    fmt::{self, Debug, Display, Formatter},
};

fn raw<T: Element>(x: T) -> String {
    if T::DTYPE == "bool" {
        if x.to_f64() != 0.0 {
            "True".into()
        } else {
            "False".into()
        }
    } else {
        let text = format!("{x:?}").to_lowercase();
        if crate::element::is_float_dtype(T::DTYPE)
            && let Some((mantissa, exponent)) = text.split_once('e')
        {
            let exp = exponent.parse::<i32>().unwrap();
            format!(
                "{mantissa}e{}{:02}",
                if exp < 0 { "-" } else { "+" },
                exp.unsigned_abs()
            )
        } else {
            text
        }
    }
}
fn shortest<T: Element>(x: T) -> String {
    if let Some(x) = (&x as &dyn Any).downcast_ref::<f32>() {
        x.to_string().to_lowercase()
    } else if let Some(x) = (&x as &dyn Any).downcast_ref::<crate::f16>() {
        x.to_string().to_lowercase()
    } else if let Some(x) = (&x as &dyn Any).downcast_ref::<crate::bf16>() {
        x.to_string().to_lowercase()
    } else {
        x.to_f64().to_string().to_lowercase()
    }
}
fn scientific<T: Element>(x: T) -> String {
    if let Some(x) = (&x as &dyn Any).downcast_ref::<f32>() {
        format!("{x:e}")
    } else if let Some(x) = (&x as &dyn Any).downcast_ref::<crate::f16>() {
        format!("{x:e}")
    } else if let Some(x) = (&x as &dyn Any).downcast_ref::<crate::bf16>() {
        format!("{x:e}")
    } else {
        format!("{:e}", x.to_f64())
    }
}
fn complex_values<T: Element>(values: &[T]) -> Vec<String> {
    // NumPy formats the real and imaginary parts as two independent float columns; the
    // imaginary column always carries a sign and the `j` goes right after the digits.
    fn columns<F: Element>(re: Vec<F>, im: Vec<F>) -> Vec<String> {
        let re_text = format_values(&re);
        let neg: Vec<bool> = im
            .iter()
            .map(|x| x.to_f64().is_sign_negative() && !x.to_f64().is_nan())
            .collect();
        let im_abs: Vec<F> = im.iter().map(|&x| F::from_f64(x.to_f64().abs())).collect();
        let im_text = format_values(&im_abs);
        re_text
            .into_iter()
            .zip(im_text)
            .zip(neg)
            .map(|((r, i), n)| {
                let trimmed = i.trim_end();
                let pad = &i[trimmed.len()..];
                let lead = trimmed.trim_start();
                let left = &trimmed[..trimmed.len() - lead.len()];
                format!("{r}{left}{}{lead}j{pad}", if n { '-' } else { '+' })
            })
            .collect()
    }
    let parts: Vec<(f64, f64)> = values.iter().map(|v| v.to_complex_parts()).collect();
    if T::DTYPE == "complex64" {
        columns(
            parts.iter().map(|p| p.0 as f32).collect(),
            parts.iter().map(|p| p.1 as f32).collect(),
        )
    } else {
        columns(
            parts.iter().map(|p| p.0).collect(),
            parts.iter().map(|p| p.1).collect(),
        )
    }
}
fn format_values<T: Element>(values: &[T]) -> Vec<String> {
    if T::DTYPE.starts_with("complex") {
        return complex_values(values);
    }
    if !crate::element::is_float_dtype(T::DTYPE) {
        let text: Vec<_> = values.iter().copied().map(raw).collect();
        let width = if T::DTYPE == "bool" {
            5
        } else {
            text.iter().map(String::len).max().unwrap_or(0)
        };
        return text.into_iter().map(|s| format!("{s:>width$}")).collect();
    }
    let finite: Vec<f64> = values
        .iter()
        .map(|x| x.to_f64())
        .filter(|x| x.is_finite() && *x != 0.0)
        .map(f64::abs)
        .collect();
    let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let max = finite.iter().copied().fold(0., f64::max);
    let sci = max >= 1e8 || (min < 1e-4) || (max > 0. && max / min > 1000.);
    let negative = values.iter().any(|x| x.to_f64().is_sign_negative());
    if sci {
        let precision = values
            .iter()
            .filter(|x| x.to_f64().is_finite())
            .map(|&x| {
                let s = scientific(x);
                let mantissa = s.split('e').next().unwrap();
                mantissa.split('.').nth(1).map_or(0, str::len).min(8)
            })
            .max()
            .unwrap_or(0);
        let exponent_width = values
            .iter()
            .filter(|x| x.to_f64().is_finite())
            .map(|&x| {
                let s = format!("{:.*e}", precision, x.to_f64());
                s.split('e')
                    .nth(1)
                    .unwrap()
                    .parse::<i32>()
                    .unwrap()
                    .unsigned_abs()
                    .to_string()
                    .len()
            })
            .max()
            .unwrap_or(2)
            .max(2);
        let text: Vec<_> = values
            .iter()
            .map(|&x| {
                let x = x.to_f64();
                if !x.is_finite() {
                    return x.to_string().to_lowercase();
                }
                let s = format!("{x:.precision$e}");
                let (mantissa, exponent) = s.split_once('e').unwrap();
                let exp = exponent.parse::<i32>().unwrap();
                let mantissa = if precision == 0 {
                    format!("{mantissa}.")
                } else {
                    mantissa.into()
                };
                format!(
                    "{}{}e{}{:0exponent_width$}",
                    if negative && !x.is_sign_negative() {
                        " "
                    } else {
                        ""
                    },
                    mantissa,
                    if exp < 0 { "-" } else { "+" },
                    exp.unsigned_abs()
                )
            })
            .collect();
        let width = text.iter().map(String::len).max().unwrap_or(0);
        return text.into_iter().map(|s| format!("{s:>width$}")).collect();
    }
    let text: Vec<_> = values
        .iter()
        .map(|&x| {
            if !x.to_f64().is_finite() {
                return x.to_f64().to_string().to_lowercase();
            }
            let short = shortest(x);
            let precision = short.split('.').nth(1).map_or(0, str::len).min(8);
            let mut s = format!("{:.*}", precision, x.to_f64());
            if s.contains('.') {
                while s.ends_with('0') {
                    s.pop();
                }
            } else {
                s.push('.');
            }
            s
        })
        .collect();
    let mut left = text
        .iter()
        .filter(|s| s.contains('.'))
        .map(|s| s.split('.').next().unwrap().len())
        .max()
        .unwrap_or(0);
    let right = text
        .iter()
        .filter_map(|s| s.split_once('.').map(|(_, r)| r.len()))
        .max()
        .unwrap_or(0);
    let special = text
        .iter()
        .filter(|s| !s.contains('.'))
        .map(String::len)
        .max()
        .unwrap_or(0);
    if !text.iter().any(|s| s.contains('.')) {
        return text.into_iter().map(|s| format!("{s:>special$}")).collect();
    }
    left = left.max(special.saturating_sub(1 + right));
    text.into_iter()
        .map(|s| {
            if let Some((l, r)) = s.split_once('.') {
                format!("{l:>left$}.{r:<right$}")
            } else {
                let width = left + 1 + right;
                format!("{s:>width$}")
            }
        })
        .collect()
}
fn selected(n: usize, summary: bool) -> Vec<Option<usize>> {
    if summary && n > 6 {
        (0..3)
            .map(Some)
            .chain([None])
            .chain((n - 3..n).map(Some))
            .collect()
    } else {
        (0..n).map(Some).collect()
    }
}
fn collect_visible<T: Element>(
    a: &Array<T>,
    axis: usize,
    c: &mut Vec<usize>,
    summary: bool,
    out: &mut Vec<T>,
) {
    if axis == a.ndim() {
        out.push(a.get(c));
        return;
    }
    for i in selected(a.shape()[axis], summary).into_iter().flatten() {
        c.push(i);
        collect_visible(a, axis + 1, c, summary, out);
        c.pop();
    }
}
struct Printer {
    text: Vec<String>,
    next: usize,
    debug: bool,
    summary: bool,
}
impl Printer {
    fn nested(&mut self, shape: &[usize], axis: usize, out: &mut String) {
        if axis == shape.len() {
            out.push_str(&self.text[self.next]);
            self.next += 1;
            return;
        }
        out.push('[');
        let choices = selected(shape[axis], self.summary);
        let last = axis + 1 == shape.len();
        let indent = axis + 1 + if self.debug { 6 } else { 0 };
        for (k, index) in choices.iter().enumerate() {
            if k > 0 {
                if last {
                    let width = if index.is_some() {
                        self.text[self.next].len()
                    } else {
                        3
                    };
                    let col = out.rsplit('\n').next().unwrap().len();
                    let sep = if self.debug { 2 } else { 1 };
                    if col + sep + width + 1 > 75 {
                        if self.debug {
                            out.push(',');
                        }
                        out.push('\n');
                        out.push_str(&" ".repeat(indent));
                    } else {
                        out.push_str(if self.debug { ", " } else { " " });
                    }
                } else {
                    if self.debug {
                        out.push(',');
                    }
                    out.push_str(&"\n".repeat(shape.len() - axis - 1));
                    out.push_str(&" ".repeat(indent));
                }
            }
            if index.is_none() {
                out.push_str("...");
            } else {
                self.nested(shape, axis + 1, out);
            }
        }
        out.push(']');
    }
}
fn visible_flat(shape: &[usize], axis: usize, base: usize, summary: bool, out: &mut Vec<usize>) {
    if axis == shape.len() {
        out.push(base);
        return;
    }
    let stride: usize = shape[axis + 1..].iter().product();
    for i in selected(shape[axis], summary).into_iter().flatten() {
        visible_flat(shape, axis + 1, base + i * stride, summary, out);
    }
}
/// Lays out arbitrary per-element text like NumPy's array printer (no column alignment).
///
/// `text(i)` returns the text of the element at C-order flat index `i`.
pub(crate) fn render_custom(shape: &[usize], text: &dyn Fn(usize) -> String) -> String {
    if shape.is_empty() {
        return text(0);
    }
    let total: usize = shape.iter().product();
    if total == 0 {
        return "[]".to_string();
    }
    let summary = total > 1000;
    let mut flat = Vec::new();
    visible_flat(shape, 0, 0, summary, &mut flat);
    let mut p = Printer {
        text: flat.into_iter().map(text).collect(),
        next: 0,
        debug: false,
        summary,
    };
    let mut out = String::new();
    p.nested(shape, 0, &mut out);
    out
}
fn render<T: Element>(a: &Array<T>, debug: bool) -> String {
    let mut out = if debug {
        "array(".to_string()
    } else {
        String::new()
    };
    if a.ndim() == 0 {
        let v = a.item();
        if debug && (crate::element::is_float_dtype(T::DTYPE) || T::DTYPE.starts_with("complex")) {
            let t = format_values(&[v]);
            out.push_str(t[0].trim());
        } else if T::DTYPE.starts_with("complex") {
            let t = format_values(&[v]);
            out.push_str(&format!("({})", t[0].trim()));
        } else {
            out.push_str(&raw(v));
        }
    } else if a.is_empty() {
        out.push_str("[]");
    } else {
        let summary = a.size() > 1000;
        let mut values = Vec::new();
        collect_visible(a, 0, &mut Vec::new(), summary, &mut values);
        let mut p = Printer {
            text: format_values(&values),
            next: 0,
            debug,
            summary,
        };
        p.nested(a.shape(), 0, &mut out);
    }
    if debug {
        out.push_str(&format!(", dtype={})", T::DTYPE));
    }
    out
}
impl<T: Element> Display for Array<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.write_str(&render(self, false))
    }
}
impl<T: Element> Debug for Array<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.write_str(&render(self, true))
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn numpy_float_alignment() {
        assert_eq!(crate::array![1., 2.5, 3.].to_string(), "[1.  2.5 3. ]");
    }
    #[test]
    fn numpy_boolean_alignment() {
        assert_eq!(crate::array![true, false].to_string(), "[ True False]");
    }
    #[test]
    fn numpy_three_dimensions() {
        assert_eq!(
            crate::arange(0, 24, 1).reshape(&[2, 3, 4]).to_string(),
            "[[[ 0  1  2  3]\n  [ 4  5  6  7]\n  [ 8  9 10 11]]\n\n [[12 13 14 15]\n  [16 17 18 19]\n  [20 21 22 23]]]"
        );
    }
    #[test]
    fn numpy_scientific() {
        assert_eq!(
            crate::array![1e-5, 1., 3.].to_string(),
            "[1.e-05 1.e+00 3.e+00]"
        );
    }
    #[test]
    fn numpy_summary() {
        assert_eq!(
            crate::arange(0, 1001, 1).to_string(),
            "[   0    1    2 ...  998  999 1000]"
        );
    }
    #[test]
    fn debug_layout() {
        assert_eq!(
            format!("{:?}", crate::array![[1., 2.], [3., 4.]]),
            "array([[1., 2.],\n       [3., 4.]], dtype=float64)"
        );
    }
}
