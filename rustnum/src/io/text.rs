//! Text files: `loadtxt`, `savetxt` and `genfromtxt`, like NumPy's functions of the same name.
use crate::{Array, Element, Error, Float, Result};
use std::any::Any;
use std::path::Path;

// ------------------------------------------------------------------ parsing

fn cast<S: Any, T: Any>(v: S) -> Option<T> {
    (&mut Some(v) as &mut dyn Any)
        .downcast_mut::<Option<T>>()
        .and_then(Option::take)
}
/// Parse one token as element type `T` (None when it is not a valid value of that type).
fn parse_token<T: Element>(tok: &str) -> Option<T> {
    match T::DTYPE {
        "float64" => cast(tok.parse::<f64>().ok()?),
        "float32" => cast(tok.parse::<f32>().ok()?),
        "int32" => cast(tok.parse::<i32>().ok()?),
        "int64" => cast(tok.parse::<i64>().ok()?),
        "uint8" => cast(tok.parse::<u8>().ok()?),
        "uint64" => cast(tok.parse::<usize>().ok()?),
        _ => match tok {
            "True" | "true" | "1" => cast(true),
            "False" | "false" | "0" => cast(false),
            _ => None,
        },
    }
}
fn strip_comment<'a>(line: &'a str, comments: Option<&str>) -> &'a str {
    match comments {
        Some(c) if !c.is_empty() => line.find(c).map_or(line, |p| &line[..p]),
        _ => line,
    }
}
fn split_fields<'a>(line: &'a str, delimiter: Option<&str>) -> Vec<&'a str> {
    match delimiter {
        None => line.split_whitespace().collect(),
        Some("") => line.split_whitespace().collect(),
        Some(d) => line.split(d).map(str::trim).collect(),
    }
}
/// One data row: its 1-based line number in the file and its fields.
struct Row<'a> {
    line: usize,
    fields: Vec<&'a str>,
}
fn read_rows<'a>(
    text: &'a str,
    comments: Option<&str>,
    delimiter: Option<&str>,
    skip: usize,
) -> Vec<Row<'a>> {
    text.lines()
        .enumerate()
        .skip(skip)
        .filter_map(|(i, raw)| {
            let line = strip_comment(raw, comments).trim();
            (!line.is_empty()).then(|| Row {
                line: i + 1,
                fields: split_fields(line, delimiter),
            })
        })
        .collect()
}
fn pick_columns(usecols: Option<&[isize]>, ncols: usize, row: usize) -> Result<Vec<usize>> {
    match usecols {
        None => Ok((0..ncols).collect()),
        Some(cols) => cols
            .iter()
            .map(|&c| {
                let k = if c < 0 { c + ncols as isize } else { c };
                if k < 0 || k as usize >= ncols {
                    Err(Error::Parse(format!(
                        "invalid column index {c} at row {row} with {ncols} columns"
                    )))
                } else {
                    Ok(k as usize)
                }
            })
            .collect(),
    }
}
fn squeeze_shape(rows: usize, cols: usize) -> Vec<usize> {
    match (rows, cols) {
        (0, _) => vec![0],
        (1, 1) => vec![],
        (1, c) => vec![c],
        (r, 1) => vec![r],
        (r, c) => vec![r, c],
    }
}

/// Load data from a text file, like np.loadtxt(fname, comments, delimiter, skiprows, usecols, max_rows).
///
/// `comments` is the comment marker (`Some("#")` is NumPy's default; `None` disables comments).
/// `delimiter` `None` splits on whitespace. `skiprows` lines are skipped first; blank and comment
/// lines are ignored; `usecols` selects columns (negative counts from the end); `max_rows` limits
/// the number of data rows read. The result is squeezed like NumPy's (one column or one row gives a
/// 1-d array, a single value a 0-d array, no data an empty 1-d array).
///
/// # Errors
/// [`Error::Io`] if the file cannot be read; [`Error::Parse`] if a field is not a valid `T`
/// ("could not convert string 'x' to float64 at row 0, column 1."), the number of columns changes,
/// or a `usecols` index is out of range.
/// ```
/// use rustnum as np;
/// let path = std::env::temp_dir().join("rustnum_doc_loadtxt.txt");
/// std::fs::write(&path, "# x y\n1 2\n3 4  # tail\n").unwrap();
/// let a: np::Array<f64> = np::io::loadtxt(&path, Some("#"), None, 0, None, None).unwrap();
/// assert_eq!(a.to_vec(), [1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(a.shape(), [2, 2]);
/// let y: np::Array<f64> = np::io::loadtxt(&path, Some("#"), None, 0, Some(&[1]), None).unwrap();
/// assert_eq!(y.to_vec(), [2.0, 4.0]);
/// # std::fs::remove_file(&path).unwrap();
/// ```
pub fn loadtxt<T: Element>(
    fname: impl AsRef<Path>,
    comments: Option<&str>,
    delimiter: Option<&str>,
    skiprows: usize,
    usecols: Option<&[isize]>,
    max_rows: Option<usize>,
) -> Result<Array<T>> {
    let text = std::fs::read_to_string(fname)?;
    let mut rows = read_rows(&text, comments, delimiter, skiprows);
    if let Some(m) = max_rows {
        rows.truncate(m);
    }
    let mut data: Vec<T> = Vec::new();
    let mut ncols: Option<usize> = None;
    let mut picked = 0;
    for (r, row) in rows.iter().enumerate() {
        match ncols {
            None => ncols = Some(row.fields.len()),
            Some(n) if n != row.fields.len() => {
                return Err(Error::Parse(format!(
                    "the number of columns changed from {} to {} at row {}; use `usecols` to select a subset and avoid this error",
                    n,
                    row.fields.len(),
                    r + 1
                )));
            }
            _ => {}
        }
        let cols = pick_columns(usecols, row.fields.len(), r + 1)?;
        picked = cols.len();
        for c in cols {
            let tok = row.fields[c];
            data.push(parse_token::<T>(tok).ok_or_else(|| {
                Error::Parse(format!(
                    "could not convert string '{}' to {} at row {}, column {}.",
                    tok,
                    T::DTYPE,
                    r,
                    c + 1
                ))
            })?);
        }
    }
    let shape = squeeze_shape(rows.len(), picked);
    Ok(Array::from_vec(data, &shape))
}

// ------------------------------------------------------------------ printf

#[derive(Clone, Copy)]
enum Num {
    F(f64),
    I(i128),
}
struct Spec {
    minus: bool,
    plus: bool,
    space: bool,
    alt: bool,
    zero: bool,
    width: usize,
    prec: Option<usize>,
    conv: char,
}
fn parse_spec(fmt: &str) -> Result<Vec<(String, Option<Spec>)>> {
    // split into literal text and conversion specs
    let chars: Vec<char> = fmt.chars().collect();
    let mut out: Vec<(String, Option<Spec>)> = Vec::new();
    let mut lit = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '%' {
            lit.push(chars[i]);
            i += 1;
            continue;
        }
        i += 1;
        if i < chars.len() && chars[i] == '%' {
            lit.push('%');
            i += 1;
            continue;
        }
        let mut s = Spec {
            minus: false,
            plus: false,
            space: false,
            alt: false,
            zero: false,
            width: 0,
            prec: None,
            conv: 'e',
        };
        while i < chars.len() && "-+ #0".contains(chars[i]) {
            match chars[i] {
                '-' => s.minus = true,
                '+' => s.plus = true,
                ' ' => s.space = true,
                '#' => s.alt = true,
                _ => s.zero = true,
            }
            i += 1;
        }
        while i < chars.len() && chars[i].is_ascii_digit() {
            s.width = s.width * 10 + chars[i].to_digit(10).unwrap() as usize;
            i += 1;
        }
        if i < chars.len() && chars[i] == '.' {
            i += 1;
            let mut p = 0;
            while i < chars.len() && chars[i].is_ascii_digit() {
                p = p * 10 + chars[i].to_digit(10).unwrap() as usize;
                i += 1;
            }
            s.prec = Some(p);
        }
        match chars.get(i) {
            Some(&c) if "diueEfFgGs".contains(c) => s.conv = c,
            _ => {
                return Err(Error::Format(format!(
                    "unsupported or incomplete format specifier in fmt: {fmt}"
                )));
            }
        }
        i += 1;
        out.push((std::mem::take(&mut lit), Some(s)));
    }
    if !lit.is_empty() {
        out.push((lit, None));
    }
    Ok(out)
}
fn c_exp(x: f64, prec: usize, upper: bool) -> String {
    let s = format!("{:.*e}", prec, x);
    let (m, e) = s.split_once('e').expect("exponent");
    let e: i32 = e.parse().expect("exponent digits");
    format!(
        "{}{}{}{:02}",
        m,
        if upper { 'E' } else { 'e' },
        if e < 0 { '-' } else { '+' },
        e.abs()
    )
}
fn strip_zeros(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}
fn format_g(x: f64, prec: Option<usize>, alt: bool, upper: bool) -> String {
    let p = prec.unwrap_or(6).max(1);
    let exp: i32 = if x == 0.0 {
        0
    } else {
        let s = format!("{:.*e}", p - 1, x);
        s.split_once('e').unwrap().1.parse().unwrap()
    };
    if exp >= -4 && (exp as i64) < p as i64 {
        let s = format!("{:.*}", (p as i64 - 1 - exp as i64) as usize, x);
        if alt { s } else { strip_zeros(&s) }
    } else {
        let s = c_exp(x, p - 1, upper);
        if alt {
            return s;
        }
        let (m, e) = s.split_at(s.find(['e', 'E']).unwrap());
        format!("{}{}", strip_zeros(m), e)
    }
}
fn format_one(spec: &Spec, v: Num) -> String {
    let body = match (spec.conv, v) {
        ('d' | 'i' | 'u', Num::I(i)) => i.to_string(),
        ('d' | 'i' | 'u', Num::F(f)) => {
            if f.is_finite() {
                (f.trunc() as i128).to_string()
            } else {
                f.to_string()
            }
        }
        ('s', Num::I(i)) => i.to_string(),
        ('s', Num::F(f)) => format!("{f:?}"),
        (c, n) => {
            let f = match n {
                Num::F(f) => f,
                Num::I(i) => i as f64,
            };
            if !f.is_finite() {
                if f.is_nan() {
                    "nan".to_string()
                } else if f < 0.0 {
                    "-inf".to_string()
                } else {
                    "inf".to_string()
                }
            } else {
                match c {
                    'e' | 'E' => c_exp(f, spec.prec.unwrap_or(6), c == 'E'),
                    'f' | 'F' => format!("{:.*}", spec.prec.unwrap_or(6), f),
                    _ => format_g(f, spec.prec, spec.alt, c == 'G'),
                }
            }
        }
    };
    let (neg, digits) = match body.strip_prefix('-') {
        Some(rest) => (true, rest.to_string()),
        None => (false, body),
    };
    let sign = if neg {
        "-"
    } else if spec.plus {
        "+"
    } else if spec.space {
        " "
    } else {
        ""
    };
    let len = sign.len() + digits.chars().count();
    let finite_digits = digits.chars().next().is_some_and(|c| c.is_ascii_digit());
    if len >= spec.width {
        format!("{sign}{digits}")
    } else if spec.minus {
        format!("{sign}{digits}{}", " ".repeat(spec.width - len))
    } else if spec.zero && finite_digits {
        format!("{sign}{}{digits}", "0".repeat(spec.width - len))
    } else {
        format!("{}{sign}{digits}", " ".repeat(spec.width - len))
    }
}
fn apply(parts: &[(String, Option<Spec>)], values: &[Num]) -> String {
    let mut out = String::new();
    let mut k = 0;
    for (lit, spec) in parts {
        out.push_str(lit);
        if let Some(s) = spec {
            out.push_str(&format_one(s, values[k]));
            k += 1;
        }
    }
    out
}

/// Save an array to a text file, like np.savetxt(fname, X, fmt, delimiter, newline, header, footer, comments).
///
/// NumPy's defaults are `fmt` `"%.18e"`, `delimiter` `" "`, `newline` `"\n"`, `header` and `footer`
/// empty, `comments` `"# "`. `fmt` supports the printf conversions `d i u e E f F g G s` with
/// flags, width and precision: either one conversion applied to every column, or one per column
/// (used as the whole row format, ignoring `delimiter`). A 1-d array is written one value per line.
///
/// # Errors
/// [`Error::Io`] if the file cannot be written; [`Error::Format`] for a 0-d or 3+-d array
/// ("Expected 1D or 2D array, got 3D array instead") or a `fmt` with the wrong number of `%` formats.
/// ```
/// use rustnum as np;
/// let path = std::env::temp_dir().join("rustnum_doc_savetxt.txt");
/// let a = np::array![[1.0, 2.5], [3.0, 4.0]];
/// np::io::savetxt(&path, &a, "%.2f", ",", "\n", "x,y", "", "# ").unwrap();
/// assert_eq!(std::fs::read_to_string(&path).unwrap(), "# x,y\n1.00,2.50\n3.00,4.00\n");
/// # std::fs::remove_file(&path).unwrap();
/// ```
#[allow(clippy::too_many_arguments)]
pub fn savetxt<T: Element>(
    fname: impl AsRef<Path>,
    x: &Array<T>,
    fmt: &str,
    delimiter: &str,
    newline: &str,
    header: &str,
    footer: &str,
    comments: &str,
) -> Result<()> {
    if x.ndim() == 0 || x.ndim() > 2 {
        return Err(Error::Format(format!(
            "Expected 1D or 2D array, got {}D array instead",
            x.ndim()
        )));
    }
    let (rows, cols) = if x.ndim() == 1 {
        (x.size(), 1)
    } else {
        (x.shape()[0], x.shape()[1])
    };
    let parts = parse_spec(fmt)?;
    let n_specs = parts.iter().filter(|p| p.1.is_some()).count();
    if n_specs != 1 && n_specs != cols {
        return Err(Error::Format(format!(
            "fmt has wrong number of % formats:  {fmt}"
        )));
    }
    let mut out = String::new();
    if !header.is_empty() {
        out.push_str(comments);
        out.push_str(&header.replace('\n', &format!("\n{comments}")));
        out.push_str(newline);
    }
    let flat: Vec<Num> = x
        .iter()
        .map(|v| match T::DTYPE {
            "float64" | "float32" => Num::F(v.to_f64()),
            _ => Num::I(int_exact(v)),
        })
        .collect();
    for r in 0..rows {
        let row = &flat[r * cols..(r + 1) * cols];
        if n_specs == 1 {
            let cells: Vec<String> = row.iter().map(|v| apply(&parts, &[*v])).collect();
            out.push_str(&cells.join(delimiter));
        } else {
            out.push_str(&apply(&parts, row));
        }
        out.push_str(newline);
    }
    if !footer.is_empty() {
        out.push_str(comments);
        out.push_str(&footer.replace('\n', &format!("\n{comments}")));
        out.push_str(newline);
    }
    std::fs::write(fname, out)?;
    Ok(())
}
fn int_exact<T: Element>(v: T) -> i128 {
    if let Some(i) = (&v as &dyn Any).downcast_ref::<i64>() {
        *i as i128
    } else if let Some(u) = (&v as &dyn Any).downcast_ref::<usize>() {
        *u as i128
    } else {
        v.to_f64() as i128
    }
}

// ------------------------------------------------------------------ genfromtxt

fn valid_name(raw: &str) -> String {
    const DELETE: &str = "!#$%&'()*+,-./:;<=>?@[\\]^{|}~\"";
    raw.trim()
        .chars()
        .filter(|c| !DELETE.contains(*c))
        .map(|c| if c == ' ' { '_' } else { c })
        .collect()
}

/// Load data with missing values from a text file, like np.genfromtxt(fname, comments, delimiter, skip_header, skip_footer, filling_values, usecols, names, max_rows).
///
/// Float data only. An empty field, or any field that does not parse as a number, is missing and
/// becomes `filling_values` (NaN when `None`); like NumPy's float converter, a field that parses is
/// never treated as missing, so there is no `missing_values` argument. Blank and comment lines are
/// skipped; `skip_footer` drops that many data rows from the end. With `names` true the first line
/// after `skip_header` (it may start with the comment marker) holds the column names; they are
/// returned as the second element (empty otherwise) and the data keeps its 2-d shape. Without
/// `names` the result is squeezed like [`loadtxt`](crate::io::loadtxt).
///
/// # Errors
/// [`Error::Io`] if the file cannot be read; [`Error::Parse`] if rows have different numbers of
/// columns ("Some errors were detected !" with the line number) or a `usecols` index is invalid.
/// ```
/// use rustnum as np;
/// let path = std::env::temp_dir().join("rustnum_doc_genfromtxt.csv");
/// std::fs::write(&path, "a,b,c\n1,,3\n4,5,x\n").unwrap();
/// let (data, names) = np::io::genfromtxt::<f64>(&path, Some("#"), Some(","), 0, 0, None, None, true, None).unwrap();
/// assert_eq!(names, ["a", "b", "c"]);
/// assert_eq!(data.shape(), [2, 3]);
/// assert!(data[[0, 1]].is_nan() && data[[1, 2]].is_nan());
/// assert_eq!(data[[1, 1]], 5.0);
/// # std::fs::remove_file(&path).unwrap();
/// ```
#[allow(clippy::too_many_arguments)]
pub fn genfromtxt<F: Float>(
    fname: impl AsRef<Path>,
    comments: Option<&str>,
    delimiter: Option<&str>,
    skip_header: usize,
    skip_footer: usize,
    filling_values: Option<F>,
    usecols: Option<&[isize]>,
    names: bool,
    max_rows: Option<usize>,
) -> Result<(Array<F>, Vec<String>)> {
    let text = std::fs::read_to_string(fname)?;
    let mut skip = skip_header;
    let mut col_names: Vec<String> = Vec::new();
    if names
        && let Some((i, raw)) = text
            .lines()
            .enumerate()
            .skip(skip)
            .find(|(_, l)| !l.trim().is_empty())
    {
        let mut line = raw.trim();
        if let Some(c) = comments.filter(|c| !c.is_empty()) {
            line = line.trim_start_matches(c).trim();
        }
        col_names = split_fields(line, delimiter)
            .into_iter()
            .map(valid_name)
            .collect();
        skip = i + 1;
    }
    let mut rows = read_rows(&text, comments, delimiter, skip);
    let keep = rows.len().saturating_sub(skip_footer);
    rows.truncate(keep);
    if let Some(m) = max_rows {
        rows.truncate(m);
    }
    let fill = filling_values.unwrap_or(F::NAN);
    let mut data: Vec<F> = Vec::new();
    let mut ncols: Option<usize> = None;
    let mut picked = 0;
    for (r, row) in rows.iter().enumerate() {
        match ncols {
            None => ncols = Some(row.fields.len()),
            Some(n) if n != row.fields.len() => {
                return Err(Error::Parse(format!(
                    "Some errors were detected !\n    Line #{} (got {} columns instead of {})",
                    row.line,
                    row.fields.len(),
                    n
                )));
            }
            _ => {}
        }
        let cols = pick_columns(usecols, row.fields.len(), r + 1)?;
        picked = cols.len();
        for c in cols {
            let tok = row.fields[c];
            let v = parse_token::<F>(tok).unwrap_or(fill);
            data.push(v);
        }
    }
    let out = if names {
        Array::from_vec(data, &[rows.len(), picked])
    } else {
        let shape = squeeze_shape(rows.len(), picked);
        Array::from_vec(data, &shape)
    };
    let col_names = match usecols {
        Some(cols) if !col_names.is_empty() => cols
            .iter()
            .filter_map(|&c| {
                let k = if c < 0 {
                    c + col_names.len() as isize
                } else {
                    c
                };
                col_names.get(k as usize).cloned()
            })
            .collect(),
        _ => col_names,
    };
    Ok((out, col_names))
}
