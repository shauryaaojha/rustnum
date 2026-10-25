//! CSV format parse and format for 1-d and 2-d numeric data.
//!
//! Operates on `(Vec<T>, Vec<usize>)` pairs — no `Array` dependency.
//! Matches `np.loadtxt` and `np.savetxt` behavior.

use super::{IoError, IoResult};

/// Format 1-d or 2-d numeric data as CSV text, like `np.savetxt`.
///
/// # Errors
///
/// Returns `Err` if the array has more than 2 dimensions, or is 0-d,
/// or if the data length does not match the shape.
pub(crate) fn format_csv<T: std::fmt::Display>(
    data: &[T],
    shape: &[usize],
    delimiter: char,
) -> IoResult<String> {
    match shape.len() {
        1 => {
            let n = shape[0];
            if data.len() != n {
                return Err(IoError::Format(format!(
                    "data length {} does not match shape [{n}]",
                    data.len()
                )));
            }
            let mut out = String::new();
            for item in data {
                use std::fmt::Write;
                let _ = writeln!(out, "{item}");
            }
            Ok(out)
        }
        2 => {
            let rows = shape[0];
            let cols = shape[1];
            if data.len() != rows * cols {
                return Err(IoError::Format(format!(
                    "data length {} does not match shape [{rows}, {cols}]",
                    data.len()
                )));
            }
            let mut out = String::new();
            for r in 0..rows {
                for c in 0..cols {
                    if c > 0 {
                        out.push(delimiter);
                    }
                    use std::fmt::Write;
                    let _ = write!(out, "{}", data[r * cols + c]);
                }
                out.push('\n');
            }
            Ok(out)
        }
        ndim => Err(IoError::Format(format!(
            "Expected 1D or 2D array, got {ndim}D array instead"
        ))),
    }
}

/// Parse 1-d or 2-d numeric data from CSV text, like `np.loadtxt`.
///
/// Skips `skip_header` initial lines, comment lines starting with `'#'`,
/// inline comments after `'#'`, and blank lines.
///
/// # Errors
///
/// Returns `Err(IoError::Parse(...))` with line numbers on parse failures
/// or ragged rows.
pub(crate) fn parse_csv<T>(
    text: &str,
    delimiter: char,
    skip_header: usize,
) -> IoResult<(Vec<T>, Vec<usize>)>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    let mut data = Vec::new();
    let mut n_rows = 0usize;
    let mut expected_cols: Option<usize> = None;

    for (raw_line_idx, raw_line) in text.lines().enumerate() {
        let line_num = raw_line_idx + 1;
        if raw_line_idx < skip_header {
            continue;
        }

        // Strip comments starting with '#'
        let line = if let Some(comment_pos) = raw_line.find('#') {
            &raw_line[..comment_pos]
        } else {
            raw_line
        };

        let trimmed = line.trim();
        // Skip blank lines
        if trimmed.is_empty() {
            continue;
        }

        // Split tokens
        let tokens: Vec<&str> = if delimiter == ' ' {
            trimmed.split_whitespace().collect()
        } else {
            trimmed.split(delimiter).map(|s| s.trim()).collect()
        };

        let cols = tokens.len();
        if let Some(exp) = expected_cols {
            if cols != exp {
                return Err(IoError::Parse(format!(
                    "line {line_num}: expected {exp} columns, got {cols} (ragged CSV)"
                )));
            }
        } else {
            expected_cols = Some(cols);
        }

        for (col_idx, token) in tokens.into_iter().enumerate() {
            let val = token.parse::<T>().map_err(|e| {
                IoError::Parse(format!(
                    "line {line_num}: could not parse '{token}' at column {}: {e}",
                    col_idx + 1
                ))
            })?;
            data.push(val);
        }

        n_rows += 1;
    }

    let shape = match (n_rows, expected_cols) {
        (0, _) => vec![0],
        (1, Some(cols)) => vec![cols],
        (rows, Some(1)) => vec![rows],
        (rows, Some(cols)) => vec![rows, cols],
        _ => unreachable!(),
    };

    Ok((data, shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_and_parse_1d() {
        let data = vec![1.0_f64, 2.5, -3.75, 4.0];
        let shape = vec![4];
        let csv = format_csv(&data, &shape, ',').unwrap();
        assert_eq!(csv, "1\n2.5\n-3.75\n4\n");

        let (parsed_data, parsed_shape) = parse_csv::<f64>(&csv, ',', 0).unwrap();
        assert_eq!(parsed_data, data);
        assert_eq!(parsed_shape, shape);
    }

    #[test]
    fn format_and_parse_2d() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let csv = format_csv(&data, &shape, ',').unwrap();
        assert_eq!(csv, "1,2,3\n4,5,6\n");

        let (parsed_data, parsed_shape) = parse_csv::<f64>(&csv, ',', 0).unwrap();
        assert_eq!(parsed_data, data);
        assert_eq!(parsed_shape, shape);
    }

    #[test]
    fn format_and_parse_i64() {
        let data = vec![10i64, -20, 30, -40];
        let shape = vec![2, 2];
        let csv = format_csv(&data, &shape, ';').unwrap();
        assert_eq!(csv, "10;-20\n30;-40\n");

        let (parsed_data, parsed_shape) = parse_csv::<i64>(&csv, ';', 0).unwrap();
        assert_eq!(parsed_data, data);
        assert_eq!(parsed_shape, shape);
    }

    #[test]
    fn custom_delimiter_tab_and_space() {
        let text_tab = "1.0\t2.0\n3.0\t4.0\n";
        let (data, shape) = parse_csv::<f64>(text_tab, '\t', 0).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape, vec![2, 2]);

        let text_space = "  1.0   2.0 \n  3.0   4.0 \n";
        let (data2, shape2) = parse_csv::<f64>(text_space, ' ', 0).unwrap();
        assert_eq!(data2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape2, vec![2, 2]);
    }

    #[test]
    fn skip_header_option() {
        let text = "col1,col2\nunits,units\n1.0,2.0\n3.0,4.0\n";
        let (data, shape) = parse_csv::<f64>(text, ',', 2).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape, vec![2, 2]);
    }

    #[test]
    fn comments_and_blank_lines() {
        let text = r#"
# Starting comment
# Another comment line

1.0, 2.0 # inline comment 1

# Intermediate comment

3.0, 4.0

# Trailing comment
"#;
        let (data, shape) = parse_csv::<f64>(text, ',', 0).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape, vec![2, 2]);
    }

    #[test]
    fn parse_error_has_line_number() {
        let text = "1.0, 2.0\n3.0, not_a_number\n5.0, 6.0\n";
        let err = parse_csv::<f64>(text, ',', 0).unwrap_err();
        match err {
            IoError::Parse(msg) => {
                assert!(
                    msg.contains("line 2"),
                    "error message should contain line 2: {msg}"
                );
                assert!(
                    msg.contains("not_a_number"),
                    "error message should contain bad token: {msg}"
                );
            }
            _ => panic!("expected IoError::Parse, got {err:?}"),
        }
    }

    #[test]
    fn ragged_rows_error_has_line_number() {
        let text = "1.0, 2.0, 3.0\n4.0, 5.0\n";
        let err = parse_csv::<f64>(text, ',', 0).unwrap_err();
        match err {
            IoError::Parse(msg) => {
                assert!(msg.contains("line 2"), "error should specify line 2: {msg}");
                assert!(msg.contains("ragged"), "error should mention ragged: {msg}");
            }
            _ => panic!("expected IoError::Parse, got {err:?}"),
        }
    }

    #[test]
    fn format_csv_unsupported_ndim() {
        let err = format_csv(&[1.0], &[], ',').unwrap_err();
        assert!(matches!(err, IoError::Format(_)));

        let err3d = format_csv(&[1.0; 8], &[2, 2, 2], ',').unwrap_err();
        assert!(matches!(err3d, IoError::Format(_)));
    }

    #[test]
    fn empty_csv_handling() {
        let (data, shape) = parse_csv::<f64>("", ',', 0).unwrap();
        assert!(data.is_empty());
        assert_eq!(shape, vec![0]);

        let (data_comment, shape_comment) = parse_csv::<f64>("# nothing here\n\n", ',', 0).unwrap();
        assert!(data_comment.is_empty());
        assert_eq!(shape_comment, vec![0]);

        let formatted = format_csv::<f64>(&[], &[0], ',').unwrap();
        assert_eq!(formatted, "");
    }

    #[test]
    fn python_csv_round_trip() {
        use std::process::Command;

        // Skip gracefully if python is unavailable
        let py_available = Command::new("python")
            .args(["-c", "import numpy; print(numpy.__version__)"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        if !py_available {
            eprintln!("SKIP: python/numpy not available for CSV round-trip test");
            return;
        }

        let dir = std::env::temp_dir().join("rustnum_csv_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("py_saved.csv");

        // Python writes CSV via np.savetxt
        let py_write = format!(
            r#"
import numpy as np
a = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
np.savetxt(r'{}', a, delimiter=',', fmt='%.1f')
"#,
            path.display()
        );
        let out = Command::new("python")
            .args(["-c", &py_write])
            .output()
            .unwrap();
        assert!(out.status.success());

        // Rust parses it
        let content = std::fs::read_to_string(&path).unwrap();
        let (data, shape) = parse_csv::<f64>(&content, ',', 0).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(data, vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5]);

        // Rust formats CSV, Python reads via np.loadtxt
        let rust_csv = format_csv(&data, &shape, ',').unwrap();
        let rust_path = dir.join("rust_saved.csv");
        std::fs::write(&rust_path, rust_csv).unwrap();

        let py_read = format!(
            r#"
import numpy as np
loaded = np.loadtxt(r'{}', delimiter=',')
expected = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
assert np.allclose(loaded, expected), f"mismatch: {{loaded}} vs {{expected}}"
"#,
            rust_path.display()
        );
        let out2 = Command::new("python")
            .args(["-c", &py_read])
            .output()
            .unwrap();
        assert!(
            out2.status.success(),
            "Python loadtxt failed: {}",
            String::from_utf8_lossy(&out2.stderr)
        );
    }
}
