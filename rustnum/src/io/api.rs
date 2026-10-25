//! Public I/O functions: `np::io::*`. Thin wrappers that move arrays through the byte-level
//! kernels in `npy` and `csv`.

use super::csv as csv_io;
use super::npy;
use crate::{Array, Element, Result};
use std::path::Path;

// ---- .npy ----

/// Saves an array to a `.npy` file, like `np.save(path, a)`.
///
/// Writes NumPy format v1.0 (v2.0 when the header exceeds 65535 bytes), little-endian, C order.
/// Views, transposes and other non-contiguous arrays are written in logical (C) order.
///
/// # Errors
///
/// Returns [`Error::Io`](crate::Error::Io) if the file cannot be written, and
/// [`Error::DType`](crate::Error::DType) for a user-defined element type that has no `.npy`
/// dtype.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// let a = np::arange(0.0_f64, 12.0, 1.0).reshape(&[3, 4]);
/// let path = std::env::temp_dir().join("rustnum_doc_save_npy.npy");
/// np::io::save_npy(&path, &a).unwrap();
/// let b: np::Array<f64> = np::io::load_npy(&path).unwrap();
/// assert_eq!(a, b);
/// # std::fs::remove_file(&path).unwrap();
/// ```
pub fn save_npy<T: Element>(path: impl AsRef<Path>, a: &Array<T>) -> Result<()> {
    std::fs::write(path, try_npy_bytes(a)?)?;
    Ok(())
}

/// Loads an array from a `.npy` file, like `np.load(path)`.
///
/// The dtype in the file must be exactly `T` (for example `<f8` for `f64`), otherwise
/// [`Error::DType`](crate::Error::DType) is returned; nothing is cast silently. Format versions
/// 1.0 and 2.0 are read, in C or Fortran order, including 0-d and empty arrays. The result is
/// always C-ordered.
///
/// # Errors
///
/// [`Error::Io`](crate::Error::Io) if the file cannot be read,
/// [`Error::DType`](crate::Error::DType) on a dtype mismatch,
/// [`Error::Format`](crate::Error::Format) for a malformed or truncated file.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// let a = np::array![1_i64, 2, 3];
/// let path = std::env::temp_dir().join("rustnum_doc_load_npy.npy");
/// np::io::save_npy(&path, &a).unwrap();
/// let b: np::Array<i64> = np::io::load_npy(&path).unwrap();
/// assert_eq!(a, b);
/// // Asking for another dtype is an error.
/// assert!(matches!(np::io::load_npy::<f64>(&path), Err(np::Error::DType { .. })));
/// # std::fs::remove_file(&path).unwrap();
/// ```
pub fn load_npy<T: Element>(path: impl AsRef<Path>) -> Result<Array<T>> {
    from_npy_bytes(&std::fs::read(path)?)
}

/// Serializes an array to `.npy` bytes in memory, like `np.save` into a buffer.
///
/// # Panics
///
/// Panics if `T` is a user-defined [`Element`] with no `.npy` dtype; the built-in types
/// (`f64 f32 i64 i32 u8 bool usize`) never panic.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// let a = np::Array::from_vec(vec![1.0_f64, 2.0, 3.0], &[3]);
/// let bytes = np::io::to_npy_bytes(&a);
/// assert_eq!(&bytes[..6], b"\x93NUMPY");
/// let b: np::Array<f64> = np::io::from_npy_bytes(&bytes).unwrap();
/// assert_eq!(a, b);
/// ```
pub fn to_npy_bytes<T: Element>(a: &Array<T>) -> Vec<u8> {
    try_npy_bytes(a).unwrap_or_else(|e| panic!("{e}"))
}

/// Fallible core of [`to_npy_bytes`]; `Err` only for element types outside the built-in dtypes.
pub(crate) fn try_npy_bytes<T: Element>(a: &Array<T>) -> Result<Vec<u8>> {
    npy::encode_elements(a.to_vec(), a.shape())
}

/// Deserializes an array from `.npy` bytes in memory (see [`load_npy`] for the rules).
///
/// # Errors
///
/// [`Error::DType`](crate::Error::DType) on a dtype mismatch,
/// [`Error::Format`](crate::Error::Format) for malformed or truncated input.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// let a = np::Array::from_vec(vec![1_i64, 2, 3], &[3]);
/// let bytes = np::io::to_npy_bytes(&a);
/// let b: np::Array<i64> = np::io::from_npy_bytes(&bytes).unwrap();
/// assert_eq!(a, b);
/// assert!(np::io::from_npy_bytes::<f64>(&bytes).is_err());
/// assert!(np::io::from_npy_bytes::<i64>(b"not a npy file").is_err());
/// ```
pub fn from_npy_bytes<T: Element>(bytes: &[u8]) -> Result<Array<T>> {
    let (data, shape) = npy::decode_elements::<T>(bytes)?;
    Ok(Array::from_vec(data, &shape))
}

// ---- CSV ----

/// Saves a 1-d or 2-d array as delimited text, like `np.savetxt(path, a, delimiter=...)`.
///
/// A 1-d array is written one value per line. Values use Rust's shortest round-trip
/// formatting, so floats are read back exactly.
///
/// # Errors
///
/// [`Error::Format`](crate::Error::Format) if the array is not 1-d or 2-d,
/// [`Error::Io`](crate::Error::Io) if the file cannot be written.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// let a = np::arange(0.0_f64, 6.0, 1.0).reshape(&[2, 3]);
/// let path = std::env::temp_dir().join("rustnum_doc_save_csv.csv");
/// np::io::save_csv(&path, &a, ',').unwrap();
/// assert_eq!(std::fs::read_to_string(&path).unwrap(), "0,1,2\n3,4,5\n");
/// # std::fs::remove_file(&path).unwrap();
/// ```
pub fn save_csv<T>(path: impl AsRef<Path>, a: &Array<T>, delimiter: char) -> Result<()>
where
    T: Element + std::fmt::Display,
{
    let text = csv_io::format_csv(&a.to_vec(), a.shape(), delimiter)?;
    std::fs::write(path, text)?;
    Ok(())
}

/// Loads delimited text as a 1-d or 2-d array, like `np.loadtxt(path, delimiter=..., skiprows=...)`.
///
/// The first `skip_header` lines are ignored; blank lines and `#` comments are skipped. As in
/// NumPy, a single row or a single column comes back 1-d. With the delimiter `' '`, any run of
/// whitespace separates values.
///
/// # Errors
///
/// [`Error::Parse`](crate::Error::Parse) (with the line number) for a value that does not parse
/// as `T` or for ragged rows, [`Error::Io`](crate::Error::Io) if the file cannot be read.
///
/// # Examples
///
/// ```
/// use rustnum as np;
/// let path = std::env::temp_dir().join("rustnum_doc_load_csv.csv");
/// std::fs::write(&path, "x,y\n1,2\n3,4\n").unwrap();
/// let a: np::Array<i64> = np::io::load_csv(&path, ',', 1).unwrap();
/// assert_eq!(a, np::array![[1_i64, 2], [3, 4]]);
/// # std::fs::remove_file(&path).unwrap();
/// ```
pub fn load_csv<T>(path: impl AsRef<Path>, delimiter: char, skip_header: usize) -> Result<Array<T>>
where
    T: Element + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    let text = std::fs::read_to_string(path)?;
    let (data, shape) = csv_io::parse_csv::<T>(&text, delimiter, skip_header)?;
    Ok(Array::from_vec(data, &shape))
}
