//! .npy format v1.0/v2.0 read and write.
//!
//! Operates on `(Vec<T>, Vec<usize>)` pairs — no `Array` dependency.
//!
//! Format reference: <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>

use crate::{Element, Error, Result};
use std::any::{Any, TypeId};

// ---------------------------------------------------------------------------
// NpyHeader
// ---------------------------------------------------------------------------

/// Parsed .npy header.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NpyHeader {
    pub descr: String,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

// ---------------------------------------------------------------------------
// NpyElement trait
// ---------------------------------------------------------------------------

/// Types that can be serialized to/from .npy element bytes.
pub(crate) trait NpyElement: Copy + Sized {
    /// NumPy dtype descriptor, e.g. `"<f8"`.
    const DESCR: &'static str;
    /// Byte size of one element in the file.
    const BYTE_SIZE: usize;
    /// Decode from little-endian bytes.
    fn from_le_bytes(bytes: &[u8]) -> Self;
    /// Encode to little-endian bytes (appended to `out`).
    fn write_le_bytes(self, out: &mut Vec<u8>);
}

impl NpyElement for f64 {
    const DESCR: &'static str = "<f8";
    const BYTE_SIZE: usize = 8;
    fn from_le_bytes(b: &[u8]) -> Self {
        f64::from_le_bytes(b[..8].try_into().unwrap())
    }
    fn write_le_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl NpyElement for f32 {
    const DESCR: &'static str = "<f4";
    const BYTE_SIZE: usize = 4;
    fn from_le_bytes(b: &[u8]) -> Self {
        f32::from_le_bytes(b[..4].try_into().unwrap())
    }
    fn write_le_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl NpyElement for i64 {
    const DESCR: &'static str = "<i8";
    const BYTE_SIZE: usize = 8;
    fn from_le_bytes(b: &[u8]) -> Self {
        i64::from_le_bytes(b[..8].try_into().unwrap())
    }
    fn write_le_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl NpyElement for i32 {
    const DESCR: &'static str = "<i4";
    const BYTE_SIZE: usize = 4;
    fn from_le_bytes(b: &[u8]) -> Self {
        i32::from_le_bytes(b[..4].try_into().unwrap())
    }
    fn write_le_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl NpyElement for u8 {
    const DESCR: &'static str = "|u1";
    const BYTE_SIZE: usize = 1;
    fn from_le_bytes(b: &[u8]) -> Self {
        b[0]
    }
    fn write_le_bytes(self, out: &mut Vec<u8>) {
        out.push(self);
    }
}

impl NpyElement for bool {
    const DESCR: &'static str = "|b1";
    const BYTE_SIZE: usize = 1;
    fn from_le_bytes(b: &[u8]) -> Self {
        b[0] != 0
    }
    fn write_le_bytes(self, out: &mut Vec<u8>) {
        out.push(if self { 1 } else { 0 });
    }
}

impl NpyElement for usize {
    const DESCR: &'static str = "<u8";
    const BYTE_SIZE: usize = 8;
    fn from_le_bytes(b: &[u8]) -> Self {
        u64::from_le_bytes(b[..8].try_into().unwrap()) as usize
    }
    fn write_le_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self as u64).to_le_bytes());
    }
}

// ---------------------------------------------------------------------------
// Header parsing
// ---------------------------------------------------------------------------

/// Parse a .npy header from the beginning of a byte slice.
/// Returns `(header, data_offset)`.
pub(crate) fn parse_npy_header(bytes: &[u8]) -> Result<(NpyHeader, usize)> {
    // Magic number: \x93NUMPY
    if bytes.len() < 10 {
        return Err(Error::Format("file too short for .npy header".into()));
    }
    if &bytes[0..6] != b"\x93NUMPY" {
        return Err(Error::Format("not a .npy file (bad magic)".into()));
    }

    let major = bytes[6];
    let minor = bytes[7];

    let (header_len, header_start) = match (major, minor) {
        (1, 0) => {
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (len, 10)
        }
        (2, 0) => {
            if bytes.len() < 12 {
                return Err(Error::Format("file too short for v2.0 header".into()));
            }
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12)
        }
        _ => {
            return Err(Error::Format(format!(
                "unsupported .npy version {major}.{minor}"
            )));
        }
    };

    let header_end = header_start + header_len;
    if bytes.len() < header_end {
        return Err(Error::Format("header truncated".into()));
    }

    let header_str = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|e| Error::Format(format!("invalid header UTF-8: {e}")))?;

    let header = parse_python_dict(header_str)?;
    Ok((header, header_end))
}

/// Parse a Python dict string like `{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }`.
fn parse_python_dict(s: &str) -> Result<NpyHeader> {
    let s = s.trim();

    // Extract descr
    let descr = extract_string_value(s, "descr")?;

    // Extract fortran_order
    let fortran_order = extract_bool_value(s, "fortran_order")?;

    // Extract shape
    let shape = extract_shape(s)?;

    Ok(NpyHeader {
        descr,
        fortran_order,
        shape,
    })
}

fn extract_string_value(s: &str, key: &str) -> Result<String> {
    // Look for 'key': 'value' or "key": "value"
    let pattern = format!("'{key}'");
    let pos = s
        .find(&pattern)
        .or_else(|| s.find(&format!("\"{key}\"")))
        .ok_or_else(|| Error::Format(format!("missing key '{key}' in header")))?;

    let after_key = &s[pos + pattern.len()..];
    let after_colon = after_key
        .find(':')
        .map(|i| &after_key[i + 1..])
        .ok_or_else(|| Error::Format(format!("missing ':' after key '{key}'")))?;
    let after_colon = after_colon.trim_start();

    // Find the quoted value
    let quote = after_colon
        .chars()
        .next()
        .ok_or_else(|| Error::Format(format!("empty value for key '{key}'")))?;
    if quote != '\'' && quote != '"' {
        return Err(Error::Format(format!(
            "expected quote for key '{key}', got '{quote}'"
        )));
    }
    let rest = &after_colon[1..];
    let end = rest
        .find(quote)
        .ok_or_else(|| Error::Format(format!("unterminated string for key '{key}'")))?;
    Ok(rest[..end].to_string())
}

fn extract_bool_value(s: &str, key: &str) -> Result<bool> {
    let pattern = format!("'{key}'");
    let pos = s
        .find(&pattern)
        .or_else(|| s.find(&format!("\"{key}\"")))
        .ok_or_else(|| Error::Format(format!("missing key '{key}' in header")))?;

    let after_key = &s[pos + pattern.len()..];
    let after_colon = after_key
        .find(':')
        .map(|i| &after_key[i + 1..])
        .ok_or_else(|| Error::Format(format!("missing ':' after key '{key}'")))?;
    let after_colon = after_colon.trim_start();

    if after_colon.starts_with("True") {
        Ok(true)
    } else if after_colon.starts_with("False") {
        Ok(false)
    } else {
        Err(Error::Format(format!(
            "expected True/False for key '{key}'"
        )))
    }
}

fn extract_shape(s: &str) -> Result<Vec<usize>> {
    let pattern = "'shape'";
    let pos = s
        .find(pattern)
        .or_else(|| s.find("\"shape\""))
        .ok_or_else(|| Error::Format("missing key 'shape' in header".into()))?;

    let after_key = &s[pos + pattern.len()..];
    let after_colon = after_key
        .find(':')
        .map(|i| &after_key[i + 1..])
        .ok_or_else(|| Error::Format("missing ':' after key 'shape'".into()))?;
    let after_colon = after_colon.trim_start();

    // Find the tuple
    let paren_start = after_colon
        .find('(')
        .ok_or_else(|| Error::Format("missing '(' in shape".into()))?;
    let paren_end = after_colon
        .find(')')
        .ok_or_else(|| Error::Format("missing ')' in shape".into()))?;

    let inner = &after_colon[paren_start + 1..paren_end];
    let inner = inner.trim();

    if inner.is_empty() {
        return Ok(vec![]); // 0-d scalar
    }

    let mut dims = Vec::new();
    for part in inner.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue; // trailing comma
        }
        let dim: usize = part
            .parse()
            .map_err(|e| Error::Format(format!("invalid shape dimension '{part}': {e}")))?;
        dims.push(dim);
    }
    Ok(dims)
}

// ---------------------------------------------------------------------------
// Header writing
// ---------------------------------------------------------------------------

/// Build a .npy header (v1.0 or v2.0) for the given descr and shape.
pub(crate) fn write_npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let shape_str = if shape.is_empty() {
        "()".to_string()
    } else if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("({})", parts.join(", "))
    };

    let dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr, shape_str
    );

    // Try v1.0 first: magic(6) + version(2) + header_len(2) + header = total
    // Total must be divisible by 64.
    let magic_and_version_len = 10; // 6 + 2 + 2
    let needed = magic_and_version_len + dict.len() + 1; // +1 for \n
    let padded = needed.div_ceil(64) * 64;
    let header_len = padded - magic_and_version_len;

    if header_len <= 65535 {
        // v1.0
        let mut buf = Vec::with_capacity(padded);
        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(1); // major
        buf.push(0); // minor
        buf.extend_from_slice(&(header_len as u16).to_le_bytes());
        buf.extend_from_slice(dict.as_bytes());
        // Pad with spaces, end with \n
        while buf.len() < padded - 1 {
            buf.push(b' ');
        }
        buf.push(b'\n');
        buf
    } else {
        // v2.0: magic(6) + version(2) + header_len(4) + header
        let magic_and_version_len = 12;
        let needed = magic_and_version_len + dict.len() + 1;
        let padded = needed.div_ceil(64) * 64;
        let header_len = padded - magic_and_version_len;

        let mut buf = Vec::with_capacity(padded);
        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(2); // major
        buf.push(0); // minor
        buf.extend_from_slice(&(header_len as u32).to_le_bytes());
        buf.extend_from_slice(dict.as_bytes());
        while buf.len() < padded - 1 {
            buf.push(b' ');
        }
        buf.push(b'\n');
        buf
    }
}

// ---------------------------------------------------------------------------
// Encode / Decode
// ---------------------------------------------------------------------------

/// Encode data and shape into .npy bytes.
pub(crate) fn encode_npy<T: NpyElement>(data: &[T], shape: &[usize]) -> Vec<u8> {
    let header = write_npy_header(T::DESCR, shape);
    let mut buf = header;
    for &val in data {
        val.write_le_bytes(&mut buf);
    }
    buf
}

/// NumPy dtype name for a `descr` string such as `"<f8"`; unknown descriptors are returned as is.
fn descr_to_dtype(descr: &str) -> String {
    match descr {
        "<f8" | "=f8" => "float64",
        "<f4" | "=f4" => "float32",
        "<i8" | "=i8" => "int64",
        "<i4" | "=i4" => "int32",
        "|u1" | "<u1" => "uint8",
        "<u8" | "=u8" => "uint64",
        "|b1" => "bool",
        other => other,
    }
    .to_string()
}

/// Decode .npy bytes into (Vec<T>, shape).
pub(crate) fn decode_npy<T: NpyElement>(bytes: &[u8]) -> Result<(Vec<T>, Vec<usize>)> {
    let (header, data_offset) = parse_npy_header(bytes)?;

    // Check dtype
    if header.descr != T::DESCR {
        return Err(Error::DType {
            expected: descr_to_dtype(T::DESCR),
            found: descr_to_dtype(&header.descr),
        });
    }

    // An empty product is 1, which is the element count of a 0-d array.
    let n_elements = header
        .shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| Error::Format("shape in .npy header is too large".into()))?;

    let data_bytes = &bytes[data_offset..];
    let expected_bytes = n_elements
        .checked_mul(T::BYTE_SIZE)
        .ok_or_else(|| Error::Format("shape in .npy header is too large".into()))?;
    if data_bytes.len() < expected_bytes {
        return Err(Error::Format(format!(
            "data truncated: need {expected_bytes} bytes, have {}",
            data_bytes.len()
        )));
    }

    let mut data = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        let offset = i * T::BYTE_SIZE;
        data.push(T::from_le_bytes(&data_bytes[offset..]));
    }

    // Handle Fortran order
    if header.fortran_order && header.shape.len() > 1 {
        data = fortran_to_c_order(&data, &header.shape);
    }

    Ok((data, header.shape))
}

/// Encode elements of any [`Element`] type, choosing the `.npy` descriptor from the type.
///
/// `Element` is open to downstream implementors, so a type that is not one of the seven
/// built-in dtypes is a [`Error::DType`] instead of a compile error.
pub(crate) fn encode_elements<T: Element>(data: Vec<T>, shape: &[usize]) -> Result<Vec<u8>> {
    macro_rules! arm {
        ($($t:ty),*) => {$(
            if TypeId::of::<T>() == TypeId::of::<$t>() {
                let boxed: Box<dyn Any> = Box::new(data);
                let d = boxed.downcast::<Vec<$t>>().expect("type id was checked");
                return Ok(encode_npy(&d, shape));
            }
        )*};
    }
    arm!(f64, f32, i64, i32, u8, bool, usize);
    Err(Error::DType {
        expected: "a built-in rustnum dtype".into(),
        found: T::DTYPE.into(),
    })
}

/// Decode `.npy` bytes into any [`Element`] type (see [`encode_elements`]).
pub(crate) fn decode_elements<T: Element>(bytes: &[u8]) -> Result<(Vec<T>, Vec<usize>)> {
    macro_rules! arm {
        ($($t:ty),*) => {$(
            if TypeId::of::<T>() == TypeId::of::<$t>() {
                let (v, shape) = decode_npy::<$t>(bytes)?;
                let boxed: Box<dyn Any> = Box::new(v);
                let v = boxed.downcast::<Vec<T>>().expect("type id was checked");
                return Ok((*v, shape));
            }
        )*};
    }
    arm!(f64, f32, i64, i32, u8, bool, usize);
    Err(Error::DType {
        expected: "a built-in rustnum dtype".into(),
        found: T::DTYPE.into(),
    })
}

/// Reorder data from Fortran (column-major) to C (row-major) order.
fn fortran_to_c_order<T: Copy>(data: &[T], shape: &[usize]) -> Vec<T> {
    let ndim = shape.len();
    let total: usize = shape.iter().product();
    if total == 0 {
        return vec![];
    }

    // Compute Fortran strides
    let mut f_strides = vec![1usize; ndim];
    for i in 1..ndim {
        f_strides[i] = f_strides[i - 1] * shape[i - 1];
    }

    // Compute C strides
    let mut c_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        c_strides[i] = c_strides[i + 1] * shape[i + 1];
    }

    let mut result = vec![data[0]; total]; // dummy init
    let mut indices = vec![0usize; ndim];

    for (c_idx, slot) in result.iter_mut().enumerate() {
        // Compute multi-index from C linear index
        let mut rem = c_idx;
        for d in 0..ndim {
            indices[d] = rem / c_strides[d];
            rem %= c_strides[d];
        }
        // Compute Fortran linear index
        let f_idx: usize = indices
            .iter()
            .zip(f_strides.iter())
            .map(|(&i, &s)| i * s)
            .sum();
        *slot = data[f_idx];
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip_f64() {
        let header_bytes = write_npy_header("<f8", &[3, 4]);
        let (header, offset) = parse_npy_header(&header_bytes).unwrap();
        assert_eq!(header.descr, "<f8");
        assert!(!header.fortran_order);
        assert_eq!(header.shape, vec![3, 4]);
        assert_eq!(offset, header_bytes.len());
    }

    #[test]
    fn header_round_trip_1d() {
        let header_bytes = write_npy_header("<i8", &[10]);
        let (header, _) = parse_npy_header(&header_bytes).unwrap();
        assert_eq!(header.descr, "<i8");
        assert_eq!(header.shape, vec![10]);
    }

    #[test]
    fn header_round_trip_0d() {
        let header_bytes = write_npy_header("<f4", &[]);
        let (header, _) = parse_npy_header(&header_bytes).unwrap();
        assert_eq!(header.descr, "<f4");
        assert_eq!(header.shape, vec![]);
    }

    #[test]
    fn header_round_trip_3d() {
        let header_bytes = write_npy_header("|u1", &[2, 3, 4]);
        let (header, _) = parse_npy_header(&header_bytes).unwrap();
        assert_eq!(header.descr, "|u1");
        assert_eq!(header.shape, vec![2, 3, 4]);
    }

    #[test]
    fn encode_decode_f64() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let bytes = encode_npy(&data, &shape);
        let (decoded, decoded_shape) = decode_npy::<f64>(&bytes).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(decoded_shape, shape);
    }

    #[test]
    fn encode_decode_f32() {
        let data = vec![1.0_f32, 2.5, 3.75];
        let shape = vec![3];
        let bytes = encode_npy(&data, &shape);
        let (decoded, decoded_shape) = decode_npy::<f32>(&bytes).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(decoded_shape, shape);
    }

    #[test]
    fn encode_decode_i64() {
        let data = vec![10i64, -20, 30, -40];
        let shape = vec![2, 2];
        let bytes = encode_npy(&data, &shape);
        let (decoded, decoded_shape) = decode_npy::<i64>(&bytes).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(decoded_shape, shape);
    }

    #[test]
    fn encode_decode_i32() {
        let data = vec![1i32, -2, 3];
        let shape = vec![3];
        let bytes = encode_npy(&data, &shape);
        let (decoded, decoded_shape) = decode_npy::<i32>(&bytes).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(decoded_shape, shape);
    }

    #[test]
    fn encode_decode_u8() {
        let data = vec![0u8, 127, 255];
        let shape = vec![3];
        let bytes = encode_npy(&data, &shape);
        let (decoded, decoded_shape) = decode_npy::<u8>(&bytes).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(decoded_shape, shape);
    }

    #[test]
    fn encode_decode_bool() {
        let data = vec![true, false, true, false];
        let shape = vec![2, 2];
        let bytes = encode_npy(&data, &shape);
        let (decoded, decoded_shape) = decode_npy::<bool>(&bytes).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(decoded_shape, shape);
    }

    #[test]
    fn encode_decode_usize() {
        let data = vec![0usize, 42, usize::MAX >> 1]; // large but safe value
        let shape = vec![3];
        let bytes = encode_npy(&data, &shape);
        let (decoded, decoded_shape) = decode_npy::<usize>(&bytes).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(decoded_shape, shape);
    }

    #[test]
    fn encode_decode_0d() {
        let data = vec![42.0_f64];
        let shape = vec![];
        let bytes = encode_npy(&data, &shape);
        let (decoded, decoded_shape) = decode_npy::<f64>(&bytes).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(decoded_shape, shape);
    }

    #[test]
    fn encode_decode_empty() {
        let data: Vec<f64> = vec![];
        let shape = vec![0];
        let bytes = encode_npy(&data, &shape);
        let (decoded, decoded_shape) = decode_npy::<f64>(&bytes).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(decoded_shape, shape);
    }

    #[test]
    fn dtype_mismatch_error() {
        let data = vec![1.0_f64, 2.0];
        let bytes = encode_npy(&data, &[2]);
        let result = decode_npy::<i64>(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, Error::DType { .. }));
    }

    #[test]
    fn bad_magic_error() {
        let result = parse_npy_header(b"NOT_NPY_DATA");
        assert!(result.is_err());
    }

    #[test]
    fn fortran_to_c_order_2d() {
        // Fortran-ordered [1,3,2,4] for shape [2,2] should become C-ordered [1,2,3,4]
        let data = vec![1, 3, 2, 4];
        let result = fortran_to_c_order(&data, &[2, 2]);
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn fortran_to_c_order_3d() {
        // shape [2,3,1] — trivial 3d
        let data: Vec<i32> = (0..6).collect();
        let result = fortran_to_c_order(&data, &[2, 3, 1]);
        // F-order strides: [1, 2, 6]
        // C-order strides: [3, 1, 1] — wait, for shape [2,3,1]:
        // C strides: [3, 1, 1] -> but last dim is 1 so effectively [3, 1]
        // Let's verify manually:
        // C index (0,0,0) -> F offset = 0*1+0*2+0*6 = 0
        // C index (0,1,0) -> F offset = 0*1+1*2+0*6 = 2
        // C index (0,2,0) -> F offset = 0*1+2*2+0*6 = 4
        // C index (1,0,0) -> F offset = 1*1+0*2+0*6 = 1
        // C index (1,1,0) -> F offset = 1*1+1*2+0*6 = 3
        // C index (1,2,0) -> F offset = 1*1+2*2+0*6 = 5
        assert_eq!(result, vec![0, 2, 4, 1, 3, 5]);
    }

    #[test]
    fn v2_header_for_large_shape() {
        // Create a shape with many dimensions to force a long header
        let shape: Vec<usize> = (0..15000).map(|i| i + 1).collect();
        let header_bytes = write_npy_header("<f8", &shape);

        // Should be v2.0
        assert_eq!(header_bytes[6], 2);
        assert_eq!(header_bytes[7], 0);

        // Parse it back
        let (header, _) = parse_npy_header(&header_bytes).unwrap();
        assert_eq!(header.descr, "<f8");
        assert_eq!(header.shape, shape);
    }

    #[test]
    fn header_alignment() {
        // Verify all headers are 64-byte aligned
        for shape in [vec![], vec![1], vec![3, 4], vec![2, 3, 4], vec![100]] {
            for descr in ["<f8", "<f4", "<i8", "<i4", "|u1", "|b1", "<u8"] {
                let header = write_npy_header(descr, &shape);
                assert_eq!(
                    header.len() % 64,
                    0,
                    "header not 64-byte aligned for descr={descr} shape={shape:?}"
                );
            }
        }
    }

    #[test]
    fn shape_1_1_1() {
        let data = vec![99.0_f64];
        let shape = vec![1, 1, 1];
        let bytes = encode_npy(&data, &shape);
        let (decoded, decoded_shape) = decode_npy::<f64>(&bytes).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(decoded_shape, shape);
    }

    #[test]
    fn decode_fortran_order() {
        // Build a raw .npy with fortran_order: True for 2x3 matrix
        // C-order: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        // Fortran byte stream: col 0 [1.0, 4.0], col 1 [2.0, 5.0], col 2 [3.0, 6.0]
        let dict = "{'descr': '<f8', 'fortran_order': True, 'shape': (2, 3), }";
        let needed = 10 + dict.len() + 1;
        let padded = needed.div_ceil(64) * 64;
        let header_len = padded - 10;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(1);
        buf.push(0);
        buf.extend_from_slice(&(header_len as u16).to_le_bytes());
        buf.extend_from_slice(dict.as_bytes());
        while buf.len() < padded - 1 {
            buf.push(b' ');
        }
        buf.push(b'\n');

        // Append Fortran-ordered data
        let f_data = [1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
        for v in f_data {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let (decoded, shape) = decode_npy::<f64>(&buf).unwrap();
        assert_eq!(shape, vec![2, 3]);
        // After decoding, data must be in C order
        assert_eq!(decoded, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn python_npy_round_trip() {
        use std::process::Command;

        let py_available = Command::new("python")
            .args(["-c", "import numpy; print(numpy.__version__)"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        if !py_available {
            eprintln!("SKIP: python/numpy not available for npy round-trip test");
            return;
        }

        let dir = std::env::temp_dir().join("rustnum_npy_engine_test");
        std::fs::create_dir_all(&dir).unwrap();

        // 1. Python writes C-order 2x3 f64 array, Rust decodes
        let p1 = dir.join("py_c_order.npy");
        let py_code1 = format!(
            r#"
import numpy as np
a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
np.save(r'{}', a)
"#,
            p1.display()
        );
        let out1 = Command::new("python")
            .args(["-c", &py_code1])
            .output()
            .unwrap();
        assert!(out1.status.success());

        let bytes1 = std::fs::read(&p1).unwrap();
        let (data1, shape1) = decode_npy::<f64>(&bytes1).unwrap();
        assert_eq!(shape1, vec![2, 3]);
        assert_eq!(data1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // 2. Python writes Fortran-order array, Rust decodes into C-order
        let p2 = dir.join("py_fortran_order.npy");
        let py_code2 = format!(
            r#"
import numpy as np
a = np.asfortranarray(np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64))
np.save(r'{}', a)
"#,
            p2.display()
        );
        let out2 = Command::new("python")
            .args(["-c", &py_code2])
            .output()
            .unwrap();
        assert!(out2.status.success());

        let bytes2 = std::fs::read(&p2).unwrap();
        let (data2, shape2) = decode_npy::<f64>(&bytes2).unwrap();
        assert_eq!(shape2, vec![2, 3]);
        assert_eq!(data2, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

        // 3. Python writes 0-d scalar array, Rust decodes
        let p3 = dir.join("py_0d.npy");
        let py_code3 = format!(
            r#"
import numpy as np
a = np.array(42.5, dtype=np.float64)
np.save(r'{}', a)
"#,
            p3.display()
        );
        let out3 = Command::new("python")
            .args(["-c", &py_code3])
            .output()
            .unwrap();
        assert!(out3.status.success());

        let bytes3 = std::fs::read(&p3).unwrap();
        let (data3, shape3) = decode_npy::<f64>(&bytes3).unwrap();
        assert_eq!(shape3, Vec::<usize>::new());
        assert_eq!(data3, vec![42.5]);

        // 4. Rust encodes array, Python loads and asserts
        let p4 = dir.join("rust_encoded.npy");
        let rust_bytes = encode_npy(&[7i64, 8, 9, 10], &[2, 2]);
        std::fs::write(&p4, rust_bytes).unwrap();

        let py_code4 = format!(
            r#"
import numpy as np
a = np.load(r'{}')
assert a.dtype == np.int64, f"wrong dtype: {{a.dtype}}"
assert a.shape == (2, 2), f"wrong shape: {{a.shape}}"
assert np.array_equal(a, [[7, 8], [9, 10]]), f"wrong values: {{a}}"
"#,
            p4.display()
        );
        let out4 = Command::new("python")
            .args(["-c", &py_code4])
            .output()
            .unwrap();
        assert!(
            out4.status.success(),
            "Python verify failed: {}",
            String::from_utf8_lossy(&out4.stderr)
        );
    }
}
