//! I/O for arrays: .npy and CSV formats, like `np.save`/`np.load`/`np.savetxt`/`np.loadtxt`.
//!
//! # Architecture
//!
//! - `npy` (private): .npy format v1.0/v2.0 read/write on `(Vec<T>, shape)`.
//! - `csv` (private): CSV parse/format on `(Vec<T>, shape)`.
//! - The public API (in `api.rs.pending`) wraps these and returns `Array`
//!   — it will be wired after Phase 1 merges.
#![allow(dead_code)]

pub(crate) mod csv;
pub(crate) mod npy;

/// Errors produced by I/O operations.
#[derive(Debug)]
pub(crate) enum IoError {
    /// File or I/O error.
    Io(std::io::Error),
    /// Dtype mismatch between file and requested type.
    DType(String),
    /// Invalid format (bad .npy header, etc.).
    Format(String),
    /// CSV parse error.
    Parse(String),
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoError::Io(e) => write!(f, "io error: {e}"),
            IoError::DType(s) => write!(f, "dtype mismatch: {s}"),
            IoError::Format(s) => write!(f, "format error: {s}"),
            IoError::Parse(s) => write!(f, "parse error: {s}"),
        }
    }
}

impl std::error::Error for IoError {}

impl From<std::io::Error> for IoError {
    fn from(e: std::io::Error) -> Self {
        IoError::Io(e)
    }
}

pub(crate) type IoResult<T> = std::result::Result<T, IoError>;
