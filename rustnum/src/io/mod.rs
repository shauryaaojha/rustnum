//! I/O for arrays: `.npy`, `.npz` and CSV files, like `np.save`, `np.load`, `np.savez`,
//! `np.savetxt` and `np.loadtxt`.
//!
//! Every function returns [`crate::Result`]: file problems are [`crate::Error::Io`], a file
//! whose dtype differs from the requested element type is [`crate::Error::DType`], and malformed
//! content is [`crate::Error::Format`] or [`crate::Error::Parse`].
//!
//! ```
//! use rustnum as np;
//! let a = np::array![[1.5, 2.5], [3.5, 4.5]];
//! let bytes = np::io::to_npy_bytes(&a);
//! let b: np::Array<f64> = np::io::from_npy_bytes(&bytes).unwrap();
//! assert_eq!(a, b);
//! ```
//!
//! # Architecture
//!
//! - `npy` (private): `.npy` format v1.0/v2.0 on `(Vec<T>, shape)`.
//! - `csv` (private): CSV parse and format on `(Vec<T>, shape)`.
//! - `api`: the public functions, which wrap those kernels and return `Array`.
//! - `npz` (private module): ZIP/DEFLATE layer behind [`savez`], [`savez_compressed`],
//!   [`load_npz`] and [`NpzFile`], which are re-exported here.

mod api;
pub(crate) mod csv;
pub(crate) mod npy;
mod text;

pub use crate::npz::api::{NpzFile, load_npz, savez, savez_compressed};
pub(crate) use api::try_npy_bytes;
pub use api::{from_npy_bytes, load_csv, load_npy, save_csv, save_npy, to_npy_bytes};
pub use text::{genfromtxt, loadtxt, savetxt};
