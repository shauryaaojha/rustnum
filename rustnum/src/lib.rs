//! NumPy-style arrays for Rust, like `numpy`.
//!
//! ```
//! use rustnum as np;
//! let a=np::array![[1,2],[3,4]];
//! assert_eq!(a.sum(),10);
//! ```
//!
//! Integer arithmetic retains the array dtype; addition, subtraction, multiplication,
//! and integer powers wrap on overflow. Generic callbacks are serial so they can
//! capture non-Sync state. Built-in element-wise operations and large contiguous
//! reductions use Rayon when enabled. Empty/all-NaN statistics return NaN without
//! Python's runtime warning mechanism. Axis parameters precede ddof.
//!
//! ```
//! use rustnum as np;
//! let a:np::Array<f64>=np::array![1.,2.];
//! assert_eq!((&a*2.).to_vec(),[2.,4.]);
//! assert_eq!((2.*&a).to_vec(),[2.,4.]);
//! assert_eq!(format!("{}",a),"[1. 2.]");
//! assert_eq!(format!("{:?}",a),"array([1., 2.], dtype=float64)");
//! ```

#![deny(missing_docs)]

mod array;
mod bitops;
mod broadcast;
mod complex_array;
mod creation;
mod display;
mod einsum_path;
mod element;
mod extra;
mod fancy;
mod functional;
mod half;
mod histext;
mod indexing;
mod kernels_mem;
mod lanes;
mod layout;
mod manip;
pub mod memory;
mod misc;
mod nanfn;
mod npz;
mod ops;
mod pool;
mod printopts;
mod reduce;
mod shapeextra;
mod simd_math;
mod slicing;
mod sortsearch;
mod stride;
mod timetypes;
mod ufunc;

pub mod datetime;
pub mod fft;
pub mod io;
pub mod linalg;
pub mod ma;
pub mod poly;
pub mod polynomial;
pub mod random;

pub use array::Array;
pub use bitops::{
    Integer, bitwise_and, bitwise_count, bitwise_not, bitwise_or, bitwise_xor, invert, left_shift,
    packbits, right_shift, unpackbits,
};
pub use broadcast::{IntoArray, broadcast_arrays, broadcast_shapes};
pub use complex_array::{Absolute, absolute};
pub use creation::*;
pub use einsum_path::einsum_path;
pub use element::{Element, Float, Num};
pub use extra::api::{
    Bins, ConvolveMode, Spacing, average, average_axis, bincount, convolve, corrcoef, correlate,
    cov, cross, diff, digitize, einsum, gradient, gradient_axis, histogram, histogram_bin_edges,
    interp, intersect1d, isin, kron, nanquantile, nanquantile_axis, quantile, quantile_axis,
    setdiff1d, setxor1d, trapezoid, union1d,
};
pub use fancy::{
    choose, ix_, put, put_along_axis, ravel_multi_index, take_along_axis, unravel_index,
};
pub use functional::{Vectorize, apply_along_axis, apply_over_axes, vectorize};
pub use half::{bf16, f16};
pub use histext::{histogram2d, histogramdd};
pub use indexing::{
    diag_indices, diag_indices_from, fill_diagonal, mask_indices, ndenumerate, ndindex,
    tril_indices, tril_indices_from, triu_indices, triu_indices_from,
};
pub use manip::*;
pub use misc::{
    Complex, angle, angle_deg, bartlett, blackman, copysign, divmod, ediff1d, ediff1d_with,
    float_power, floor_divide, fmod, gcd, hamming, hanning, heaviside, kaiser, lcm, ldexp,
    logaddexp, logaddexp2, nextafter, remainder,
};
pub use ops::*;
pub use polynomial::{Chebyshev, Hermite, HermiteE, Laguerre, Legendre, Polynomial};
pub use printopts::{
    PrintOptions, PrintOptionsGuard, get_printoptions, printoptions, set_printoptions,
};
pub use shapeextra::{
    around, atleast_1d, atleast_2d, atleast_3d, column_stack, dstack, frombuffer, fromiter,
    geomspace, real_if_close, rot90, row_stack, vander,
};
pub use slicing::{NEW_AXIS, SliceArg};
pub use sortsearch::{copyto, extract, flatnonzero, lexsort, piecewise, place, select, trim_zeros};
pub use stride::{as_strided, sliding_window_view};
pub use timetypes::{DateTime64, TimeDelta64, datetime_as_string, isnat};
pub use ufunc::*;
/// A data error, like NumPy's I/O, parsing, or linear algebra exceptions.
/// ```
/// use rustnum as np;
/// let e=np::Error::Singular;
/// assert_eq!(e.to_string(),"Singular matrix");
/// let _: &dyn std::error::Error=&e;
/// ```
#[derive(Debug)]
pub enum Error {
    /// Represents a Shape error, like a NumPy exception.
    /// ```
    /// use rustnum as np;
    /// let e=np::Error::Shape("bad shape".into()); assert_eq!(e.to_string(),"bad shape");
    /// ```
    Shape(String),
    /// Represents a DType error, like a NumPy exception.
    /// ```
    /// use rustnum as np;
    /// let e=np::Error::DType{expected:"float64".into(),found:"int32".into()}; assert!(e.to_string().contains("int32"));
    /// ```
    DType {
        /// Requested dtype, like NumPy dtype.
        /// ```
        /// let e=rustnum::Error::DType{expected:"float64".into(),found:"int32".into()};
        /// assert!(e.to_string().contains("int32"));
        /// ```
        expected: String,
        /// Encountered dtype, like NumPy dtype.
        /// ```
        /// let e=rustnum::Error::DType{expected:"float64".into(),found:"int32".into()};
        /// assert!(e.to_string().contains("int32"));
        /// ```
        found: String,
    },
    /// Represents a Singular error, like a NumPy exception.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Error::Singular.to_string(),"Singular matrix");
    /// ```
    Singular,
    /// Represents a NotPositiveDefinite error, like a NumPy exception.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Error::NotPositiveDefinite.to_string(),"Matrix is not positive definite");
    /// ```
    NotPositiveDefinite,
    /// Represents a Io error, like a NumPy exception.
    /// ```
    /// use rustnum as np;
    /// let e=np::Error::from(std::io::Error::other("failed")); assert_eq!(e.to_string(),"failed");
    /// ```
    Io(std::io::Error),
    /// Represents a Parse error, like a NumPy exception.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Error::Parse("bad number".into()).to_string(),"bad number");
    /// ```
    Parse(String),
    /// Represents a Format error, like a NumPy exception.
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::Error::Format("bad format".into()).to_string(),"bad format");
    /// ```
    Format(String),
    /// An iterative linear-algebra solver did not converge, like NumPy's `LinAlgError`.
    ///
    /// The message follows NumPy: `"Eigenvalues did not converge"` for `eig`/`eigh` and
    /// `"SVD did not converge"` for `svd` and the functions built on it. Non-finite input is
    /// the usual cause.
    /// ```
    /// use rustnum as np;
    /// let e=np::Error::NoConvergence("SVD did not converge".into());
    /// assert_eq!(e.to_string(),"SVD did not converge");
    /// let a=np::array![[1.0,2.0],[f64::NAN,4.0]];
    /// assert_eq!(np::linalg::eig(&a).unwrap_err().to_string(),"Eigenvalues did not converge");
    /// ```
    NoConvergence(String),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape(s) | Self::Parse(s) | Self::Format(s) | Self::NoConvergence(s) => {
                write!(f, "{s}")
            }
            Self::DType { expected, found } => {
                write!(f, "expected dtype {expected}, found {found}")
            }
            Self::Singular => write!(f, "Singular matrix"),
            Self::NotPositiveDefinite => write!(f, "Matrix is not positive definite"),
            Self::Io(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
/// A fallible result, like a NumPy call that may raise a data exception.
/// ```
/// use rustnum as np;
/// let result:np::Result<i32>=Ok(3);
/// assert_eq!(result.unwrap(),3);
/// ```
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    #[test]
    fn io_error_source() {
        use std::error::Error;
        let e = super::Error::from(std::io::Error::other("broken"));
        assert_eq!(e.source().unwrap().to_string(), "broken");
    }
}
