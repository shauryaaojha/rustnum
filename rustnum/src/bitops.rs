//! Bitwise operations on integer arrays: `bitwise_and`, `left_shift`, `packbits`, `bitwise_count`, ...
use crate::lanes::map_lanes;
use crate::slicing::axis_index;
use crate::{Array, Element, IntoArray, Num};
use std::ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr};

/// Bit-level helpers of the integer element types (sealed; implemented for i32, i64, u8 and usize).
pub trait IntHooks: Copy {
    /// Width of the type in bits.
    const BITS: u32;
    /// Shift left; counts outside `0..BITS` give 0 like NumPy.
    fn shl_np(self, n: Self) -> Self;
    /// Shift right; counts outside `0..BITS` give 0, or -1 for negative values, like NumPy.
    fn shr_np(self, n: Self) -> Self;
    /// Number of set bits of the absolute value.
    fn popcount_abs(self) -> u32;
}
/// An integer array element, like NumPy's integer dtypes (`i32`, `i64`, `u8`, `usize`).
/// ```
/// use rustnum as np;
/// fn is_integer<T: np::Integer>() {}
/// is_integer::<i64>();
/// ```
pub trait Integer:
    Num
    + Eq
    + Ord
    + IntHooks
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + Not<Output = Self>
{
}
macro_rules! integer {
    ($t:ty) => {
        impl IntHooks for $t {
            const BITS: u32 = <$t>::BITS;
            fn shl_np(self, n: Self) -> Self {
                if (n as i128) < 0 || (n as i128) >= <$t>::BITS as i128 {
                    0
                } else {
                    self << (n as u32)
                }
            }
            fn shr_np(self, n: Self) -> Self {
                if (n as i128) < 0 || (n as i128) >= <$t>::BITS as i128 {
                    if (self as i128) < 0 { !(0 as $t) } else { 0 }
                } else {
                    self >> (n as u32)
                }
            }
            fn popcount_abs(self) -> u32 {
                (self as i128).unsigned_abs().count_ones()
            }
        }
        impl Integer for $t {}
    };
}
integer!(i32);
integer!(i64);
integer!(u8);
integer!(usize);

/// Element-wise bitwise AND, like np.bitwise_and(a, b) or `a & b`.
/// ```
/// use rustnum as np;
/// assert_eq!(np::bitwise_and(np::array![12, 10], 6).to_vec(), [4, 2]);
/// ```
pub fn bitwise_and<T: Integer>(a: impl IntoArray<T>, b: impl IntoArray<T>) -> Array<T> {
    a.into_array().zip_map(b, |x, y| x & y)
}
/// Element-wise bitwise OR, like np.bitwise_or(a, b) or `a | b`.
/// ```
/// use rustnum as np;
/// assert_eq!(np::bitwise_or(np::array![12, 10], 6).to_vec(), [14, 14]);
/// ```
pub fn bitwise_or<T: Integer>(a: impl IntoArray<T>, b: impl IntoArray<T>) -> Array<T> {
    a.into_array().zip_map(b, |x, y| x | y)
}
/// Element-wise bitwise XOR, like np.bitwise_xor(a, b) or `a ^ b`.
/// ```
/// use rustnum as np;
/// assert_eq!(np::bitwise_xor(np::array![12, 10], 6).to_vec(), [10, 12]);
/// ```
pub fn bitwise_xor<T: Integer>(a: impl IntoArray<T>, b: impl IntoArray<T>) -> Array<T> {
    a.into_array().zip_map(b, |x, y| x ^ y)
}
/// Element-wise bitwise NOT, like np.bitwise_not(a) or `!a`.
/// ```
/// use rustnum as np;
/// assert_eq!(np::bitwise_not(&np::array![0, 5]).to_vec(), [-1, -6]);
/// assert_eq!(np::bitwise_not(&np::array![0u8, 5]).to_vec(), [255, 250]);
/// ```
pub fn bitwise_not<T: Integer>(a: &Array<T>) -> Array<T> {
    a.map(|x| !x)
}
/// Element-wise bitwise NOT, like np.invert(a).
/// ```
/// use rustnum as np;
/// assert_eq!(np::invert(&np::array![13u8]).to_vec(), [242]);
/// ```
pub fn invert<T: Integer>(a: &Array<T>) -> Array<T> {
    bitwise_not(a)
}
/// Shift bits left, like np.left_shift(a, n) or `a << n`.
///
/// A shift count that is negative or at least the bit width gives 0, as in NumPy.
/// ```
/// use rustnum as np;
/// assert_eq!(np::left_shift(np::array![1, 2], 3).to_vec(), [8, 16]);
/// ```
pub fn left_shift<T: Integer>(a: impl IntoArray<T>, n: impl IntoArray<T>) -> Array<T> {
    a.into_array().zip_map(n, |x, s| x.shl_np(s))
}
/// Shift bits right, like np.right_shift(a, n) or `a >> n`.
///
/// Signed values shift arithmetically; a count that is negative or at least the bit width gives 0
/// (or -1 for a negative value), as in NumPy.
/// ```
/// use rustnum as np;
/// assert_eq!(np::right_shift(np::array![16, -16], 2).to_vec(), [4, -4]);
/// ```
pub fn right_shift<T: Integer>(a: impl IntoArray<T>, n: impl IntoArray<T>) -> Array<T> {
    a.into_array().zip_map(n, |x, s| x.shr_np(s))
}
/// Number of set bits of the absolute value of each element, like np.bitwise_count(a).
/// ```
/// use rustnum as np;
/// assert_eq!(np::bitwise_count(&np::array![0, 1, 255, -7]).to_vec(), [0u8, 1, 8, 3]);
/// ```
pub fn bitwise_count<T: Integer>(a: &Array<T>) -> Array<u8> {
    a.map(|x| x.popcount_abs() as u8)
}

macro_rules! bit_operator {
    ($t:ty, $trait:ident, $method:ident, $f:expr) => {
        impl $trait<Array<$t>> for Array<$t> {
            type Output = Array<$t>;
            fn $method(self, r: Array<$t>) -> Array<$t> {
                self.zip_map(r, $f)
            }
        }
        impl $trait<&Array<$t>> for &Array<$t> {
            type Output = Array<$t>;
            fn $method(self, r: &Array<$t>) -> Array<$t> {
                self.zip_map(r, $f)
            }
        }
        impl $trait<&Array<$t>> for Array<$t> {
            type Output = Array<$t>;
            fn $method(self, r: &Array<$t>) -> Array<$t> {
                self.zip_map(r, $f)
            }
        }
        impl $trait<Array<$t>> for &Array<$t> {
            type Output = Array<$t>;
            fn $method(self, r: Array<$t>) -> Array<$t> {
                self.zip_map(r, $f)
            }
        }
        impl $trait<$t> for Array<$t> {
            type Output = Array<$t>;
            fn $method(self, r: $t) -> Array<$t> {
                self.zip_map(r, $f)
            }
        }
        impl $trait<$t> for &Array<$t> {
            type Output = Array<$t>;
            fn $method(self, r: $t) -> Array<$t> {
                self.zip_map(r, $f)
            }
        }
    };
}
macro_rules! bit_operators {
    ($t:ty) => {
        bit_operator!($t, BitAnd, bitand, |x: $t, y: $t| x & y);
        bit_operator!($t, BitOr, bitor, |x: $t, y: $t| x | y);
        bit_operator!($t, BitXor, bitxor, |x: $t, y: $t| x ^ y);
        bit_operator!($t, Shl, shl, |x: $t, y: $t| x.shl_np(y));
        bit_operator!($t, Shr, shr, |x: $t, y: $t| x.shr_np(y));
        impl Not for Array<$t> {
            type Output = Array<$t>;
            fn not(self) -> Array<$t> {
                self.map(|x| !x)
            }
        }
        impl Not for &Array<$t> {
            type Output = Array<$t>;
            fn not(self) -> Array<$t> {
                self.map(|x| !x)
            }
        }
    };
}
bit_operators!(i32);
bit_operators!(i64);
bit_operators!(u8);
bit_operators!(usize);

fn bit_order(bitorder: &str) -> bool {
    match bitorder.chars().next() {
        Some('b') | Some('B') => true,
        Some('l') | Some('L') => false,
        _ => panic!("'order' must begin with 'l' or 'b'"),
    }
}
/// Pack the bits of a boolean or integer array into `u8`s, like np.packbits(a, axis, bitorder).
///
/// Non-zero elements count as 1. With `axis` `None` the array is flattened first. The last byte is
/// padded with zero bits. `bitorder` is "big" (default in NumPy) or "little". Panics for float input.
/// ```
/// use rustnum as np;
/// let a = np::array![1, 0, 1, 0, 0, 0, 1, 1, 1, 1];
/// assert_eq!(np::packbits(&a, None, "big").to_vec(), [163, 192]);
/// assert_eq!(np::packbits(&a, None, "little").to_vec(), [197, 3]);
/// ```
pub fn packbits<T: Element>(a: &Array<T>, axis: Option<isize>, bitorder: &str) -> Array<u8> {
    assert!(
        !T::DTYPE.starts_with("float"),
        "Expected an input array of integer or boolean data type"
    );
    let big = bit_order(bitorder);
    let flat;
    let (src, ax) = match axis {
        None => {
            flat = a.ravel();
            (&flat, 0)
        }
        Some(ax) => (a, axis_index(ax, a.ndim())),
    };
    let n = src.shape()[ax];
    map_lanes(src, ax, n.div_ceil(8), |l, o| {
        o.fill(0);
        for (i, &x) in l.iter().enumerate() {
            if x != T::default() {
                let bit = if big { 7 - i % 8 } else { i % 8 };
                o[i / 8] |= 1 << bit;
            }
        }
    })
}
/// Unpack `u8`s into bits, like np.unpackbits(a, axis, count, bitorder).
///
/// With `axis` `None` the array is flattened first. `count` limits the number of bits per lane
/// (negative trims that many from the end; larger than available pads with zeros).
/// ```
/// use rustnum as np;
/// let a = np::array![163u8];
/// assert_eq!(np::unpackbits(&a, None, None, "big").to_vec(), [1, 0, 1, 0, 0, 0, 1, 1]);
/// assert_eq!(np::unpackbits(&a, None, Some(-3), "little").to_vec(), [1, 1, 0, 0, 0]);
/// ```
pub fn unpackbits(
    a: &Array<u8>,
    axis: Option<isize>,
    count: Option<isize>,
    bitorder: &str,
) -> Array<u8> {
    let big = bit_order(bitorder);
    let flat;
    let (src, ax) = match axis {
        None => {
            flat = a.ravel();
            (&flat, 0)
        }
        Some(ax) => (a, axis_index(ax, a.ndim())),
    };
    let total = (src.shape()[ax] * 8) as isize;
    let out_n = match count {
        None => total,
        Some(c) if c >= 0 => c,
        Some(c) => {
            assert!(total + c >= 0, "-count larger than number of elements");
            total + c
        }
    } as usize;
    map_lanes(src, ax, out_n, |l, o| {
        for (i, d) in o.iter_mut().enumerate() {
            *d = if i / 8 < l.len() {
                let bit = if big { 7 - i % 8 } else { i % 8 };
                (l[i / 8] >> bit) & 1
            } else {
                0
            };
        }
    })
}
