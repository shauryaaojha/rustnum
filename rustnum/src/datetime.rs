//! `datetime64` and `timedelta64` array elements, like NumPy's.
//!
//! The unit is part of the type: `DateTime64<datetime::D>` is `datetime64[D]` and
//! `TimeDelta64<datetime::ns>` is `timedelta64[ns]`. Values are `i64` counts since
//! 1970-01-01T00:00 (or plain counts for time deltas); `i64::MIN` is `NaT`, which never compares
//! equal to anything and sorts last, as in NumPy.
//!
//! Arithmetic works within one unit (`dt - dt = td`, `dt +- td`, `td +- td`, `td * int`,
//! `td / td = f64`); convert explicitly with `astype` (`a.astype::<DateTime64<s>>()`) first, which
//! follows NumPy's rules: datetimes use the calendar for years and months, time deltas treat a year
//! as 365.2425 days and a month as 30.436875 days, and coarser targets round toward minus infinity.
//!
//! ```
//! use rustnum as np;
//! use np::datetime::{D, s};
//! use np::{DateTime64, TimeDelta64};
//! let a = DateTime64::<D>::parse("2024-03-05").unwrap();
//! let b = DateTime64::<D>::parse("2024-01-01").unwrap();
//! assert_eq!(a - b, TimeDelta64::<D>::new(64));
//! assert_eq!(a.to_string(), "2024-03-05");
//! let t = a.cast::<s>();
//! assert_eq!(t.to_string(), "2024-03-05T00:00:00");
//! ```

use std::fmt;

mod sealed {
    pub trait Sealed {}
}

/// A NumPy time unit marker: `Y M W D h m s ms us ns`.
///
/// ```
/// use rustnum::datetime::{self, TimeUnit};
/// assert_eq!(<datetime::ns as TimeUnit>::NAME, "ns");
/// assert_eq!(<datetime::D as TimeUnit>::DATETIME_DTYPE, "datetime64[D]");
/// ```
pub trait TimeUnit:
    sealed::Sealed + Copy + Clone + Send + Sync + Default + fmt::Debug + PartialEq + 'static
{
    /// Unit symbol used by NumPy, e.g. `"D"`.
    const NAME: &'static str;
    /// Position in `Y M W D h m s ms us ns` (0 to 9).
    const CODE: u8;
    /// Plural unit word used when printing time deltas, e.g. `"days"`.
    const PLURAL: &'static str;
    /// `datetime64[unit]`.
    const DATETIME_DTYPE: &'static str;
    /// `timedelta64[unit]`.
    const TIMEDELTA_DTYPE: &'static str;
    /// `.npy` descriptor of the datetime dtype, e.g. `"<M8[D]"`.
    const DATETIME_DESCR: &'static str;
    /// `.npy` descriptor of the timedelta dtype, e.g. `"<m8[D]"`.
    const TIMEDELTA_DESCR: &'static str;
}

macro_rules! units {
    ($($(#[$m:meta])* $t:ident = $code:expr, $name:literal, $plural:literal;)*) => {$(
        $(#[$m])*
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct $t;
        impl sealed::Sealed for $t {}
        impl TimeUnit for $t {
            const NAME: &'static str = $name;
            const CODE: u8 = $code;
            const PLURAL: &'static str = $plural;
            const DATETIME_DTYPE: &'static str = concat!("datetime64[", $name, "]");
            const TIMEDELTA_DTYPE: &'static str = concat!("timedelta64[", $name, "]");
            const DATETIME_DESCR: &'static str = concat!("<M8[", $name, "]");
            const TIMEDELTA_DESCR: &'static str = concat!("<m8[", $name, "]");
        }
    )*};
}
units! {
    /// Years, NumPy unit `Y`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::DateTime64::<np::datetime::Y>::parse("2024").unwrap().value(), 54);
    /// ```
    Y = 0, "Y", "years";
    /// Months, NumPy unit `M`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::DateTime64::<np::datetime::M>::parse("1971-03").unwrap().value(), 14);
    /// ```
    M = 1, "M", "months";
    /// Weeks, NumPy unit `W`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::TimeDelta64::<np::datetime::W>::new(2).to_string(), "2 weeks");
    /// ```
    W = 2, "W", "weeks";
    /// Days, NumPy unit `D`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::DateTime64::<np::datetime::D>::parse("1970-01-11").unwrap().value(), 10);
    /// ```
    D = 3, "D", "days";
    /// Hours, NumPy unit `h`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::DateTime64::<np::datetime::h>::parse("1970-01-01T05").unwrap().value(), 5);
    /// ```
    h = 4, "h", "hours";
    /// Minutes, NumPy unit `m`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::DateTime64::<np::datetime::m>::parse("1970-01-01T00:07").unwrap().value(), 7);
    /// ```
    m = 5, "m", "minutes";
    /// Seconds, NumPy unit `s`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::DateTime64::<np::datetime::s>::parse("1970-01-01T00:00:09").unwrap().value(), 9);
    /// ```
    s = 6, "s", "seconds";
    /// Milliseconds, NumPy unit `ms`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::DateTime64::<np::datetime::ms>::parse("1970-01-01T00:00:00.250").unwrap().value(), 250);
    /// ```
    ms = 7, "ms", "milliseconds";
    /// Microseconds, NumPy unit `us`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::DateTime64::<np::datetime::us>::parse("1970-01-01T00:00:00.000001").unwrap().value(), 1);
    /// ```
    us = 8, "us", "microseconds";
    /// Nanoseconds, NumPy unit `ns`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::DateTime64::<np::datetime::ns>::parse("1970-01-01T00:00:00.000000007").unwrap().value(), 7);
    /// ```
    ns = 9, "ns", "nanoseconds";
}
