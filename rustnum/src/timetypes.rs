//! Implementation of `datetime64` / `timedelta64`; the public surface lives in [`crate::datetime`].

use crate::datetime::TimeUnit;
use crate::{Array, Element, Error, Result};
use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use std::str::FromStr;

const NAT: i64 = i64::MIN;
/// Average nanoseconds per unit (years and months use the 400-year Gregorian averages, as NumPy
/// does when it converts time deltas).
const AVG_NS: [i128; 10] = [
    31_556_952_000_000_000,
    2_629_746_000_000_000,
    604_800_000_000_000,
    86_400_000_000_000,
    3_600_000_000_000,
    60_000_000_000,
    1_000_000_000,
    1_000_000,
    1_000,
    1,
];

const PLURALS: [&str; 10] = [
    "years",
    "months",
    "weeks",
    "days",
    "hours",
    "minutes",
    "seconds",
    "milliseconds",
    "microseconds",
    "nanoseconds",
];

/// `str(x)` of a datetime or timedelta scalar (`2024-01-01`, `5 days`, `NaT`), `None` for other types.
pub(crate) fn scalar_str<T: Element>(x: T) -> Option<String> {
    let (v, code, delta) = x.time_parts()?;
    Some(if v == NAT {
        "NaT".to_string()
    } else if delta {
        format!("{v} {}", PLURALS[code as usize])
    } else {
        format_iso(v, code)
    })
}

/// `astype` between time types and numbers without a lossy trip through `f64`.
pub(crate) fn astype_time<T: Element, U: Element>(a: &Array<T>) -> Option<Array<U>> {
    let from_time = T::default().time_parts().is_some();
    let to_time = U::from_time_parts(0, 3, false).is_some();
    if !from_time && !to_time {
        return None;
    }
    Some(a.par_map(|x| {
        if let Some((v, code, delta)) = x.time_parts() {
            return U::from_time_parts(v, code, delta).unwrap_or_else(|| U::from_i64_exact(v));
        }
        match x.to_i64_exact() {
            Some(v) => U::from_i64_exact(v),
            None => U::from_f64(x.to_f64()),
        }
    }))
}

fn floor_div(a: i128, b: i128) -> i128 {
    a.div_euclid(b) - if b < 0 && a.rem_euclid(b) != 0 { 1 } else { 0 }
}

/// Days since 1970-01-01 of a proleptic Gregorian date.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `(year, month, day)` of a day count since 1970-01-01.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
    }
}

/// Converts a time delta count between units (floor division, like NumPy).
fn convert_td(v: i64, from: u8, to: u8) -> i64 {
    if v == NAT || from == to {
        return v;
    }
    floor_div(v as i128 * AVG_NS[from as usize], AVG_NS[to as usize]) as i64
}

/// Converts a datetime count between units; years and months use the real calendar.
fn convert_dt(v: i64, from: u8, to: u8) -> i64 {
    if v == NAT || from == to {
        return v;
    }
    let (fc, tc) = (from <= 1, to <= 1);
    if fc && tc {
        return if from == 0 {
            v.wrapping_mul(12)
        } else {
            floor_div(v as i128, 12) as i64
        };
    }
    if fc {
        let days = if from == 0 {
            days_from_civil(1970 + v, 1, 1)
        } else {
            days_from_civil(
                1970 + floor_div(v as i128, 12) as i64,
                v.rem_euclid(12) + 1,
                1,
            )
        };
        return convert_td(days, 3, to);
    }
    if tc {
        let days = floor_div(v as i128 * AVG_NS[from as usize], AVG_NS[3]) as i64;
        let (y, m, _) = civil_from_days(days);
        return if to == 0 {
            y - 1970
        } else {
            (y - 1970) * 12 + m - 1
        };
    }
    convert_td(v, from, to)
}

/// Parses an ISO 8601 datetime; returns the value in its natural unit and that unit's code.
fn parse_iso(text: &str) -> Result<(i64, u8)> {
    let bad = || Error::Parse(format!("Error parsing datetime string \"{text}\""));
    let s = text.trim();
    if s.eq_ignore_ascii_case("nat") {
        return Ok((NAT, 3));
    }
    let (neg, rest) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let (date, time) = match rest.find(['T', ' ']) {
        Some(i) => (&rest[..i], Some(&rest[i + 1..])),
        None => (rest, None),
    };
    let mut parts = date.split('-');
    let year: i64 = parts.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
    let year = if neg { -year } else { year };
    let month: Option<i64> = parts
        .next()
        .map(|p| p.parse().map_err(|_| bad()))
        .transpose()?;
    let day: Option<i64> = parts
        .next()
        .map(|p| p.parse().map_err(|_| bad()))
        .transpose()?;
    if parts.next().is_some() {
        return Err(bad());
    }
    let (mo, d) = (month.unwrap_or(1), day.unwrap_or(1));
    if !(1..=12).contains(&mo) || d < 1 || d > days_in_month(year, mo) {
        return Err(bad());
    }
    if time.is_none() {
        return Ok(match (month, day) {
            (None, _) => (year - 1970, 0),
            (Some(_), None) => ((year - 1970) * 12 + mo - 1, 1),
            _ => (days_from_civil(year, mo, d), 3),
        });
    }
    if day.is_none() {
        return Err(bad());
    }
    let days = days_from_civil(year, mo, d);
    let time = time.unwrap();
    let (hms, frac) = match time.split_once('.') {
        Some((a, b)) => (a, Some(b)),
        None => (time, None),
    };
    let mut t = hms.split(':');
    let hour: i64 = t.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
    let minute: Option<i64> = t.next().map(|p| p.parse().map_err(|_| bad())).transpose()?;
    let second: Option<i64> = t.next().map(|p| p.parse().map_err(|_| bad())).transpose()?;
    if t.next().is_some() || !(0..24).contains(&hour) {
        return Err(bad());
    }
    if minute.is_some_and(|x| !(0..60).contains(&x))
        || second.is_some_and(|x| !(0..60).contains(&x))
    {
        return Err(bad());
    }
    if frac.is_some() && second.is_none() {
        return Err(bad());
    }
    let (h, mi) = (days * 24 + hour, minute);
    if mi.is_none() {
        return Ok((h, 4));
    }
    let m = h * 60 + mi.unwrap();
    if second.is_none() {
        return Ok((m, 5));
    }
    let sec = m * 60 + second.unwrap();
    match frac {
        None => Ok((sec, 6)),
        Some(f) => {
            if f.is_empty() || f.len() > 9 || !f.bytes().all(|b| b.is_ascii_digit()) {
                return Err(bad());
            }
            let (unit, width) = match f.len() {
                1..=3 => (7u8, 3usize),
                4..=6 => (8, 6),
                _ => (9, 9),
            };
            let padded = format!("{f:0<width$}");
            let fv: i64 = padded.parse().map_err(|_| bad())?;
            let scale = 10i64.pow(width as u32);
            Ok((sec * scale + fv, unit))
        }
    }
}

/// ISO 8601 text of a datetime count in unit `code` (`NaT` for missing values).
fn format_iso(v: i64, code: u8) -> String {
    if v == NAT {
        return "NaT".into();
    }
    let per_day = |u: u8| floor_div(AVG_NS[3], AVG_NS[u as usize]) as i64;
    match code {
        0 => format!("{:04}", 1970 + v),
        1 => {
            let y = 1970 + floor_div(v as i128, 12) as i64;
            format!("{:04}-{:02}", y, v.rem_euclid(12) + 1)
        }
        2 | 3 => {
            let days = if code == 2 { v.wrapping_mul(7) } else { v };
            let (y, m, d) = civil_from_days(days);
            format!("{y:04}-{m:02}-{d:02}")
        }
        _ => {
            let n = per_day(code);
            let days = floor_div(v as i128, n as i128) as i64;
            let rem = v.rem_euclid(n);
            let (y, mo, d) = civil_from_days(days);
            let date = format!("{y:04}-{mo:02}-{d:02}");
            let (h, mi, sec, sub) = match code {
                4 => (rem, 0, 0, 0),
                5 => (rem / 60, rem % 60, 0, 0),
                6 => (rem / 3600, rem / 60 % 60, rem % 60, 0),
                7 => (
                    rem / 3_600_000,
                    rem / 60_000 % 60,
                    rem / 1000 % 60,
                    rem % 1000,
                ),
                8 => (
                    rem / 3_600_000_000,
                    rem / 60_000_000 % 60,
                    rem / 1_000_000 % 60,
                    rem % 1_000_000,
                ),
                _ => (
                    rem / 3_600_000_000_000,
                    rem / 60_000_000_000 % 60,
                    rem / 1_000_000_000 % 60,
                    rem % 1_000_000_000,
                ),
            };
            match code {
                4 => format!("{date}T{h:02}"),
                5 => format!("{date}T{h:02}:{mi:02}"),
                6 => format!("{date}T{h:02}:{mi:02}:{sec:02}"),
                7 => format!("{date}T{h:02}:{mi:02}:{sec:02}.{sub:03}"),
                8 => format!("{date}T{h:02}:{mi:02}:{sec:02}.{sub:06}"),
                _ => format!("{date}T{h:02}:{mi:02}:{sec:02}.{sub:09}"),
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// DateTime64
// ---------------------------------------------------------------------------------------

/// A point in time with unit `U`, like NumPy `datetime64[U]`.
///
/// ```
/// use rustnum as np;
/// use np::datetime::D;
/// let d = np::DateTime64::<D>::parse("2024-02-29").unwrap();
/// assert_eq!(d.value(), 19782);
/// assert!(np::DateTime64::<D>::NAT.is_nat());
/// ```
#[derive(Clone, Copy)]
pub struct DateTime64<U: TimeUnit>(i64, PhantomData<U>);

impl<U: TimeUnit> DateTime64<U> {
    /// Not-a-time, like `np.datetime64('NaT')`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert!(np::DateTime64::<np::datetime::s>::NAT.is_nat());
    /// ```
    pub const NAT: Self = Self(NAT, PhantomData);

    /// Builds a value from its count of units since 1970-01-01, like `np.datetime64(n, 'D')`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::DateTime64::<np::datetime::D>::new(10).to_string(), "1970-01-11");
    /// ```
    pub const fn new(value: i64) -> Self {
        Self(value, PhantomData)
    }

    /// The raw count, like `a.astype(np.int64)`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::DateTime64::<np::datetime::D>::new(3).value(), 3);
    /// ```
    pub const fn value(self) -> i64 {
        self.0
    }

    /// True for `NaT`, like `np.isnat(x)`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert!(!np::DateTime64::<np::datetime::D>::new(0).is_nat());
    /// ```
    pub const fn is_nat(self) -> bool {
        self.0 == NAT
    }

    /// Parses an ISO 8601 string, like `np.datetime64(text, unit)`.
    ///
    /// `"NaT"` gives `NaT`. A string finer than the unit (for example seconds into
    /// `datetime64[D]`) is an error, as in NumPy; a coarser string is widened.
    ///
    /// ```
    /// use rustnum as np;
    /// use np::datetime::{D, ms};
    /// assert_eq!(np::DateTime64::<D>::parse("1970-01-02").unwrap().value(), 1);
    /// assert_eq!(np::DateTime64::<ms>::parse("1970-01-01T00:00:01.5").unwrap().value(), 1500);
    /// assert!(np::DateTime64::<D>::parse("2024-02-30").is_err());
    /// assert!(np::DateTime64::<D>::parse("2024-01-01T10:00").is_err());
    /// ```
    pub fn parse(text: &str) -> Result<Self> {
        let (v, natural) = parse_iso(text)?;
        if v == NAT {
            return Ok(Self::NAT);
        }
        if natural > U::CODE {
            return Err(Error::Parse(format!(
                "Cannot parse \"{text}\" as unit '{}' using casting rule 'same_kind'",
                U::NAME
            )));
        }
        Ok(Self::new(convert_dt(v, natural, U::CODE)))
    }

    /// Converts to another unit, like `x.astype('datetime64[V]')`.
    ///
    /// Years and months use the calendar; coarser targets round toward minus infinity.
    ///
    /// ```
    /// use rustnum as np;
    /// use np::datetime::{D, M, Y};
    /// let d = np::DateTime64::<D>::parse("2024-03-05").unwrap();
    /// assert_eq!(d.cast::<M>().to_string(), "2024-03");
    /// assert_eq!(d.cast::<Y>().value(), 54);
    /// ```
    pub fn cast<V: TimeUnit>(self) -> DateTime64<V> {
        DateTime64::new(convert_dt(self.0, U::CODE, V::CODE))
    }
}

impl<U: TimeUnit> Default for DateTime64<U> {
    fn default() -> Self {
        Self::new(0)
    }
}
impl<U: TimeUnit> PartialEq for DateTime64<U> {
    fn eq(&self, o: &Self) -> bool {
        self.0 != NAT && o.0 != NAT && self.0 == o.0
    }
}
impl<U: TimeUnit> PartialOrd for DateTime64<U> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        if self.0 == NAT || o.0 == NAT {
            None
        } else {
            Some(self.0.cmp(&o.0))
        }
    }
}
impl<U: TimeUnit> fmt::Display for DateTime64<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&format_iso(self.0, U::CODE))
    }
}
impl<U: TimeUnit> fmt::Debug for DateTime64<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "numpy.datetime64('{}')", format_iso(self.0, U::CODE))
    }
}
impl<U: TimeUnit> FromStr for DateTime64<U> {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

// ---------------------------------------------------------------------------------------
// TimeDelta64
// ---------------------------------------------------------------------------------------

/// A duration with unit `U`, like NumPy `timedelta64[U]`.
///
/// ```
/// use rustnum as np;
/// use np::datetime::D;
/// let t = np::TimeDelta64::<D>::new(7);
/// assert_eq!(t.to_string(), "7 days");
/// assert_eq!((t * 2).value(), 14);
/// ```
#[derive(Clone, Copy)]
pub struct TimeDelta64<U: TimeUnit>(i64, PhantomData<U>);

impl<U: TimeUnit> TimeDelta64<U> {
    /// Not-a-time, like `np.timedelta64('NaT')`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert!(np::TimeDelta64::<np::datetime::D>::NAT.is_nat());
    /// ```
    pub const NAT: Self = Self(NAT, PhantomData);

    /// Builds a duration of `value` units, like `np.timedelta64(value, 'D')`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::TimeDelta64::<np::datetime::h>::new(5).value(), 5);
    /// ```
    pub const fn new(value: i64) -> Self {
        Self(value, PhantomData)
    }

    /// The raw count of units.
    ///
    /// ```
    /// use rustnum as np;
    /// assert_eq!(np::TimeDelta64::<np::datetime::h>::new(5).value(), 5);
    /// ```
    pub const fn value(self) -> i64 {
        self.0
    }

    /// True for `NaT`.
    ///
    /// ```
    /// use rustnum as np;
    /// assert!(np::TimeDelta64::<np::datetime::h>::NAT.is_nat());
    /// ```
    pub const fn is_nat(self) -> bool {
        self.0 == NAT
    }

    /// Converts to another unit, like `x.astype('timedelta64[V]')`.
    ///
    /// A year counts as 365.2425 days and a month as 30.436875 days; results round toward
    /// minus infinity.
    ///
    /// ```
    /// use rustnum as np;
    /// use np::datetime::{D, Y, h, m};
    /// assert_eq!(np::TimeDelta64::<Y>::new(10).cast::<D>().value(), 3652);
    /// assert_eq!(np::TimeDelta64::<h>::new(90).cast::<m>().value(), 5400);
    /// ```
    pub fn cast<V: TimeUnit>(self) -> TimeDelta64<V> {
        TimeDelta64::new(convert_td(self.0, U::CODE, V::CODE))
    }

    /// Floor division of durations, like `td // td`. `NaT` or a zero divisor gives 0.
    ///
    /// ```
    /// use rustnum as np;
    /// let (a, b) = (np::TimeDelta64::<np::datetime::D>::new(-7), np::TimeDelta64::new(2));
    /// assert_eq!(a.floor_div(b), -4);
    /// ```
    pub fn floor_div(self, o: Self) -> i64 {
        if self.0 == NAT || o.0 == NAT || o.0 == 0 {
            0
        } else {
            floor_div(self.0 as i128, o.0 as i128) as i64
        }
    }
}

impl<U: TimeUnit> Default for TimeDelta64<U> {
    fn default() -> Self {
        Self::new(0)
    }
}
impl<U: TimeUnit> PartialEq for TimeDelta64<U> {
    fn eq(&self, o: &Self) -> bool {
        self.0 != NAT && o.0 != NAT && self.0 == o.0
    }
}
impl<U: TimeUnit> PartialOrd for TimeDelta64<U> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        if self.0 == NAT || o.0 == NAT {
            None
        } else {
            Some(self.0.cmp(&o.0))
        }
    }
}
impl<U: TimeUnit> fmt::Display for TimeDelta64<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == NAT {
            f.pad("NaT")
        } else {
            f.pad(&format!("{} {}", self.0, U::PLURAL))
        }
    }
}
impl<U: TimeUnit> fmt::Debug for TimeDelta64<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == NAT {
            write!(f, "numpy.timedelta64('NaT','{}')", U::NAME)
        } else {
            write!(f, "numpy.timedelta64({},'{}')", self.0, U::NAME)
        }
    }
}

// ---------------------------------------------------------------------------------------
// Scalar arithmetic
// ---------------------------------------------------------------------------------------

fn nat_or(a: i64, b: i64, f: impl Fn(i64, i64) -> i64) -> i64 {
    if a == NAT || b == NAT { NAT } else { f(a, b) }
}

impl<U: TimeUnit> Sub for DateTime64<U> {
    type Output = TimeDelta64<U>;
    fn sub(self, o: Self) -> TimeDelta64<U> {
        TimeDelta64::new(nat_or(self.0, o.0, i64::wrapping_sub))
    }
}
impl<U: TimeUnit> Add<TimeDelta64<U>> for DateTime64<U> {
    type Output = DateTime64<U>;
    fn add(self, o: TimeDelta64<U>) -> DateTime64<U> {
        DateTime64::new(nat_or(self.0, o.0, i64::wrapping_add))
    }
}
impl<U: TimeUnit> Add<DateTime64<U>> for TimeDelta64<U> {
    type Output = DateTime64<U>;
    fn add(self, o: DateTime64<U>) -> DateTime64<U> {
        o + self
    }
}
impl<U: TimeUnit> Sub<TimeDelta64<U>> for DateTime64<U> {
    type Output = DateTime64<U>;
    fn sub(self, o: TimeDelta64<U>) -> DateTime64<U> {
        DateTime64::new(nat_or(self.0, o.0, i64::wrapping_sub))
    }
}
impl<U: TimeUnit> Add for TimeDelta64<U> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(nat_or(self.0, o.0, i64::wrapping_add))
    }
}
impl<U: TimeUnit> Sub for TimeDelta64<U> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(nat_or(self.0, o.0, i64::wrapping_sub))
    }
}
impl<U: TimeUnit> Neg for TimeDelta64<U> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(if self.0 == NAT {
            NAT
        } else {
            self.0.wrapping_neg()
        })
    }
}
impl<U: TimeUnit> Mul<i64> for TimeDelta64<U> {
    type Output = Self;
    fn mul(self, k: i64) -> Self {
        Self::new(if self.0 == NAT {
            NAT
        } else {
            self.0.wrapping_mul(k)
        })
    }
}
impl<U: TimeUnit> Mul<TimeDelta64<U>> for i64 {
    type Output = TimeDelta64<U>;
    fn mul(self, t: TimeDelta64<U>) -> TimeDelta64<U> {
        t * self
    }
}
impl<U: TimeUnit> Mul<f64> for TimeDelta64<U> {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        let r = self.0 as f64 * k;
        Self::new(if self.0 == NAT || r.is_nan() {
            NAT
        } else {
            r as i64
        })
    }
}
impl<U: TimeUnit> Mul<TimeDelta64<U>> for f64 {
    type Output = TimeDelta64<U>;
    fn mul(self, t: TimeDelta64<U>) -> TimeDelta64<U> {
        t * self
    }
}
impl<U: TimeUnit> Div<i64> for TimeDelta64<U> {
    type Output = Self;
    fn div(self, k: i64) -> Self {
        Self::new(if self.0 == NAT || k == 0 {
            NAT
        } else {
            self.0.wrapping_div(k)
        })
    }
}
impl<U: TimeUnit> Div<f64> for TimeDelta64<U> {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        let r = self.0 as f64 / k;
        Self::new(if self.0 == NAT || k == 0.0 || r.is_nan() {
            NAT
        } else {
            r as i64
        })
    }
}
impl<U: TimeUnit> Div for TimeDelta64<U> {
    type Output = f64;
    fn div(self, o: Self) -> f64 {
        if self.0 == NAT || o.0 == NAT {
            f64::NAN
        } else {
            self.0 as f64 / o.0 as f64
        }
    }
}
impl<U: TimeUnit> Rem for TimeDelta64<U> {
    type Output = Self;
    fn rem(self, o: Self) -> Self {
        if self.0 == NAT || o.0 == NAT || o.0 == 0 {
            return Self::NAT;
        }
        // Python-style modulo: the result takes the divisor's sign.
        let r = (self.0 as i128).rem_euclid((o.0 as i128).abs());
        let r = if o.0 < 0 && r != 0 {
            r + o.0 as i128
        } else {
            r
        };
        Self::new(r as i64)
    }
}

// ---------------------------------------------------------------------------------------
// Element impls
// ---------------------------------------------------------------------------------------

impl<U: TimeUnit> Element for DateTime64<U> {
    const DTYPE: &'static str = U::DATETIME_DTYPE;
    fn to_f64(self) -> f64 {
        self.0 as f64
    }
    fn from_f64(v: f64) -> Self {
        Self::new(if v.is_nan() { NAT } else { v as i64 })
    }
    fn time_parts(self) -> Option<(i64, u8, bool)> {
        Some((self.0, U::CODE, false))
    }
    fn from_time_parts(v: i64, code: u8, is_delta: bool) -> Option<Self> {
        Some(Self::new(if is_delta {
            v
        } else {
            convert_dt(v, code, U::CODE)
        }))
    }
    fn to_i64_exact(self) -> Option<i64> {
        Some(self.0)
    }
    fn from_i64_exact(v: i64) -> Self {
        Self::new(v)
    }
    fn text(self) -> Option<String> {
        Some(format!("'{}'", format_iso(self.0, U::CODE)))
    }
    fn npy_descr() -> Option<String> {
        Some(U::DATETIME_DESCR.to_string())
    }
    fn npy_itemsize() -> usize {
        8
    }
    fn npy_write(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0.to_le_bytes());
    }
    fn npy_read(b: &[u8]) -> Self {
        Self::new(i64::from_le_bytes(b[..8].try_into().unwrap()))
    }
}

impl<U: TimeUnit> Element for TimeDelta64<U> {
    const DTYPE: &'static str = U::TIMEDELTA_DTYPE;
    fn to_f64(self) -> f64 {
        self.0 as f64
    }
    fn from_f64(v: f64) -> Self {
        Self::new(if v.is_nan() { NAT } else { v as i64 })
    }
    fn time_parts(self) -> Option<(i64, u8, bool)> {
        Some((self.0, U::CODE, true))
    }
    fn from_time_parts(v: i64, code: u8, is_delta: bool) -> Option<Self> {
        Some(Self::new(if is_delta {
            convert_td(v, code, U::CODE)
        } else {
            v
        }))
    }
    fn to_i64_exact(self) -> Option<i64> {
        Some(self.0)
    }
    fn from_i64_exact(v: i64) -> Self {
        Self::new(v)
    }
    fn text(self) -> Option<String> {
        Some(if self.0 == NAT {
            "'NaT'".to_string()
        } else {
            self.0.to_string()
        })
    }
    fn npy_descr() -> Option<String> {
        Some(U::TIMEDELTA_DESCR.to_string())
    }
    fn npy_itemsize() -> usize {
        8
    }
    fn npy_write(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0.to_le_bytes());
    }
    fn npy_read(b: &[u8]) -> Self {
        Self::new(i64::from_le_bytes(b[..8].try_into().unwrap()))
    }
}

// ---------------------------------------------------------------------------------------
// Array operators and methods
// ---------------------------------------------------------------------------------------

macro_rules! array_binop {
    ($Tr:ident, $m:ident, $L:ident, $R:ident => $O:ty, $f:expr) => {
        impl<U: TimeUnit> $Tr<&Array<$R<U>>> for &Array<$L<U>> {
            type Output = Array<$O>;
            fn $m(self, r: &Array<$R<U>>) -> Array<$O> {
                self.par_zip(r, $f)
            }
        }
        impl<U: TimeUnit> $Tr<Array<$R<U>>> for Array<$L<U>> {
            type Output = Array<$O>;
            fn $m(self, r: Array<$R<U>>) -> Array<$O> {
                (&self).par_zip(&r, $f)
            }
        }
        impl<U: TimeUnit> $Tr<&Array<$R<U>>> for Array<$L<U>> {
            type Output = Array<$O>;
            fn $m(self, r: &Array<$R<U>>) -> Array<$O> {
                (&self).par_zip(r, $f)
            }
        }
        impl<U: TimeUnit> $Tr<Array<$R<U>>> for &Array<$L<U>> {
            type Output = Array<$O>;
            fn $m(self, r: Array<$R<U>>) -> Array<$O> {
                self.par_zip(&r, $f)
            }
        }
        impl<U: TimeUnit> $Tr<$R<U>> for &Array<$L<U>> {
            type Output = Array<$O>;
            fn $m(self, r: $R<U>) -> Array<$O> {
                self.par_zip(r, $f)
            }
        }
    };
}
array_binop!(Sub, sub, DateTime64, DateTime64 => TimeDelta64<U>, |a, b| a - b);
array_binop!(Add, add, DateTime64, TimeDelta64 => DateTime64<U>, |a, b| a + b);
array_binop!(Sub, sub, DateTime64, TimeDelta64 => DateTime64<U>, |a, b| a - b);
array_binop!(Add, add, TimeDelta64, DateTime64 => DateTime64<U>, |a, b| a + b);
array_binop!(Add, add, TimeDelta64, TimeDelta64 => TimeDelta64<U>, |a, b| a + b);
array_binop!(Sub, sub, TimeDelta64, TimeDelta64 => TimeDelta64<U>, |a, b| a - b);
array_binop!(Div, div, TimeDelta64, TimeDelta64 => f64, |a, b| a / b);
array_binop!(Rem, rem, TimeDelta64, TimeDelta64 => TimeDelta64<U>, |a, b| a % b);

impl<U: TimeUnit> Mul<i64> for &Array<TimeDelta64<U>> {
    type Output = Array<TimeDelta64<U>>;
    fn mul(self, k: i64) -> Array<TimeDelta64<U>> {
        self.par_map(move |t| t * k)
    }
}
impl<U: TimeUnit> Mul<f64> for &Array<TimeDelta64<U>> {
    type Output = Array<TimeDelta64<U>>;
    fn mul(self, k: f64) -> Array<TimeDelta64<U>> {
        self.par_map(move |t| t * k)
    }
}
impl<U: TimeUnit> Div<i64> for &Array<TimeDelta64<U>> {
    type Output = Array<TimeDelta64<U>>;
    fn div(self, k: i64) -> Array<TimeDelta64<U>> {
        self.par_map(move |t| t / k)
    }
}
impl<U: TimeUnit> Div<f64> for &Array<TimeDelta64<U>> {
    type Output = Array<TimeDelta64<U>>;
    fn div(self, k: f64) -> Array<TimeDelta64<U>> {
        self.par_map(move |t| t / k)
    }
}
impl<U: TimeUnit> Neg for &Array<TimeDelta64<U>> {
    type Output = Array<TimeDelta64<U>>;
    fn neg(self) -> Array<TimeDelta64<U>> {
        self.par_map(|t| -t)
    }
}

impl<U: TimeUnit> Array<TimeDelta64<U>> {
    /// Sum of all durations; any `NaT` makes the result `NaT`, like `np.sum`.
    ///
    /// ```
    /// use rustnum as np;
    /// use np::datetime::D;
    /// let a = np::Array::from(vec![np::TimeDelta64::<D>::new(1), np::TimeDelta64::new(2)]);
    /// assert_eq!(a.sum().value(), 3);
    /// ```
    pub fn sum(&self) -> TimeDelta64<U> {
        self.iter().fold(TimeDelta64::new(0), |acc, t| acc + t)
    }

    /// Element-wise floor division of durations into whole counts, like `a // b` for `timedelta64`.
    ///
    /// `NaT` or a zero divisor gives 0, as in NumPy.
    ///
    /// ```
    /// use rustnum as np;
    /// use np::datetime::D;
    /// type T = np::TimeDelta64<D>;
    /// let a = np::Array::from(vec![T::new(7), T::new(-7)]);
    /// assert_eq!(a.floor_div(T::new(2)).to_vec(), [3, -4]);
    /// ```
    pub fn floor_div(&self, other: impl crate::IntoArray<TimeDelta64<U>>) -> Array<i64> {
        self.par_zip(other, |a, b| a.floor_div(b))
    }

    /// Mean duration (sum divided by the count, truncated), like `np.mean`.
    ///
    /// ```
    /// use rustnum as np;
    /// use np::datetime::D;
    /// let a = np::Array::from(vec![np::TimeDelta64::<D>::new(1), np::TimeDelta64::new(2)]);
    /// assert_eq!(a.mean().value(), 1);
    /// ```
    pub fn mean(&self) -> TimeDelta64<U> {
        self.sum() / self.size() as i64
    }

    /// Running sum, like `np.cumsum`.
    ///
    /// ```
    /// use rustnum as np;
    /// use np::datetime::D;
    /// let a = np::Array::from(vec![np::TimeDelta64::<D>::new(1), np::TimeDelta64::new(2)]);
    /// assert_eq!(a.cumsum().iter().map(|t| t.value()).collect::<Vec<_>>(), [1, 3]);
    /// ```
    pub fn cumsum(&self) -> Array<TimeDelta64<U>> {
        let mut acc = TimeDelta64::new(0);
        let v: Vec<_> = self
            .iter()
            .map(|t| {
                acc = acc + t;
                acc
            })
            .collect();
        Array::from_vec(v, self.shape())
    }

    /// Evenly spaced durations in `[start, stop)`, like `np.arange(start, stop, step)` with
    /// `timedelta64` arguments.
    ///
    /// ```
    /// use rustnum as np;
    /// use np::datetime::h;
    /// type T = np::TimeDelta64<h>;
    /// let a = np::Array::<T>::arange(T::new(0), T::new(5), T::new(2));
    /// assert_eq!(a.iter().map(|t| t.value()).collect::<Vec<_>>(), [0, 2, 4]);
    /// ```
    pub fn arange(start: TimeDelta64<U>, stop: TimeDelta64<U>, step: TimeDelta64<U>) -> Self {
        let v = range_values(start.0, stop.0, step.0);
        Array::from(v.into_iter().map(TimeDelta64::new).collect::<Vec<_>>())
    }
}

fn range_values(start: i64, stop: i64, step: i64) -> Vec<i64> {
    assert!(
        start != NAT && stop != NAT && step != NAT,
        "arange: cannot compute length"
    );
    assert!(step != 0, "arange: step must not be zero");
    let (a, b, s) = (start as i128, stop as i128, step as i128);
    let n = if s > 0 {
        (b - a + s - 1).div_euclid(s)
    } else {
        (a - b - s - 1).div_euclid(-s)
    };
    (0..n.max(0)).map(|i| (a + i * s) as i64).collect()
}

impl<U: TimeUnit> Array<DateTime64<U>> {
    /// Evenly spaced dates in `[start, stop)`, like `np.arange(start, stop, step)` with
    /// `datetime64` arguments.
    ///
    /// ```
    /// use rustnum as np;
    /// use np::datetime::D;
    /// type T = np::DateTime64<D>;
    /// let a = np::Array::<T>::arange(
    ///     T::parse("2024-01-01").unwrap(),
    ///     T::parse("2024-01-10").unwrap(),
    ///     np::TimeDelta64::new(3),
    /// );
    /// assert_eq!(np::datetime_as_string(&a), ["2024-01-01", "2024-01-04", "2024-01-07"]);
    /// ```
    pub fn arange(start: DateTime64<U>, stop: DateTime64<U>, step: TimeDelta64<U>) -> Self {
        let v = range_values(start.0, stop.0, step.0);
        Array::from(v.into_iter().map(DateTime64::new).collect::<Vec<_>>())
    }
}

/// ISO 8601 strings of a datetime array in C order, like `np.datetime_as_string(a)`.
///
/// Precision follows the unit (`2024-03-05` for days, `...T10:20:30.500` for milliseconds);
/// `NaT` prints as `"NaT"`. NumPy returns a string array; here the flattened strings come back
/// as a `Vec` because `String` is not an array element.
///
/// ```
/// use rustnum as np;
/// use np::datetime::ms;
/// let a = np::Array::from(vec![np::DateTime64::<ms>::parse("2024-03-05T10:20:30.5").unwrap()]);
/// assert_eq!(np::datetime_as_string(&a), ["2024-03-05T10:20:30.500"]);
/// ```
pub fn datetime_as_string<U: TimeUnit>(a: &Array<DateTime64<U>>) -> Vec<String> {
    a.iter().map(|d| format_iso(d.0, U::CODE)).collect()
}

/// True where a datetime or timedelta is `NaT`, like `np.isnat(a)`.
///
/// ```
/// use rustnum as np;
/// use np::datetime::D;
/// let a = np::Array::from(vec![np::DateTime64::<D>::new(0), np::DateTime64::NAT]);
/// assert_eq!(np::isnat(&a).to_vec(), [false, true]);
/// ```
pub fn isnat<T: Element>(a: &Array<T>) -> Array<bool> {
    a.map(|x| x.time_parts().is_some_and(|(v, _, _)| v == NAT))
}
