//! Structured (record) dtypes, like NumPy `dtype=[('x', 'f8'), ('y', 'f8')]`.
//!
//! The declarative [`record!`](crate::record) macro defines a plain `Copy` struct and makes it an
//! array element. `Array<R>` then offers NumPy's field access (`a['x']`, `a['x'] = v`) and
//! sorting by field names (`np.sort(a, order=...)`), and `.npy` files store the structured
//! `descr` (`[('x', '<f8'), ('y', '<f8')]`) so NumPy reads and writes the same files.

use crate::io::npy::NpyElement;
use crate::{Array, Element, IntoArray};
use std::any::Any;
use std::any::TypeId;
use std::cmp::Ordering;

/// A structured element type declared with [`record!`](crate::record).
///
/// ```
/// use rustnum as np;
/// np::record! { pub struct Point { x: f64, y: f64 } }
/// assert_eq!(<Point as np::Record>::FIELDS, ["x", "y"]);
/// assert_eq!(<Point as np::Record>::descr(), "[('x', '<f8'), ('y', '<f8')]");
/// ```
pub trait Record: Element {
    /// Field names in declaration order.
    const FIELDS: &'static [&'static str];
    /// NumPy `descr` of the dtype, e.g. `[('x', '<f8'), ('y', '<f8')]`.
    /// ```
    /// use rustnum as np;
    /// np::record! { pub struct P { a: i32 } }
    /// assert_eq!(<P as np::Record>::descr(), "[('a', '<i4')]");
    /// ```
    fn descr() -> String;
    /// A reference to the named field, `None` for an unknown name.
    /// ```
    /// use rustnum as np;
    /// np::record! { pub struct P { a: i32 } }
    /// let p = P { a: 3 };
    /// let v = np::Record::field_any(&p, "a").unwrap().downcast_ref::<i32>();
    /// assert_eq!(v, Some(&3));
    /// ```
    fn field_any(&self, name: &str) -> Option<&dyn Any>;
    /// Overwrites a field from a value of the field's exact type; `false` if the name or type
    /// does not match.
    /// ```
    /// use rustnum as np;
    /// np::record! { pub struct P { a: i32 } }
    /// let mut p = P { a: 3 };
    /// assert!(np::Record::set_field_any(&mut p, "a", &7i32));
    /// assert_eq!(p.a, 7);
    /// ```
    fn set_field_any(&mut self, name: &str, value: &dyn Any) -> bool;
    /// Compares one field of two records (NaN sorts last); `None` for an unknown name.
    /// ```
    /// use rustnum as np;
    /// np::record! { pub struct P { a: i32 } }
    /// let o = np::Record::cmp_field(&P { a: 1 }, &P { a: 2 }, "a");
    /// assert_eq!(o, Some(std::cmp::Ordering::Less));
    /// ```
    fn cmp_field(&self, other: &Self, name: &str) -> Option<Ordering>;
}

/// Text of one scalar the way NumPy prints it inside a structured array.
#[doc(hidden)]
pub fn __element_text<T: Element>(x: T) -> String {
    {
        let o = crate::get_printoptions();
        crate::display::format_values(&[x], &o)
            .pop()
            .unwrap_or_default()
            .trim()
            .to_string()
    }
}

/// Aligned text of a column of scalars (one field of many records).
#[doc(hidden)]
pub fn __column_text<T: Element>(values: &[T]) -> Vec<String> {
    crate::display::format_values(values, &crate::get_printoptions())
}

/// `.npy` descriptor of any element type, including the seven built-ins.
fn element_descr<T: Element>() -> Option<String> {
    macro_rules! arm {
        ($($t:ty),*) => {$(
            if TypeId::of::<T>() == TypeId::of::<$t>() {
                return Some(<$t as NpyElement>::DESCR.to_string());
            }
        )*};
    }
    arm!(f64, f32, i64, i32, u8, bool, usize);
    T::npy_descr()
}

/// Bytes per element of any `.npy` element type, including the seven built-ins.
fn element_itemsize<T: Element>() -> usize {
    macro_rules! arm {
        ($($t:ty),*) => {$(
            if TypeId::of::<T>() == TypeId::of::<$t>() {
                return <$t as NpyElement>::BYTE_SIZE;
            }
        )*};
    }
    arm!(f64, f32, i64, i32, u8, bool, usize);
    T::npy_itemsize()
}

/// Appends the `.npy` bytes of any element type, including the seven built-ins.
fn element_write<T: Element>(x: T, out: &mut Vec<u8>) {
    macro_rules! arm {
        ($($t:ty),*) => {$(
            if let Some(v) = (&x as &dyn Any).downcast_ref::<$t>() {
                NpyElement::write_le_bytes(*v, out);
                return;
            }
        )*};
    }
    arm!(f64, f32, i64, i32, u8, bool, usize);
    x.npy_write(out);
}

/// Decodes any element type, including the seven built-ins, from `.npy` bytes.
fn element_read<T: Element>(bytes: &[u8]) -> T {
    macro_rules! arm {
        ($($t:ty),*) => {$(
            if TypeId::of::<T>() == TypeId::of::<$t>() {
                let v = <$t as NpyElement>::from_le_bytes(bytes);
                return *(&v as &dyn Any).downcast_ref::<T>().expect("type id was checked");
            }
        )*};
    }
    arm!(f64, f32, i64, i32, u8, bool, usize);
    T::npy_read(bytes)
}

/// `.npy` descriptor of a field type (works for the built-in numbers too).
#[doc(hidden)]
pub fn __npy_descr<T: Element>() -> Option<String> {
    element_descr::<T>()
}

/// Bytes of a field type in a `.npy` record.
#[doc(hidden)]
pub fn __npy_itemsize<T: Element>() -> usize {
    element_itemsize::<T>()
}

/// Appends the `.npy` bytes of a field value.
#[doc(hidden)]
pub fn __npy_write<T: Element>(x: T, out: &mut Vec<u8>) {
    element_write(x, out)
}

/// Decodes a field value from `.npy` bytes.
#[doc(hidden)]
pub fn __npy_read<T: Element>(bytes: &[u8]) -> T {
    element_read::<T>(bytes)
}

/// NaN-last comparison used by `order=` sorting.
#[doc(hidden)]
pub fn __compare<T: Element>(a: &T, b: &T) -> Ordering {
    crate::manip::compare(a, b)
}

/// Defines a structured element type, like `np.dtype([('x', 'f8'), ('y', 'f8')])`.
///
/// ```
/// use rustnum as np;
/// np::record! {
///     /// A 2-d point.
///     pub struct Point { x: f64, y: f64 }
/// }
/// let a = np::Array::from(vec![Point { x: 3.0, y: 1.0 }, Point { x: 1.0, y: 2.0 }]);
/// assert_eq!(a.field::<f64>("x").to_vec(), [3.0, 1.0]);
/// assert_eq!(a.argsort_order(&["x"]).to_vec(), [1, 0]);
/// assert_eq!(a.to_string(), "[(3., 1.) (1., 2.)]");
/// ```
///
/// Field types must be array elements with a `.npy` dtype (`f32 f64 i32 i64 u8 usize bool`,
/// complex, `f16`, datetimes, other records). The struct derives `Clone`, `Copy`, `Debug`,
/// `Default` and `PartialEq`; ordering is lexicographic in field order with NaN fields last,
/// like NumPy's `sort` on structured arrays (so `partial_cmp` never returns `None`).
#[macro_export]
macro_rules! record {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident { $($(#[$fmeta:meta])* $fvis:vis $field:ident : $fty:ty),+ $(,)? }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        $vis struct $name { $($(#[$fmeta])* $fvis $field: $fty),+ }

        impl ::core::cmp::PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> ::core::option::Option<::core::cmp::Ordering> {
                // Field by field with NaN last (NaN ties with NaN), like NumPy's structured sort,
                // so a NaN in one field never makes the whole record unordered.
                $(
                    match $crate::__compare(&self.$field, &other.$field) {
                        ::core::cmp::Ordering::Equal => {}
                        o => return ::core::option::Option::Some(o),
                    }
                )+
                ::core::option::Option::Some(::core::cmp::Ordering::Equal)
            }
        }

        impl $crate::Element for $name {
            const DTYPE: &'static str = concat!("record:", stringify!($name));
            fn to_f64(self) -> f64 {
                panic!("cannot convert a structured value to float")
            }
            fn from_f64(_: f64) -> Self {
                panic!("cannot convert a float to a structured value")
            }
            fn npy_descr() -> ::core::option::Option<::std::string::String> {
                ::core::option::Option::Some(<Self as $crate::Record>::descr())
            }
            fn npy_itemsize() -> usize {
                0 $(+ $crate::__npy_itemsize::<$fty>())+
            }
            fn npy_write(self, out: &mut ::std::vec::Vec<u8>) {
                $($crate::__npy_write(self.$field, out);)+
            }
            fn npy_read(bytes: &[u8]) -> Self {
                let mut _offset = 0usize;
                $(
                    let $field = $crate::__npy_read::<$fty>(&bytes[_offset..]);
                    _offset += $crate::__npy_itemsize::<$fty>();
                )+
                Self { $($field),+ }
            }
            fn text(self) -> ::core::option::Option<::std::string::String> {
                let parts: ::std::vec::Vec<::std::string::String> =
                    ::std::vec![$($crate::__element_text(self.$field)),+];
                ::core::option::Option::Some(::std::format!("({})", parts.join(", ")))
            }
            fn text_all(values: &[Self]) -> ::core::option::Option<::std::vec::Vec<::std::string::String>> {
                let columns: ::std::vec::Vec<::std::vec::Vec<::std::string::String>> = ::std::vec![
                    $($crate::__column_text(&values.iter().map(|v| v.$field).collect::<::std::vec::Vec<$fty>>())),+
                ];
                ::core::option::Option::Some(
                    (0..values.len())
                        .map(|i| {
                            let row: ::std::vec::Vec<&str> = columns.iter().map(|c| c[i].as_str()).collect();
                            ::std::format!("({})", row.join(", "))
                        })
                        .collect(),
                )
            }
        }

        impl $crate::Record for $name {
            const FIELDS: &'static [&'static str] = &[$(stringify!($field)),+];
            fn descr() -> ::std::string::String {
                let parts: ::std::vec::Vec<::std::string::String> = ::std::vec![$({
                    let d = $crate::__npy_descr::<$fty>()
                        .expect("record field type has no .npy dtype");
                    if d.starts_with('[') {
                        ::std::format!("('{}', {})", stringify!($field), d)
                    } else {
                        ::std::format!("('{}', '{}')", stringify!($field), d)
                    }
                }),+];
                ::std::format!("[{}]", parts.join(", "))
            }
            fn field_any(&self, name: &str) -> ::core::option::Option<&dyn ::core::any::Any> {
                $(if name == stringify!($field) {
                    return ::core::option::Option::Some(&self.$field);
                })+
                ::core::option::Option::None
            }
            fn set_field_any(&mut self, name: &str, value: &dyn ::core::any::Any) -> bool {
                $(if name == stringify!($field) {
                    return match value.downcast_ref::<$fty>() {
                        ::core::option::Option::Some(v) => {
                            self.$field = *v;
                            true
                        }
                        ::core::option::Option::None => false,
                    };
                })+
                false
            }
            fn cmp_field(&self, other: &Self, name: &str) -> ::core::option::Option<::core::cmp::Ordering> {
                $(if name == stringify!($field) {
                    return ::core::option::Option::Some($crate::__compare(&self.$field, &other.$field));
                })+
                ::core::option::Option::None
            }
        }
    };
}

fn unknown_field<R: Record>(name: &str) -> ! {
    panic!(
        "no field of name {name}; fields are {:?}",
        <R as Record>::FIELDS
    )
}

impl<R: Record> Array<R> {
    /// One field as an array, like `a['x']`.
    ///
    /// Panics for an unknown field name or when `F` is not the field's type.
    ///
    /// ```
    /// use rustnum as np;
    /// np::record! { pub struct P { id: i32, w: f64 } }
    /// let a = np::Array::from(vec![P { id: 1, w: 0.5 }, P { id: 2, w: 1.5 }]);
    /// assert_eq!(a.field::<i32>("id").to_vec(), [1, 2]);
    /// assert_eq!(a.field::<f64>("w").to_vec(), [0.5, 1.5]);
    /// ```
    pub fn field<F: Element>(&self, name: &str) -> Array<F> {
        let first = R::default();
        match first.field_any(name) {
            None => unknown_field::<R>(name),
            Some(v) => assert!(
                v.is::<F>(),
                "field '{name}' has a different dtype than {}",
                F::DTYPE
            ),
        }
        self.map(|r| {
            *r.field_any(name)
                .and_then(|v| v.downcast_ref::<F>())
                .expect("field type was checked")
        })
    }

    /// Assigns a whole field, like `a['x'] = values` (values broadcast to the array's shape).
    ///
    /// ```
    /// use rustnum as np;
    /// np::record! { pub struct P { id: i32, w: f64 } }
    /// let mut a = np::Array::from(vec![P::default(), P::default()]);
    /// a.set_field("w", np::array![2.5, 3.5]);
    /// assert_eq!(a.field::<f64>("w").to_vec(), [2.5, 3.5]);
    /// ```
    pub fn set_field<F: Element>(&mut self, name: &str, values: impl IntoArray<F>) {
        let _ = self.field_check::<F>(name);
        let values = values.into_array().broadcast_to(self.shape()).to_vec();
        let mut flat = self.to_vec();
        for (r, v) in flat.iter_mut().zip(values) {
            assert!(
                r.set_field_any(name, &v),
                "field type mismatch for '{name}'"
            );
        }
        *self = Array::from_vec(flat, self.shape());
    }

    fn field_check<F: Element>(&self, name: &str) -> bool {
        match R::default().field_any(name) {
            None => unknown_field::<R>(name),
            Some(v) => {
                assert!(
                    v.is::<F>(),
                    "field '{name}' has a different dtype than {}",
                    F::DTYPE
                );
                true
            }
        }
    }

    /// Indices that sort a 1-d record array by the named fields (first name most significant),
    /// like `np.argsort(a, order=[...], kind='stable')`.
    ///
    /// As in NumPy, fields not named in `order` still break ties, in declaration order. The sort is
    /// stable and NaN sorts last.
    ///
    /// ```
    /// use rustnum as np;
    /// np::record! { pub struct P { k: i32, v: f64 } }
    /// let a = np::Array::from(vec![P { k: 2, v: 1.0 }, P { k: 1, v: 9.0 }, P { k: 2, v: 0.5 }]);
    /// assert_eq!(a.argsort_order(&["k", "v"]).to_vec(), [1, 2, 0]);
    /// // `v` is not named, but still breaks the tie between the two `k == 2` rows
    /// assert_eq!(a.argsort_order(&["k"]).to_vec(), [1, 2, 0]);
    /// assert_eq!(a.argsort_order(&["v"]).to_vec(), [2, 0, 1]);
    /// ```
    pub fn argsort_order(&self, order: &[&str]) -> Array<usize> {
        assert!(
            self.ndim() == 1,
            "argsort_order supports 1-d structured arrays"
        );
        for name in order {
            if R::default().field_any(name).is_none() {
                unknown_field::<R>(name);
            }
        }
        let names: Vec<&str> = order
            .iter()
            .copied()
            .chain(R::FIELDS.iter().copied().filter(|f| !order.contains(f)))
            .collect();
        let values = self.to_vec();
        let mut idx: Vec<usize> = (0..values.len()).collect();
        idx.sort_by(|&i, &j| {
            for name in &names {
                let o = values[i]
                    .cmp_field(&values[j], name)
                    .unwrap_or(Ordering::Equal);
                if o != Ordering::Equal {
                    return o;
                }
            }
            Ordering::Equal
        });
        Array::from(idx)
    }

    /// A copy of a 1-d record array sorted by the named fields, like `np.sort(a, order=[...])`.
    ///
    /// ```
    /// use rustnum as np;
    /// np::record! { pub struct P { k: i32, v: f64 } }
    /// let a = np::Array::from(vec![P { k: 2, v: 1.0 }, P { k: 1, v: 9.0 }]);
    /// assert_eq!(a.sort_order(&["k"]).field::<i32>("k").to_vec(), [1, 2]);
    /// ```
    pub fn sort_order(&self, order: &[&str]) -> Array<R> {
        let idx = self.argsort_order(order).to_vec();
        let values = self.to_vec();
        Array::from(idx.into_iter().map(|i| values[i]).collect::<Vec<_>>())
    }
}
