//! Printing options for array `Display`/`Debug`, like `np.set_printoptions`.
//!
//! The options are per thread, like NumPy's context-local print options: a thread starts with the
//! defaults and `set_printoptions` only affects the calling thread.
use std::cell::RefCell;

/// The array printing options, like the dict returned by np.get_printoptions().
///
/// Build one from the defaults with struct update syntax:
/// ```
/// use rustnum as np;
/// let opts = np::PrintOptions { precision: 3, ..np::PrintOptions::default() };
/// assert_eq!(opts.precision, 3);
/// assert_eq!(opts.threshold, 1000);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrintOptions {
    /// Digits after the decimal point for floats (NumPy default 8).
    pub precision: usize,
    /// Arrays with more elements than this are summarised with `...` (default 1000).
    pub threshold: usize,
    /// Items kept at each edge of a summarised axis (default 3).
    pub edgeitems: usize,
    /// Characters per line before wrapping (default 75).
    pub linewidth: usize,
    /// Never use scientific notation for small floats when true (default false).
    pub suppress: bool,
    /// Sign handling: `'-'` only for negatives (default), `'+'` always, `' '` a space for positives.
    pub sign: char,
}
impl Default for PrintOptions {
    /// The NumPy defaults: precision 8, threshold 1000, edgeitems 3, linewidth 75, no suppress, sign '-'.
    /// ```
    /// use rustnum as np;
    /// let d = np::PrintOptions::default();
    /// assert_eq!((d.precision, d.threshold, d.edgeitems, d.linewidth), (8, 1000, 3, 75));
    /// ```
    fn default() -> Self {
        Self {
            precision: 8,
            threshold: 1000,
            edgeitems: 3,
            linewidth: 75,
            suppress: false,
            sign: '-',
        }
    }
}
thread_local! {
    static OPTIONS: RefCell<PrintOptions> = RefCell::new(PrintOptions::default());
}
fn validate(o: &PrintOptions) {
    assert!(
        matches!(o.sign, '-' | '+' | ' '),
        "sign option must be one of ' ', '+', or '-'"
    );
}
/// Set the print options of the current thread, like np.set_printoptions(...).
///
/// All fields are replaced; start from [`get_printoptions`] to change only some. Panics if `sign`
/// is not one of `'-'`, `'+'`, `' '`.
/// ```
/// use rustnum as np;
/// let before = np::get_printoptions();
/// np::set_printoptions(np::PrintOptions { precision: 3, ..before.clone() });
/// assert_eq!(np::array![1.23456789, 2.5].to_string(), "[1.235 2.5  ]");
/// np::set_printoptions(before);
/// ```
pub fn set_printoptions(options: PrintOptions) {
    validate(&options);
    OPTIONS.with(|o| *o.borrow_mut() = options);
}
/// The print options of the current thread, like np.get_printoptions().
/// ```
/// use rustnum as np;
/// assert_eq!(np::get_printoptions().precision, 8);
/// ```
pub fn get_printoptions() -> PrintOptions {
    OPTIONS.with(|o| o.borrow().clone())
}
/// Restores the previous print options when dropped; made by [`printoptions`].
#[must_use = "the previous print options come back when the guard is dropped"]
#[derive(Debug)]
pub struct PrintOptionsGuard {
    saved: PrintOptions,
}
impl Drop for PrintOptionsGuard {
    fn drop(&mut self) {
        OPTIONS.with(|o| *o.borrow_mut() = self.saved.clone());
    }
}
/// Set print options until the returned guard is dropped, like `with np.printoptions(...)`.
/// ```
/// use rustnum as np;
/// let a = np::array![1.0, 2.5, 3.25];
/// {
///     let _g = np::printoptions(np::PrintOptions { sign: '+', ..np::get_printoptions() });
///     assert_eq!(a.to_string(), "[+1.   +2.5  +3.25]");
/// }
/// assert_eq!(a.to_string(), "[1.   2.5  3.25]");
/// ```
pub fn printoptions(options: PrintOptions) -> PrintOptionsGuard {
    validate(&options);
    let saved = OPTIONS.with(|o| std::mem::replace(&mut *o.borrow_mut(), options));
    PrintOptionsGuard { saved }
}
