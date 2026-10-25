//! rustnum: NumPy-style n-dimensional arrays for Rust.
//!
//! See `docs/PLAN.md` for the design.

mod array;
mod broadcast;
mod creation;
mod display;
mod element;
mod layout;
mod manip;
mod misc;
mod ops;
mod reduce;
mod slicing;
mod ufunc;

pub mod io;
pub mod linalg;
pub mod random;
