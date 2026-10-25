//! Linear-algebra implementation details.
//!
//! The public `Array` adapters live in `api.rs.pending` until the core array
//! implementation is merged.  Keeping numerical kernels independent makes
//! them directly testable and usable by those adapters without copies.

// The Array adapter is intentionally pending the Phase 1 merge.  Until it is
// wired, these callable internal kernels have no non-test call sites.
#[allow(dead_code)]
pub(crate) mod kernels;

#[cfg(test)]
mod tests;
