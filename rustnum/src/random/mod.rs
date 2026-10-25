//! Random number generation, like `np.random`.
//!
//! The engine uses xoshiro256** seeded by SplitMix64.
//! Same seed produces the same sequence on every platform.
//!
//! # Architecture
//!
//! - `engine` (private): raw RNG, distributions, Vec-producing functions.
//! - The public `Rng` API (in `api.rs.pending`) wraps `engine` and returns
//!   `Array` — it will be wired after Phase 1 merges.
#![allow(dead_code)]

pub(crate) mod engine;

// Re-export for internal use
#[allow(unused_imports)]
pub(crate) use engine::Xoshiro256StarStar;
