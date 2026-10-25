//! Random number generation, like `np.random`.
//!
//! The engine uses xoshiro256** seeded by SplitMix64, so the same seed produces the same
//! sequence on every platform. The streams are not NumPy's: `Rng::seed(42)` does not reproduce
//! `np.random.default_rng(42)` (which uses PCG64).
//!
//! ```
//! use rustnum as np;
//! let mut rng = np::random::Rng::seed(7);
//! let a = rng.random(&[2, 3]);
//! assert_eq!(a.shape(), &[2, 3]);
//! assert_eq!(a, np::random::Rng::seed(7).random(&[2, 3]));
//! ```
//!
//! # Architecture
//!
//! - `engine` (private): raw generator, distributions and `Vec`-producing functions.
//! - `api`: the public [`Rng`] wrapper, which returns `Array`.

mod api;
pub(crate) mod engine;

pub use api::Rng;
