//! Polynomial packages and convenience classes, like `numpy.polynomial`.
//!
//! Provides orthogonal polynomial series classes (`Polynomial`, `Chebyshev`, `Legendre`,
//! `Laguerre`, `Hermite`, `HermiteE`), along with submodules providing mathematical
//! operations for each polynomial family.

#![allow(clippy::module_inception)]

pub mod chebyshev;
pub mod classes;
pub mod hermite;
pub mod hermite_e;
pub mod laguerre;
pub mod legendre;
pub mod polynomial;
pub mod polyutils;

pub use classes::{Chebyshev, Hermite, HermiteE, Laguerre, Legendre, PolyKind, Polynomial};

// Re-export standard polynomial functions
pub use polynomial::{
    polyadd, polycompanion, polyder, polydiv, polyfit, polyfromroots, polyint, polyline, polymul,
    polymulx, polypow, polyroots, polysub, polytrim, polyval, polyval_array, polyvander,
};

// Re-export Chebyshev functions
pub use chebyshev::{
    cheb2poly, chebadd, chebcompanion, chebder, chebdiv, chebfit, chebfromroots, chebgauss,
    chebint, chebline, chebmul, chebmulx, chebpow, chebroots, chebsub, chebtrim, chebval,
    chebval_array, chebvander, chebweight, chebweight_array, poly2cheb,
};

// Re-export Legendre functions
pub use legendre::{
    leg2poly, legadd, legcompanion, legder, legdiv, legfit, legfromroots, leggauss, legint,
    legline, legmul, legmulx, legpow, legroots, legsub, legtrim, legval, legval_array, legvander,
    legweight, legweight_array, poly2leg,
};

// Re-export Laguerre functions
pub use laguerre::{
    lag2poly, lagadd, lagcompanion, lagder, lagdiv, lagfit, lagfromroots, laggauss, lagint,
    lagline, lagmul, lagmulx, lagpow, lagroots, lagsub, lagtrim, lagval, lagval_array, lagvander,
    lagweight, lagweight_array, poly2lag,
};

// Re-export Hermite functions
pub use hermite::{
    herm2poly, hermadd, hermcompanion, hermder, hermdiv, hermfit, hermfromroots, hermgauss,
    hermint, hermline, hermmul, hermmulx, hermpow, hermroots, hermsub, hermtrim, hermval,
    hermval_array, hermvander, hermweight, hermweight_array, poly2herm,
};

// Re-export HermiteE functions
pub use hermite_e::{
    herme2poly, hermeadd, hermecompanion, hermeder, hermediv, hermefit, hermefromroots, hermegauss,
    hermeint, hermeline, hermemul, hermemulx, hermepow, hermeroots, hermesub, hermetrim, hermeval,
    hermeval_array, hermevander, hermeweight, hermeweight_array, poly2herme,
};
