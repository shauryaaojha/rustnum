//! A tour of rustnum: creation, arithmetic, broadcasting, slicing, reductions, printing.
//!
//! Run with: `cargo run --release -p rustnum --example quickstart`
//!
//! The equivalent NumPy code:
//!
//! ```python
//! import numpy as np
//!
//! a = np.arange(0.0, 6.0, 1.0).reshape(2, 3)
//! b = np.array([10.0, 20.0, 30.0])
//! c = a + b                        # broadcasting: (2, 3) + (3,) -> (2, 3)
//! print(c)
//! print(c.sum(), c.mean(axis=0), c.max(axis=1))
//! print(c[:, 1], c[::-1, :2])
//! print(c[c > 20])
//! print(np.linspace(0, 1, 5))
//! print(a @ a.T)
//! row = a[0].copy(); row[:] = 9.0  # `a` is unchanged
//! ```

use rustnum as np;

fn main() {
    // Creation.
    let a = np::arange(0.0, 6.0, 1.0).reshape(&[2, 3]);
    let b = np::array![10.0, 20.0, 30.0];
    let zeros = np::zeros::<f64>(&[2, 2]);
    let eye = np::eye::<f64>(3);
    let ramp = np::linspace(0.0, 1.0, 5);
    println!("a =\n{a}");
    println!("b = {b}");
    println!("zeros =\n{zeros}");
    println!("eye(3) =\n{eye}");
    println!("linspace(0, 1, 5) = {ramp}");
    println!(
        "shape = {:?}, ndim = {}, size = {}",
        a.shape(),
        a.ndim(),
        a.size()
    );

    // Arithmetic with broadcasting: shape [2, 3] + shape [3] -> shape [2, 3].
    let c = &a + &b;
    println!("\na + b =\n{c}");
    println!("a * 2.0 + 1.0 =\n{}", &a * 2.0 + 1.0);
    println!("1.0 / (a + 1.0) =\n{}", 1.0 / (&a + 1.0));

    // Slicing. These are views: no data is copied.
    println!("\nc[:, 1] = {}", c.slice(&np::s![.., 1]));
    println!("c[::-1, :2] =\n{}", c.slice(&np::s![..;-1, ..2]));
    println!("last column = {}", c.slice(&np::s![.., -1]));

    // Boolean mask: the mask has the same shape as the array.
    let big = c.gt(20.0);
    println!("\nc > 20 =\n{big}");
    println!("c[c > 20] = {}", c.mask(&big));

    // Reductions: whole array, then along an axis.
    println!("\nsum = {}", c.sum());
    println!("mean = {}", c.mean());
    println!("max = {}", c.max());
    println!("column means (axis 0) = {}", c.mean_axis(0));
    println!("row maxima (axis 1) = {}", c.max_axis(1));
    println!("argmax (flat index) = {}", c.argmax());

    // Shape changes and the matrix product.
    println!("\nc.T =\n{}", c.t());
    println!("a @ a.T =\n{}", a.matmul(&a.t()));
    println!("a.reshape(-1) = {}", a.reshape_infer(&[-1]));

    // Element-wise math is a method.
    println!("\nsqrt(a) =\n{}", a.sqrt());
    println!("clip(a, 1, 4) =\n{}", a.clip(1.0, 4.0));

    // Copy-on-write: changing `row` does not change `a`.
    let mut row = a.row(0);
    row.fill(9.0);
    println!("\nrow after fill = {row}");
    println!("a is unchanged =\n{a}");

    // To write into part of an array, say so explicitly.
    let mut d = a.copy();
    d.slice_assign(&np::s![.., 0], -1.0);
    println!("d with first column set to -1 =\n{d}");
}
