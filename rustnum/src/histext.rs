//! Multi-dimensional histograms: `histogram2d` and `histogramdd`.
use crate::{Array, Bins, histogram_bin_edges};

/// Histogram of `D`-dimensional samples, like np.histogramdd(sample, bins, range, density, weights).
///
/// `sample` has shape `(N, D)`. `bins` gives one [`Bins`] per dimension (a count or explicit
/// edges); `range` optionally gives `(min, max)` per dimension. Bins are half-open except the last
/// of each dimension, which includes its right edge; samples outside the edges (and NaNs) are
/// ignored. Returns the `D`-dimensional counts (or density) and the `D` edge arrays.
/// ```
/// use rustnum as np;
/// let sample = np::array![[0.0, 0.0], [1.0, 1.0], [1.0, 0.5], [2.0, 2.0]];
/// let (h, edges) = np::histogramdd(&sample, &[np::Bins::Count(2), np::Bins::Count(2)], None, false, None);
/// assert_eq!(h.to_vec(), [1.0, 0.0, 1.0, 2.0]);
/// assert_eq!(edges[0].to_vec(), [0.0, 1.0, 2.0]);
/// ```
pub fn histogramdd(
    sample: &Array<f64>,
    bins: &[Bins],
    range: Option<&[(f64, f64)]>,
    density: bool,
    weights: Option<&Array<f64>>,
) -> (Array<f64>, Vec<Array<f64>>) {
    assert!(
        sample.ndim() == 2,
        "sample must be a 2-d array of shape (N, D)"
    );
    let (n, d) = (sample.shape()[0], sample.shape()[1]);
    assert!(
        bins.len() == d,
        "The dimension of bins must be equal to the dimension of the sample x."
    );
    if let Some(r) = range {
        assert!(
            r.len() == d,
            "range argument must have one entry per dimension"
        );
    }
    if let Some(w) = weights {
        assert!(
            w.size() == n,
            "weights should have the same shape as sample[:, 0]"
        );
    }
    let edges: Vec<Array<f64>> = (0..d)
        .map(|k| {
            histogram_bin_edges(
                &sample.index_axis(1, k).to_contiguous(),
                bins[k].clone(),
                range.map(|r| r[k]),
            )
        })
        .collect();
    let edge_vecs: Vec<Vec<f64>> = edges.iter().map(|e| e.to_vec()).collect();
    let shape: Vec<usize> = edge_vecs.iter().map(|e| e.len() - 1).collect();
    let total: usize = shape.iter().product();
    let mut hist = vec![0.0f64; total];
    let w = weights.map(|w| w.to_vec());
    let data = sample.to_contiguous();
    let data = data.as_slice().expect("contiguous");
    'points: for i in 0..n {
        let mut flat = 0usize;
        for k in 0..d {
            let x = data[i * d + k];
            let e = &edge_vecs[k];
            let nb = e.len() - 1;
            let mut idx = e.partition_point(|&v| v <= x); // searchsorted side="right"
            if x == e[nb] {
                idx -= 1;
            }
            if x.is_nan() || idx < 1 || idx > nb {
                continue 'points;
            }
            flat = flat * nb + (idx - 1);
        }
        hist[flat] += w.as_ref().map_or(1.0, |w| w[i]);
    }
    if density {
        let s: f64 = hist.iter().sum();
        let mut stride = 1usize;
        for k in (0..d).rev() {
            let nb = shape[k];
            for (flat, h) in hist.iter_mut().enumerate() {
                let b = (flat / stride) % nb;
                *h /= edge_vecs[k][b + 1] - edge_vecs[k][b];
            }
            stride *= nb;
        }
        for h in hist.iter_mut() {
            *h /= s;
        }
    }
    (Array::from_vec(hist, &shape), edges)
}

/// Two-dimensional histogram of `x` and `y`, like np.histogram2d(x, y, bins, range, density, weights).
///
/// Returns `(H, xedges, yedges)` where `H[i, j]` counts samples with `x` in x-bin `i` and `y` in
/// y-bin `j`. See [`histogramdd`] for binning rules. NumPy's default is 10 bins per axis.
/// ```
/// use rustnum as np;
/// let x = np::array![0.0, 1.0, 1.0, 2.0];
/// let y = np::array![0.0, 1.0, 0.5, 2.0];
/// let (h, xe, ye) = np::histogram2d(&x, &y, [np::Bins::Count(2), np::Bins::Count(2)], None, false, None);
/// assert_eq!(h.to_vec(), [1.0, 0.0, 1.0, 2.0]);
/// assert_eq!(xe.to_vec(), [0.0, 1.0, 2.0]);
/// assert_eq!(ye.to_vec(), [0.0, 1.0, 2.0]);
/// ```
pub fn histogram2d(
    x: &Array<f64>,
    y: &Array<f64>,
    bins: [Bins; 2],
    range: Option<[(f64, f64); 2]>,
    density: bool,
    weights: Option<&Array<f64>>,
) -> (Array<f64>, Array<f64>, Array<f64>) {
    assert!(x.size() == y.size(), "x and y must have the same length.");
    let mut data = Vec::with_capacity(2 * x.size());
    for (a, b) in x.iter().zip(y.iter()) {
        data.push(a);
        data.push(b);
    }
    let sample = Array::from_vec(data, &[x.size(), 2]);
    let (h, mut edges) = histogramdd(
        &sample,
        &bins,
        range.as_ref().map(|r| &r[..]),
        density,
        weights,
    );
    let ye = edges.pop().expect("two edge arrays");
    let xe = edges.pop().expect("two edge arrays");
    (h, xe, ye)
}
