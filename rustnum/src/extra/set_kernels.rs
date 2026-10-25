//! Set kernels: `intersect1d`, `union1d`, `setdiff1d`, `setxor1d`, `isin`.
//!
//! All results are sorted with NaN last, like NumPy. NaNs are never equal to each other, so
//! `intersect1d` and `isin` drop them, while `union1d` keeps one NaN and `setxor1d` follows
//! NumPy's rule exactly (a NaN present in both inputs appears twice).

use super::{cmp_nan_last, is_nan_like};

/// `np.unique(x)`: sorted distinct values, all NaNs collapsed into one at the end.
pub(crate) fn unique<T: PartialOrd + Copy>(x: &[T]) -> Vec<T> {
    let mut v = x.to_vec();
    v.sort_by(cmp_nan_last);
    v.dedup_by(|a, b| a == b || (is_nan_like(a) && is_nan_like(b)));
    v
}

/// Is `x` present in the NaN-last sorted slice `sorted`? (NaN is never present.)
fn contains_sorted<T: PartialOrd + Copy>(sorted: &[T], x: &T) -> bool {
    let i = sorted.partition_point(|e| cmp_nan_last(e, x).is_lt());
    i < sorted.len() && sorted[i] == *x
}

/// `np.intersect1d(a, b)`: sorted values present in both.
pub(crate) fn intersect1d<T: PartialOrd + Copy>(a: &[T], b: &[T]) -> Vec<T> {
    let (ua, ub) = (unique(a), unique(b));
    ua.into_iter().filter(|x| contains_sorted(&ub, x)).collect()
}

/// `np.union1d(a, b)`: sorted distinct values present in either.
pub(crate) fn union1d<T: PartialOrd + Copy>(a: &[T], b: &[T]) -> Vec<T> {
    let mut all = a.to_vec();
    all.extend_from_slice(b);
    unique(&all)
}

/// `np.setdiff1d(a, b)`: sorted distinct values of `a` that are not in `b`.
pub(crate) fn setdiff1d<T: PartialOrd + Copy>(a: &[T], b: &[T]) -> Vec<T> {
    let ub = unique(b);
    unique(a)
        .into_iter()
        .filter(|x| !contains_sorted(&ub, x))
        .collect()
}

/// `np.setxor1d(a, b)`: sorted values present in exactly one of the inputs.
pub(crate) fn setxor1d<T: PartialOrd + Copy>(a: &[T], b: &[T]) -> Vec<T> {
    let mut aux = unique(a);
    aux.extend(unique(b));
    aux.sort_by(cmp_nan_last);
    let n = aux.len();
    // Keep an element when it differs from both neighbours (NaN differs from everything).
    (0..n)
        .filter(|&i| (i == 0 || aux[i] != aux[i - 1]) && (i + 1 == n || aux[i] != aux[i + 1]))
        .map(|i| aux[i])
        .collect()
}

/// `np.isin(element, test_elements)`: one flag per element of `element`.
pub(crate) fn isin<T: PartialOrd + Copy>(element: &[T], test_elements: &[T]) -> Vec<bool> {
    let ut = unique(test_elements);
    element.iter().map(|x| contains_sorted(&ut, x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::assert_close;

    fn check_sets(
        a: &[i64],
        b: &[i64],
        inter: &[i64],
        uni: &[i64],
        diff: &[i64],
        xor: &[i64],
        isin_a: &[bool],
    ) {
        assert_eq!(intersect1d(a, b), inter);
        assert_eq!(union1d(a, b), uni);
        assert_eq!(setdiff1d(a, b), diff);
        assert_eq!(setxor1d(a, b), xor);
        assert_eq!(isin(a, b), isin_a);
    }

    #[test]
    fn float_sets_with_nan_match_numpy() {
        let a = [1.0, 2.0, f64::NAN, f64::NAN, 3.0];
        let b = [f64::NAN, 2.0, 5.0];
        assert_close(&intersect1d(&a, &b), &[2.0]);
        assert_close(&union1d(&a, &b), &[1.0, 2.0, 3.0, 5.0, f64::NAN]);
        assert_close(&setdiff1d(&a, &b), &[1.0, 3.0, f64::NAN]);
        assert_close(&setxor1d(&a, &b), &[1.0, 3.0, 5.0, f64::NAN, f64::NAN]);
        assert_eq!(isin(&a, &b), vec![false, true, false, false, false]);
        assert_close(&unique(&a), &[1.0, 2.0, 3.0, f64::NAN]);
        assert_close(&setxor1d(&a, &[1.0]), &[2.0, 3.0, f64::NAN]);
        assert_close(&setdiff1d(&b, &a), &[5.0, f64::NAN]);
    }

    #[test]
    fn integer_sets() {
        let a: Vec<i64> = vec![5, 3, 3, 9, -1, 0, 5, 7];
        let b: Vec<i64> = vec![7, 7, 4, 3, 0, 12];
        assert_eq!(intersect1d(&a, &b), vec![0, 3, 7]);
        assert_eq!(union1d(&a, &b), vec![-1, 0, 3, 4, 5, 7, 9, 12]);
        assert_eq!(setdiff1d(&a, &b), vec![-1, 5, 9]);
        assert_eq!(setdiff1d(&b, &a), vec![4, 12]);
        assert_eq!(setxor1d(&a, &b), vec![-1, 4, 5, 9, 12]);
        assert_eq!(
            isin(&a, &b),
            vec![false, true, true, false, false, true, false, true]
        );
        assert_eq!(isin(&b, &a), vec![true, true, false, true, true, false]);
        assert_eq!(unique(&a), vec![-1, 0, 3, 5, 7, 9]);
    }

    #[test]
    fn empty_and_single_inputs() {
        let e: [f64; 0] = [];
        assert!(intersect1d(&e, &[1.0]).is_empty());
        assert!(intersect1d(&[1.0, 2.0], &e).is_empty());
        assert_eq!(union1d(&e, &[2.0, 1.0, 2.0]), vec![1.0, 2.0]);
        assert!(union1d(&e, &e).is_empty());
        assert!(setdiff1d(&e, &[1.0]).is_empty());
        assert_eq!(setdiff1d(&[3.0, 1.0, 3.0], &e), vec![1.0, 3.0]);
        assert_eq!(setxor1d(&[1.0], &[1.0]), Vec::<f64>::new());
        assert_eq!(setxor1d(&[1.0], &e), vec![1.0]);
        assert!(isin(&e, &[1.0]).is_empty());
        assert_eq!(isin(&[1.0, 2.0], &e), vec![false, false]);
        assert_eq!(intersect1d(&[2.0], &[2.0]), vec![2.0]);
        assert_eq!(isin(&[f64::NAN], &[f64::NAN]), vec![false]);
        assert_eq!(unique(&[f64::NAN, 1.0, f64::NAN, 1.0]).len(), 2);
        assert_eq!(
            unique(&[f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0]),
            vec![f64::NEG_INFINITY, 0.0, f64::INFINITY]
        );
        assert_eq!(union1d(&[true, false], &[true]), vec![false, true]);
    }

    #[test]
    fn random_integer_cases_match_numpy() {
        check_sets(
            &[1, 2, 2, 2, 9, 10, 6, 0, 1, 3, 5, 7],
            &[3, 1, 8, 8, 0, 1, 5],
            &[0, 1, 3, 5],
            &[0, 1, 2, 3, 5, 6, 7, 8, 9, 10],
            &[2, 6, 7, 9, 10],
            &[2, 6, 7, 8, 9, 10],
            &[
                true, false, false, false, false, false, false, true, true, true, true, false,
            ],
        );
        check_sets(
            &[10, 6, 5, 5, 7],
            &[2, 8, 9, 11, 9, 3, 3, 7],
            &[7],
            &[2, 3, 5, 6, 7, 8, 9, 10, 11],
            &[5, 6, 10],
            &[2, 3, 5, 6, 8, 9, 10, 11],
            &[false, false, false, false, true],
        );
        check_sets(
            &[8, 10, 3, 11, 0, 0, 11, 11, 3],
            &[3, 0],
            &[0, 3],
            &[0, 3, 8, 10, 11],
            &[8, 10, 11],
            &[8, 10, 11],
            &[false, false, true, false, true, true, false, false, true],
        );
        check_sets(
            &[7, 7, 2, 5, 2, 9, 5, 0, 3, 8, 6, 4, 3],
            &[7],
            &[7],
            &[0, 2, 3, 4, 5, 6, 7, 8, 9],
            &[0, 2, 3, 4, 5, 6, 8, 9],
            &[0, 2, 3, 4, 5, 6, 8, 9],
            &[
                true, true, false, false, false, false, false, false, false, false, false, false,
                false,
            ],
        );
        check_sets(
            &[6, 11, 11, 2, 7, 7, 2, 3, 5],
            &[3, 8, 7, 2, 4, 9, 10, 7, 0, 8, 2],
            &[2, 3, 7],
            &[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            &[5, 6, 11],
            &[0, 4, 5, 6, 8, 9, 10, 11],
            &[false, false, false, true, true, true, true, true, false],
        );
        check_sets(
            &[10, 5, 11, 9, 3, 10, 4, 1, 7, 10, 7, 4],
            &[5, 4, 1, 2, 8, 9, 3, 7, 10, 1, 3, 6, 6],
            &[1, 3, 4, 5, 7, 9, 10],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            &[11],
            &[2, 6, 8, 11],
            &[
                true, true, false, true, true, true, true, true, true, true, true, true,
            ],
        );
    }
}
