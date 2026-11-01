//! Group 3: index helpers. Expected values come from NumPy 2.3.2.
use rustnum as np;

fn pair(p: (np::Array<usize>, np::Array<usize>)) -> (Vec<usize>, Vec<usize>) {
    (p.0.to_vec(), p.1.to_vec())
}

#[test]
fn tril_triu_indices() {
    assert_eq!(
        pair(np::tril_indices(4, 1, None)),
        (
            vec![0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            vec![0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3]
        )
    );
    assert_eq!(
        pair(np::triu_indices(3, -1, None)),
        (vec![0, 0, 0, 1, 1, 1, 2, 2], vec![0, 1, 2, 0, 1, 2, 1, 2])
    );
    assert_eq!(
        pair(np::tril_indices(2, 0, Some(4))),
        (vec![0, 1, 1], vec![0, 0, 1])
    );
    assert_eq!(
        pair(np::triu_indices(3, 0, Some(2))),
        (vec![0, 0, 1], vec![0, 1, 1])
    );
}

#[test]
fn tri_indices_from() {
    let a = np::zeros::<f64>([2, 3]);
    assert_eq!(
        pair(np::tril_indices_from(&a, 0)),
        (vec![0, 1, 1], vec![0, 0, 1])
    );
    assert_eq!(
        pair(np::triu_indices_from(&a, 1)),
        (vec![0, 0, 1], vec![1, 2, 2])
    );
}

#[test]
#[should_panic(expected = "input array must be 2-d")]
fn tri_indices_from_needs_2d() {
    np::tril_indices_from(&np::zeros::<f64>([3]), 0);
}

#[test]
fn diag_indices_forms() {
    let d = np::diag_indices(3, 3);
    assert_eq!(d.len(), 3);
    for k in &d {
        assert_eq!(k.to_vec(), [0, 1, 2]);
    }
    assert!(np::diag_indices(3, 0).is_empty());
    let a = np::zeros::<i32>([4, 4]);
    assert_eq!(np::diag_indices_from(&a)[1].to_vec(), [0, 1, 2, 3]);
}

#[test]
#[should_panic(expected = "All dimensions of input must be of equal length")]
fn diag_indices_from_unequal() {
    np::diag_indices_from(&np::zeros::<i32>([2, 3]));
}

#[test]
#[should_panic(expected = "input array must be at least 2-d")]
fn diag_indices_from_1d() {
    np::diag_indices_from(&np::zeros::<i32>([3]));
}

fn filled(shape: &[usize], wrap: bool) -> Vec<i32> {
    let mut z = np::zeros::<i32>(shape);
    np::fill_diagonal(&mut z, 1, wrap);
    z.to_vec()
}

#[test]
fn fill_diagonal_matches_numpy() {
    assert_eq!(filled(&[5, 2], true), [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
    assert_eq!(filled(&[5, 2], false), [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]);
    assert_eq!(filled(&[2, 5], false), [1, 0, 0, 0, 0, 0, 1, 0, 0, 0]);
    assert_eq!(filled(&[2, 5], true), [1, 0, 0, 0, 0, 0, 1, 0, 0, 0]);
    let mut want = vec![0; 27];
    for i in [0, 13, 26] {
        want[i] = 1;
    }
    assert_eq!(filled(&[3, 3, 3], false), want);
    assert_eq!(
        filled(&[7, 3], true),
        [
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1
        ]
    );
    let mut z = np::zeros::<i32>([3, 3]);
    np::fill_diagonal(&mut z, [1, 2], false);
    assert_eq!(z.to_vec(), [1, 0, 0, 0, 2, 0, 0, 0, 1]);
}

#[test]
fn fill_diagonal_copy_on_write_and_views() {
    let base = np::zeros::<i32>([3, 3]);
    let mut c = base.clone();
    np::fill_diagonal(&mut c, 7, false);
    assert_eq!(base.to_vec(), [0; 9]);
    let mut t = np::zeros::<i32>([3, 3]).t();
    np::fill_diagonal(&mut t, 4, false);
    assert_eq!(t.to_vec(), [4, 0, 0, 0, 4, 0, 0, 0, 4]);
}

#[test]
#[should_panic(expected = "array must be at least 2-d")]
fn fill_diagonal_1d() {
    np::fill_diagonal(&mut np::zeros::<i32>([3]), 1, false);
}

#[test]
#[should_panic(expected = "All dimensions of input must be of equal length")]
fn fill_diagonal_nd_unequal() {
    np::fill_diagonal(&mut np::zeros::<i32>([2, 3, 2]), 1, false);
}

#[test]
fn ndindex_ndenumerate() {
    let all: Vec<Vec<usize>> = np::ndindex(&[2, 3]).collect();
    assert_eq!(
        all,
        [
            vec![0, 0],
            vec![0, 1],
            vec![0, 2],
            vec![1, 0],
            vec![1, 1],
            vec![1, 2]
        ]
    );
    assert_eq!(np::ndindex(&[]).collect::<Vec<_>>(), [Vec::<usize>::new()]);
    assert_eq!(np::ndindex(&[0, 2]).count(), 0);
    let a = np::array![[1, 2], [3, 4]].t();
    let pairs: Vec<(Vec<usize>, i32)> = np::ndenumerate(&a).collect();
    assert_eq!(
        pairs,
        [
            (vec![0, 0], 1),
            (vec![0, 1], 3),
            (vec![1, 0], 2),
            (vec![1, 1], 4)
        ]
    );
}

#[test]
fn mask_indices_triu() {
    let upper = |m: &np::Array<i64>, k: isize| {
        np::Array::from_shape_fn(m.shape(), |c| {
            if c[1] as isize - c[0] as isize >= k {
                m.get(c)
            } else {
                0
            }
        })
    };
    assert_eq!(
        pair(np::mask_indices(3, upper, 1)),
        (vec![0, 0, 1], vec![1, 2, 2])
    );
}
