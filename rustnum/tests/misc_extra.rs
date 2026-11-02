//! Group 10: histogram2d, histogramdd, einsum_path, tensorinv, tensorsolve.
//! Expected values and report strings come from NumPy 2.3.2.
use rustnum as np;
use rustnum::Bins;

const NAN: f64 = f64::NAN;

fn close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= 1e-12 * (1.0 + w.abs()),
            "{got:?} vs {want:?}"
        );
    }
}

fn three() -> Vec<Bins> {
    vec![Bins::Count(2), Bins::Count(2), Bins::Count(2)]
}

fn xy() -> (np::Array<f64>, np::Array<f64>) {
    (
        np::array![0.1, 0.5, 1.0, 1.5, 2.0, 2.0, 3.5, -1.0, NAN],
        np::array![1.0, 2.0, 0.5, 0.5, 3.0, 2.5, 1.0, 1.0, 1.0],
    )
}

#[test]
fn histogram2d_matches_numpy() {
    let (x, y) = xy();
    let (h, xe, ye) = np::histogram2d(
        &x,
        &y,
        [Bins::Count(3), Bins::Count(2)],
        Some([(0.0, 3.0), (0.0, 3.0)]),
        false,
        None,
    );
    assert_eq!(h.shape(), [3, 2]);
    close(&h.to_vec(), &[1., 1., 2., 0., 0., 2.]);
    close(&xe.to_vec(), &[0., 1., 2., 3.]);
    close(&ye.to_vec(), &[0., 1.5, 3.]);

    let x8 = np::array![0.1, 0.5, 1.0, 1.5, 2.0, 2.0, 3.5, -1.0];
    let y8 = np::array![1.0, 2.0, 0.5, 0.5, 3.0, 2.5, 1.0, 1.0];
    let edges = [
        Bins::Edges(vec![0., 1., 2., 4.]),
        Bins::Edges(vec![0., 1., 3.]),
    ];
    let (h, xe, ye) = np::histogram2d(&x8, &y8, edges.clone(), None, false, None);
    close(&h.to_vec(), &[0., 2., 2., 0., 0., 3.]);
    close(&xe.to_vec(), &[0., 1., 2., 4.]);
    close(&ye.to_vec(), &[0., 1., 3.]);
    let (d, _, _) = np::histogram2d(&x8, &y8, edges, None, true, None);
    close(
        &d.to_vec(),
        &[
            0.,
            0.14285714285714285,
            0.2857142857142857,
            0.,
            0.,
            0.10714285714285714,
        ],
    );

    let w = np::arange(1.0, 9.0, 1.0);
    let (h, xe, ye) = np::histogram2d(
        &x8,
        &y8,
        [Bins::Count(3), Bins::Count(3)],
        None,
        false,
        Some(&w),
    );
    close(&h.to_vec(), &[9., 0., 0., 7., 2., 0., 7., 0., 11.]);
    close(&xe.to_vec(), &[-1.0, 0.5, 2.0, 3.5]);
    close(
        &ye.to_vec(),
        &[0.5, 1.3333333333333335, 2.166666666666667, 3.0],
    );

    let (d, _, _) = np::histogram2d(&x8, &y8, [Bins::Count(2), Bins::Count(3)], None, true, None);
    close(
        &d.to_vec(),
        &[
            0.19999999999999996,
            0.06666666666666665,
            0.,
            0.1333333333333333,
            0.,
            0.1333333333333334,
        ],
    );
}

#[test]
#[should_panic(expected = "x and y must have the same length.")]
fn histogram2d_length_mismatch() {
    np::histogram2d(
        &np::arange(0.0, 3.0, 1.0),
        &np::arange(0.0, 4.0, 1.0),
        [Bins::Count(2), Bins::Count(2)],
        None,
        false,
        None,
    );
}

#[test]
fn histogramdd_matches_numpy() {
    let s = np::array![
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [1.0, 2.0, 0.5],
        [0.5, 1.0, 2.0]
    ];
    let (h, e) = np::histogramdd(&s, &three(), None, false, None);
    assert_eq!(h.shape(), [2, 2, 2]);
    close(&h.to_vec(), &[1., 0., 0., 1., 0., 0., 1., 2.]);
    for edges in &e {
        close(&edges.to_vec(), &[0., 1., 2.]);
    }
    let bins = [
        Bins::Edges(vec![0., 1., 2.]),
        Bins::Count(3),
        Bins::Edges(vec![0., 2.]),
    ];
    let (h, e) = np::histogramdd(&s, &bins, None, false, None);
    assert_eq!(h.shape(), [2, 3, 1]);
    close(&h.to_vec(), &[1., 1., 0., 0., 1., 2.]);
    close(
        &e[1].to_vec(),
        &[0., 0.6666666666666666, 1.3333333333333333, 2.0],
    );
    let w = np::arange(1.0, 6.0, 1.0);
    let bins = [Bins::Count(2), Bins::Count(3), Bins::Count(2)];
    let (d, _) = np::histogramdd(&s, &bins, None, true, Some(&w));
    close(
        &d.to_vec(),
        &[
            0.1,
            0.,
            0.,
            0.5,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.2,
            0.39999999999999997,
            0.29999999999999993,
        ],
    );
    let (h, _) = np::histogramdd(
        &s,
        &three(),
        Some(&[(0., 2.), (0., 2.), (0., 2.)]),
        false,
        None,
    );
    close(&h.to_vec(), &[1., 0., 0., 1., 0., 0., 1., 2.]);
}

#[test]
#[should_panic(expected = "The dimension of bins must be equal to the dimension of the sample x.")]
fn histogramdd_bins_len() {
    np::histogramdd(
        &np::zeros::<f64>([4, 3]),
        &[Bins::Count(2)],
        None,
        false,
        None,
    );
}

fn zeros(shapes: &[&[usize]]) -> Vec<np::Array<f64>> {
    shapes.iter().map(|s| np::zeros::<f64>(*s)).collect()
}
fn path_of(sub: &str, shapes: &[&[usize]], opt: &str) -> (Vec<Vec<usize>>, String) {
    let arrays = zeros(shapes);
    let refs: Vec<&np::Array<f64>> = arrays.iter().collect();
    np::einsum_path(sub, &refs, opt)
}

#[test]
fn einsum_path_chain_report() {
    let (path, report) = path_of("ij,jk,kl->il", &[&[2, 2], &[2, 5], &[5, 2]], "greedy");
    assert_eq!(path, [vec![1, 2], vec![0, 1]]);
    let want = "  Complete contraction:  ij,jk,kl->il
         Naive scaling:  4
     Optimized scaling:  3
      Naive FLOP count:  1.200e+02
  Optimized FLOP count:  5.700e+01
   Theoretical speedup:  2.105
  Largest intermediate:  4.000e+00 elements
--------------------------------------------------------------------------
scaling                  current                                remaining
--------------------------------------------------------------------------
   3                   kl,jk->jl                                ij,jl->il
   3                   jl,ij->il                                   il->il";
    assert_eq!(report, want);
    assert_eq!(
        path_of("ij,jk,kl->il", &[&[2, 2], &[2, 5], &[5, 2]], "optimal").0,
        path
    );
}

#[test]
fn einsum_path_four_index_transform() {
    let c: &[usize] = &[10, 10];
    let i: &[usize] = &[10, 10, 10, 10];
    let (path, report) = path_of("ea,fb,abcd,gc,hd->efgh", &[c, c, i, c, c], "greedy");
    assert_eq!(path, [vec![0, 2], vec![0, 3], vec![0, 2], vec![0, 1]]);
    let want = "  Complete contraction:  ea,fb,abcd,gc,hd->efgh
         Naive scaling:  8
     Optimized scaling:  5
      Naive FLOP count:  5.000e+08
  Optimized FLOP count:  8.000e+05
   Theoretical speedup:  624.999
  Largest intermediate:  1.000e+04 elements
--------------------------------------------------------------------------
scaling                  current                                remaining
--------------------------------------------------------------------------
   5               abcd,ea->bcde                      fb,gc,hd,bcde->efgh
   5               bcde,fb->cdef                         gc,hd,cdef->efgh
   5               cdef,gc->defg                            hd,defg->efgh
   5               defg,hd->efgh                               efgh->efgh";
    assert_eq!(report, want);
}

#[test]
fn einsum_path_various_expressions() {
    let (p, r) = path_of("ij,jk->ik", &[&[3, 4], &[4, 5]], "greedy");
    assert_eq!(p, [vec![0, 1]]);
    assert!(r.contains("Optimized FLOP count:  1.210e+02"));
    assert!(r.contains("Theoretical speedup:  0.992"));
    assert!(
        r.ends_with("   3                   jk,ij->ik                                   ik->ik")
    );

    let (p, r) = path_of("ii", &[&[3, 3]], "greedy");
    assert_eq!(p, [vec![0]]);
    assert!(r.contains("Complete contraction:  ii->\n"));
    assert!(r.contains("Theoretical speedup:  0.429"));

    let shapes: [&[usize]; 4] = [&[2, 30], &[30, 3], &[3, 40], &[40, 5]];
    assert_eq!(
        path_of("ab,bc,cd,de->ae", &shapes, "greedy").0,
        [vec![2, 3], vec![0, 1], vec![0, 1]]
    );
    assert_eq!(
        path_of("ab,bc,cd,de->ae", &shapes, "optimal").0,
        [vec![0, 1], vec![0, 1], vec![0, 1]]
    );
    let (p, r) = path_of("ab,bc,cd,de->ae", &shapes, "greedy");
    assert!(r.contains("Optimized FLOP count:  1.621e+03"));
    assert!(r.contains("Theoretical speedup:  88.834"));
    assert!(
        r.contains("   3                   de,cd->ce                             ab,bc,ce->ae")
    );
    assert!(
        r.contains("   3                   bc,ab->ac                                ce,ac->ae")
    );
    let (_, ro) = path_of("ab,bc,cd,de->ae", &shapes, "optimal");
    assert!(
        ro.contains("   3                   bc,ab->ac                             cd,de,ac->ae")
    );
    assert_eq!(p.len(), 3);

    // outer products only: one step contracting everything
    let (p, r) = path_of("ab,cd,ef->abcdef", &[&[2, 2], &[2, 2], &[2, 2]], "greedy");
    assert_eq!(p, [vec![0, 1, 2]]);
    assert!(
        r.contains("   6            ef,cd,ab->abcdef                           abcdef->abcdef")
    );

    let shapes: [&[usize]; 3] = [&[5, 6], &[6, 7], &[7, 8]];
    assert_eq!(
        path_of("ij,jk,kl->", &shapes, "greedy").0,
        [vec![1, 2], vec![0, 1]]
    );
    assert_eq!(
        path_of("ij,jk,kl->", &shapes, "optimal").0,
        [vec![0, 1], vec![0, 1]]
    );
    // implicit output
    let shapes: [&[usize]; 3] = [&[2, 3], &[3, 4], &[4, 2]];
    let (p, r) = path_of("ij,jk,ki", &shapes, "greedy");
    assert_eq!(p, [vec![1, 2], vec![0, 1]]);
    assert!(r.contains("Complete contraction:  ij,jk,ki->\n"));
    assert!(r.contains("Optimized FLOP count:  6.100e+01"));

    let shapes: [&[usize]; 4] = [&[3, 4, 5], &[5, 6], &[6, 7], &[7, 3]];
    assert_eq!(
        path_of("abc,cd,de,ea->b", &shapes, "greedy").0,
        [vec![2, 3], vec![1, 2], vec![0, 1]]
    );
}

#[test]
fn einsum_path_no_optimisation_and_ellipsis() {
    let shapes: [&[usize]; 3] = [&[2, 2], &[2, 5], &[5, 2]];
    let (p, r) = path_of("ij,jk,kl->il", &shapes, "false");
    assert_eq!(p, [vec![0, 1, 2]]);
    assert!(r.contains("Optimized scaling:  4"));
    assert!(
        r.contains("   4                kl,jk,ij->il                                   il->il")
    );
    let (p, _) = path_of("...ij,...jk->...ik", &[&[4, 2, 3], &[4, 3, 5]], "greedy");
    assert_eq!(p, [vec![0, 1]]);
}

#[test]
#[should_panic(expected = "Size of label 'j' for operand 1 (3) does not match previous terms (4).")]
fn einsum_path_size_mismatch() {
    path_of("ij,jk->ik", &[&[2, 3], &[4, 2]], "greedy");
}

#[test]
#[should_panic(expected = "Output character z did not appear in the input")]
fn einsum_path_bad_output() {
    path_of("ij,jk->iz", &[&[2, 3], &[3, 2]], "greedy");
}

#[test]
#[should_panic(expected = "Number of einsum subscripts must be equal to the number of operands.")]
fn einsum_path_operand_count() {
    path_of("ij,jk->ik", &[&[2, 3]], "greedy");
}

fn m4() -> np::Array<f64> {
    np::array![
        [2., 1., 0., 1.],
        [1., 3., 1., 0.],
        [0., 1., 4., 1.],
        [1., 0., 1., 5.]
    ]
}

#[test]
fn tensorinv_matches_numpy() {
    let a = m4().reshape(&[4, 2, 2]);
    let ai = np::linalg::tensorinv(&a, 1).unwrap();
    assert_eq!(ai.shape(), [2, 2, 4]);
    close(
        &ai.to_vec(),
        &[
            0.7222222222222223,
            -0.2777777777777778,
            0.11111111111111113,
            -0.16666666666666666,
            -0.27777777777777785,
            0.47222222222222227,
            -0.13888888888888892,
            0.08333333333333333,
            0.11111111111111112,
            -0.1388888888888889,
            0.3055555555555556,
            -0.08333333333333333,
            -0.16666666666666666,
            0.08333333333333334,
            -0.08333333333333333,
            0.25,
        ],
    );
    // defining property: tensordot(ai, a, 1) == identity on the (2, 2) index pair
    let back = np::linalg::tensordot_n(&ai, &a, 1);
    assert_eq!(back.shape(), [2, 2, 2, 2]);
    assert!(
        back.reshape(&[4, 4])
            .allclose(np::eye::<f64>(4), 1e-12, 1e-12)
    );
    let t = (np::eye::<f64>(6) * 3.0
        + np::Array::from_shape_fn([6, 6], |c| ((c[0] * 6 + c[1]) as f64).cos()))
    .reshape(&[2, 3, 3, 2]);
    let ti = np::linalg::tensorinv(&t, 2).unwrap();
    assert_eq!(ti.shape(), [3, 2, 2, 3]);
    close(
        &ti.to_vec()[..5],
        &[
            0.24777963231358666,
            -0.08318236277079166,
            -0.004333543805490977,
            0.07849951534941692,
            0.08916048211113375,
        ],
    );
}

#[test]
fn tensorinv_errors() {
    let e = np::linalg::tensorinv(&np::eye::<f64>(6).reshape(&[2, 3, 6]), 1).unwrap_err();
    assert_eq!(
        e.to_string(),
        "Last 2 dimensions of the array must be square"
    );
    let e = np::linalg::tensorinv(&np::eye::<f64>(4).reshape(&[2, 2, 4]), 0).unwrap_err();
    assert_eq!(e.to_string(), "Invalid ind argument.");
    let e = np::linalg::tensorinv(&np::zeros::<f64>([4, 2, 2]), 1).unwrap_err();
    assert!(matches!(e, np::Error::Singular));
}

#[test]
fn tensorsolve_matches_numpy() {
    let a = m4().reshape(&[2, 2, 4]);
    let b = np::array![[1., 2.], [3., 4.]];
    let x = np::linalg::tensorsolve(&a, &b, None).unwrap();
    assert_eq!(x.shape(), [4]);
    close(
        &x.to_vec(),
        &[
            -0.16666666666666669,
            0.5833333333333334,
            0.41666666666666663,
            0.75,
        ],
    );
    let a7 = np::eye::<f64>(6) * 3.0
        + np::Array::from_shape_fn([6, 6], |c| ((c[0] * 6 + c[1]) as f64).cos());
    let bq = np::arange(1.0, 7.0, 1.0).reshape(&[2, 3]);
    let want = [
        0.9353010241358928,
        1.3462706626231151,
        1.7031034364099151,
        2.0039273935420168,
        2.251332212289961,
        2.452162908862737,
    ];
    let x = np::linalg::tensorsolve(&a7.reshape(&[2, 3, 6]), &bq, None).unwrap();
    assert_eq!(x.shape(), [6]);
    close(&x.to_vec(), &want);
    let x = np::linalg::tensorsolve(&a7.reshape(&[6, 2, 3]), &bq, Some(&[0])).unwrap();
    close(
        &x.to_vec(),
        &[
            0.11058418550381482,
            -1.5459202345200231,
            -1.1681824614601022,
            1.2029722675805188,
            3.6939803592834215,
            4.3210855914305855,
        ],
    );
}

#[test]
fn tensorsolve_errors() {
    let e = np::linalg::tensorsolve(
        &np::zeros::<f64>([2, 3, 5]),
        &np::zeros::<f64>([2, 3]),
        None,
    )
    .unwrap_err();
    assert_eq!(
        e.to_string(),
        "Input arrays must satisfy the requirement             prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])"
    );
    let e = np::linalg::tensorsolve(
        &np::zeros::<f64>([2, 3, 6]),
        &np::zeros::<f64>([2, 3]),
        None,
    )
    .unwrap_err();
    assert!(matches!(e, np::Error::Singular));
}
