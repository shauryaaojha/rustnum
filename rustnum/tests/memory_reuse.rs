use rustnum as np;
fn ptr<T: np::Element>(a: &np::Array<T>) -> *const T {
    a.as_slice().unwrap().as_ptr()
}
#[test]
fn owned_operators_reuse_both_sides() {
    macro_rules! check { ($op:tt) => {{
        let a=np::array![[6i32,8],[10,12]]; let b=np::array![2,4];
        let expected=(&a $op &b).to_vec(); let p=ptr(&a);
        let out=a $op &b; assert_eq!(ptr(&out),p); assert_eq!(out.to_vec(),expected);
        let a=np::array![[6i32,8],[10,12]]; let p=ptr(&a); let expected=(&a $op 2).to_vec();
        let out=a $op 2; assert_eq!(ptr(&out),p); assert_eq!(out.to_vec(),expected);
        let a=np::array![[6i32,8],[10,12]]; let b=np::array![2,4]; let p=ptr(&a);
        let expected=(&a $op &b).to_vec(); let out=a $op b;
        assert_eq!(ptr(&out),p); assert_eq!(out.to_vec(),expected);
        let a=np::array![6i32,8]; let b=np::array![[2,4],[2,4]]; let p=ptr(&b);
        let expected=(&a $op &b).to_vec(); let out=a $op b;
        assert_eq!(ptr(&out),p); assert_eq!(out.to_vec(),expected);
        let a=np::array![6i32,8]; let b=np::array![[2,4],[2,4]]; let p=ptr(&b);
        let expected=(&a $op &b).to_vec(); let out=&a $op b;
        assert_eq!(ptr(&out),p); assert_eq!(out.to_vec(),expected);
        let a=np::array![[6i32,8],[10,12]]; let shared=a.clone(); let p=ptr(&a);
        let expected=(&a $op 2).to_vec(); let out=a $op 2;
        assert_ne!(ptr(&out),p); assert_eq!(out.to_vec(),expected); assert_eq!(shared.to_vec(),[6,8,10,12]);
    }}; }
    check!(+);
    check!(-);
    check!(*);
    check!(/);
    check!(%);
    let a = np::array![-1i64, 2];
    let p = ptr(&a);
    let a = -a;
    assert_eq!(ptr(&a), p);
    assert_eq!(a.to_vec(), [1, -2]);
    let a = np::array![true, false];
    let p = ptr(&a);
    let a = !a;
    assert_eq!(ptr(&a), p);
    assert_eq!(a.to_vec(), [false, true]);
}
#[test]
fn owned_shared_and_offset_views() {
    let a = np::array![[1, 2], [3, 4]];
    let p = ptr(&a);
    let b = a.clone();
    let out = a + &b;
    assert_ne!(ptr(&out), p);
    assert_eq!(b.to_vec(), [1, 2, 3, 4]);
    let a = np::array![[1, 2], [3, 4]].t();
    let out = a + 10;
    assert_eq!(out.to_vec(), [11, 13, 12, 14]);
    let a = np::array![0, 1, 2, 3].slice(&np::s![1..3]);
    let p = ptr(&a);
    let a = a * 2;
    assert_eq!(ptr(&a), p);
    assert_eq!(a.to_vec(), [2, 4]);
    let a = np::array![i64::MAX, 1];
    let p = ptr(&a);
    let b = a.astype::<i64>();
    assert_eq!(ptr(&b), p);
    let b = b.into_astype::<i64>();
    assert_eq!(ptr(&b), p);
    assert_eq!(b.to_vec(), [i64::MAX, 1]);
}
#[test]
fn assignment_preserves_unique_buffer_and_cow() {
    let mut a = np::array![[1., 2.], [3., 4.]];
    let p = ptr(&a);
    a += np::array![10., 20.];
    a *= 2.;
    a -= 1.;
    a /= 2.;
    a %= 10.;
    assert_eq!(ptr(&a), p);
    assert_eq!(a.to_vec(), [0.5, 1.5, 2.5, 3.5]);
    let shared = a.clone();
    a += 1.;
    assert_ne!(ptr(&a), p);
    assert_eq!(shared.to_vec(), [0.5, 1.5, 2.5, 3.5]);
    let mut v = np::array![[1, 2], [3, 4]].t();
    v += np::array![10, 20];
    assert_eq!(v.to_vec(), [11, 23, 12, 24]);
    let mut v = np::Array::scalar(2).broadcast_to(&[3]);
    v += np::array![1, 2, 3];
    assert_eq!(v.to_vec(), [3, 4, 5]);
}
#[test]
fn unary_variants() {
    macro_rules! check { ($base:ident,$inplace:ident,$into:ident $(,$arg:expr)*) => {{
        let a=np::array![0.25f64,0.5,0.75]; let expected=a.$base($($arg),*).to_vec(); let p=ptr(&a);
        let mut out=a.$into($($arg),*); assert_eq!(ptr(&out),p); assert_eq!(out.to_vec(),expected);
        let shared=out.clone(); out.$inplace($($arg),*); assert_ne!(ptr(&out),p); assert_eq!(shared.to_vec(),expected);
    }}; }
    check!(exp, exp_inplace, into_exp);
    check!(exp2, exp2_inplace, into_exp2);
    check!(expm1, expm1_inplace, into_expm1);
    check!(log, log_inplace, into_log);
    check!(log2, log2_inplace, into_log2);
    check!(log10, log10_inplace, into_log10);
    check!(log1p, log1p_inplace, into_log1p);
    check!(sqrt, sqrt_inplace, into_sqrt);
    check!(cbrt, cbrt_inplace, into_cbrt);
    check!(sin, sin_inplace, into_sin);
    check!(cos, cos_inplace, into_cos);
    check!(tan, tan_inplace, into_tan);
    check!(arcsin, arcsin_inplace, into_arcsin);
    check!(arccos, arccos_inplace, into_arccos);
    check!(arctan, arctan_inplace, into_arctan);
    check!(sinh, sinh_inplace, into_sinh);
    check!(cosh, cosh_inplace, into_cosh);
    check!(tanh, tanh_inplace, into_tanh);
    check!(floor, floor_inplace, into_floor);
    check!(ceil, ceil_inplace, into_ceil);
    check!(round, round_inplace, into_round);
    check!(trunc, trunc_inplace, into_trunc);
    check!(reciprocal, reciprocal_inplace, into_reciprocal);
    check!(sigmoid, sigmoid_inplace, into_sigmoid);
    check!(relu, relu_inplace, into_relu);
    check!(abs, abs_inplace, into_abs);
    check!(square, square_inplace, into_square);
    check!(sign, sign_inplace, into_sign);
    check!(sinc, sinc_inplace, into_sinc);
    check!(i0, i0_inplace, into_i0);
    check!(deg2rad, deg2rad_inplace, into_deg2rad);
    check!(rad2deg, rad2deg_inplace, into_rad2deg);
    check!(rint, rint_inplace, into_rint);
    check!(fix, fix_inplace, into_fix);
    check!(fabs, fabs_inplace, into_fabs);
    check!(positive, positive_inplace, into_positive);
    check!(negative, negative_inplace, into_negative);
    check!(spacing, spacing_inplace, into_spacing);
    check!(nan_to_num, nan_to_num_inplace, into_nan_to_num);

    check!(powf, powf_inplace, into_powf, 2.0);
    check!(powi, powi_inplace, into_powi, 2);
    check!(clip, clip_inplace, into_clip, 0.0, 0.5);
    check!(
        round_decimals,
        round_decimals_inplace,
        into_round_decimals,
        1
    );
    let a = np::full([262144], 0.5f64);
    let p = ptr(&a);
    let a = (a / 2.).into_tanh();
    assert_eq!(ptr(&a), p);
    assert!(a.iter().all(|x| x == 0.25f64.tanh()));
    let mut a = np::array![0., 1., 2.].flip(0);
    a.tanh_inplace();
    assert_eq!(a.to_vec(), [2f64.tanh(), 1f64.tanh(), 0.]);
}
#[test]
fn copy_broadcast_and_reduce_layouts() {
    let a = np::Array::from_vec((0..24).collect(), &[2, 3, 4]);
    for view in [
        a.swapaxes(-1, -2),
        a.permute_axes(&[2, 0, 1]),
        a.flip(1),
        a.slice(&np::s![..,..,..;2]),
        a.swapaxes(-1, -2).flip(-1),
    ] {
        let expected: Vec<_> = view.iter().collect();
        assert_eq!(view.to_contiguous().to_vec(), expected);
        assert_eq!(
            (&view + 1).to_vec(),
            expected.iter().map(|x| x + 1).collect::<Vec<_>>()
        );
        for axis in 0..view.ndim() {
            let reduced = view.sum_axis(axis as isize);
            // Ground truth from explicit indexing, independent of strides and kernel traversal.
            let mut sh = view.shape().to_vec();
            let n = sh.remove(axis);
            let expected = np::Array::from_shape_fn(&sh, |coord| {
                let mut c = coord.to_vec();
                c.insert(axis, 0);
                (0..n)
                    .map(|i| {
                        c[axis] = i;
                        view.get(&c)
                    })
                    .sum::<i32>()
            });
            assert_eq!(reduced, expected);
        }
    }
    let a = np::array![[1, 2, 3], [4, 5, 6]];
    assert_eq!(
        (&a + np::array![10, 20, 30]).to_vec(),
        [11, 22, 33, 14, 25, 36]
    );
    assert_eq!(
        (&a + np::array![[10], [20]]).to_vec(),
        [11, 12, 13, 24, 25, 26]
    );
    assert_eq!(
        (np::array![10, 20, 30] + &a).to_vec(),
        [11, 22, 33, 14, 25, 36]
    );
    assert_eq!(
        (&a + np::array![30, 20, 10].flip(0)).to_vec(),
        [11, 22, 33, 14, 25, 36]
    );
    assert_eq!(np::Array::scalar(3).copy().item(), 3);
    assert_eq!((np::Array::scalar(3) + np::Array::scalar(2)).item(), 5);
    let empty = np::zeros::<f64>([0, 3]);
    assert!(empty.t().to_contiguous().is_empty());
    assert!((&empty + np::ones::<f64>([3])).is_empty());
    assert_eq!(empty.sum_axis(0).to_vec(), [0.; 3]);
    assert!(empty.mean_axis(0).iter().all(f64::is_nan));
    assert!(np::zeros::<f64>([3, 0]).sum_axis(0).is_empty());
}
#[test]
fn tiled_copy_edges_and_middle_axis_accuracy() {
    let a = np::Array::from_shape_fn([2, 67, 71], |i| (i[0] * 10000 + i[1] * 100 + i[2]) as f64);
    for view in [
        a.swapaxes(1, 2),
        a.swapaxes(1, 2).flip(1),
        a.swapaxes(1, 2).flip(2),
    ] {
        assert_eq!(
            view.to_contiguous().to_vec(),
            view.iter().collect::<Vec<_>>()
        );
    }
    let a = np::Array::from_shape_fn([2, 10000, 3], |i| if i[1] == 0 { 1e8f32 } else { 1. });
    for x in a.sum_axis(1).iter() {
        assert!((x - 100009999.).abs() <= 24.);
    }
    let a = np::array![[1., f64::NAN], [3., 4.]];
    assert_eq!(a.nanmean_axis(0).to_vec(), [2., 4.]);
    assert_eq!(a.nansum_axis(1).to_vec(), [1., 7.]);
    assert_eq!(a.argmax_axis(0).to_vec(), [1, 0]);
}

#[test]
fn parallel_transpose_and_broadcast_ground_truth() {
    let a = np::Array::from_shape_fn([513, 517], |i| (i[0] * 1000 + i[1]) as i32);
    let t = a.t().flip(0).to_contiguous();
    for r in 0..517 {
        for c in 0..513 {
            assert_eq!(t[[r, c]], (c * 1000 + 516 - r) as i32);
        }
    }
    let row = np::Array::from((0..517).map(|i| i * 2).collect::<Vec<i32>>());
    let col = np::Array::from_shape_fn([513, 1], |i| i[0] as i32);
    let sum = &a + &row;
    let cols = &a + &col;
    for r in 0..513 {
        for c in 0..517 {
            assert_eq!(sum[[r, c]], (r * 1000 + c * 3) as i32);
            assert_eq!(cols[[r, c]], (r * 1001 + c) as i32);
        }
    }
    let empty = np::zeros::<i32>([2, 0, 3]);
    assert!(empty.permute_axes(&[2, 0, 1]).copy().is_empty());
    let high = np::Array::from_vec(vec![1, 2, 3, 4], &[1, 1, 1, 1, 1, 1, 1, 2, 2]);
    assert_eq!(high.flip(-1).iter().collect::<Vec<_>>(), [2, 1, 4, 3]);
}
