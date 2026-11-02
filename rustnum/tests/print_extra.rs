//! Group 9: print options honoured by Display and Debug. Expected strings come from NumPy 2.3.2
//! (`np.printoptions(...)` + `str(a)`). Options are per thread, so these tests do not interfere.
use rustnum as np;

fn with(opts: np::PrintOptions, a: &np::Array<f64>) -> String {
    let _g = np::printoptions(opts);
    a.to_string()
}
fn base() -> np::PrintOptions {
    np::PrintOptions::default()
}

#[test]
fn defaults_and_get_set() {
    assert_eq!(
        np::get_printoptions(),
        np::PrintOptions {
            precision: 8,
            threshold: 1000,
            edgeitems: 3,
            linewidth: 75,
            suppress: false,
            sign: '-'
        }
    );
    np::set_printoptions(np::PrintOptions {
        precision: 2,
        ..base()
    });
    assert_eq!(np::get_printoptions().precision, 2);
    assert_eq!(
        format!("{:?}", np::array![[1.234, 2.0], [3.0, 4.5678]]),
        "array([[1.23, 2.  ],\n       [3.  , 4.57]], dtype=float64)"
    );
    np::set_printoptions(base());
    assert_eq!(np::get_printoptions().precision, 8);
}

#[test]
fn guard_restores_previous() {
    let a = np::array![1.23456789, 2.5, 100.0];
    assert_eq!(a.to_string(), "[  1.23456789   2.5        100.        ]");
    {
        let _g = np::printoptions(np::PrintOptions {
            precision: 3,
            ..base()
        });
        assert_eq!(a.to_string(), "[  1.235   2.5   100.   ]");
        {
            let _h = np::printoptions(np::PrintOptions {
                precision: 0,
                ..base()
            });
            assert_eq!(a.to_string(), "[  1.   2. 100.]");
        }
        assert_eq!(a.to_string(), "[  1.235   2.5   100.   ]");
    }
    assert_eq!(a.to_string(), "[  1.23456789   2.5        100.        ]");
}

#[test]
fn options_are_per_thread() {
    let _g = np::printoptions(np::PrintOptions {
        precision: 1,
        ..base()
    });
    let other = std::thread::spawn(|| np::get_printoptions().precision)
        .join()
        .unwrap();
    assert_eq!(other, 8);
    assert_eq!(np::get_printoptions().precision, 1);
}

#[test]
fn precision() {
    let a = np::array![1.23456789, 2.5, 100.0];
    let p10 = np::PrintOptions {
        precision: 10,
        ..base()
    };
    assert_eq!(with(p10, &a), "[  1.23456789   2.5        100.        ]");
    let b = np::array![0.1, 0.12345678912];
    let p4 = np::PrintOptions {
        precision: 4,
        ..base()
    };
    assert_eq!(with(p4, &b), "[0.1    0.1235]");
    let c = np::array![1e-5, 1e5];
    let p2 = np::PrintOptions {
        precision: 2,
        ..base()
    };
    assert_eq!(with(p2, &c), "[1.e-05 1.e+05]");
}

#[test]
fn sign() {
    let plus = np::PrintOptions {
        sign: '+',
        ..base()
    };
    let space = np::PrintOptions {
        sign: ' ',
        ..base()
    };
    assert_eq!(
        with(plus.clone(), &np::array![1.0, 2.5, 3.25]),
        "[+1.   +2.5  +3.25]"
    );
    assert_eq!(
        with(space.clone(), &np::array![1.0, -2.5, 3.25]),
        "[ 1.   -2.5   3.25]"
    );
    assert_eq!(
        with(plus.clone(), &np::array![1.5, f64::NAN, f64::INFINITY]),
        "[+1.5 +nan +inf]"
    );
    assert_eq!(with(plus.clone(), &np::array![-1.5, 2.0]), "[-1.5 +2. ]");
    assert_eq!(
        with(plus.clone(), &np::array![1e-5, -2.0]),
        "[+1.e-05 -2.e+00]"
    );
    let _g = np::printoptions(plus);
    assert_eq!(np::array![1, -2, 3].to_string(), "[+1 -2 +3]");
    assert_eq!(np::array![true, false].to_string(), "[ True False]");
    drop(_g);
    let _h = np::printoptions(space);
    assert_eq!(np::array![1, 2, 3].to_string(), "[ 1  2  3]");
}

#[test]
#[should_panic(expected = "sign option must be one of ' ', '+', or '-'")]
fn bad_sign() {
    np::set_printoptions(np::PrintOptions {
        sign: '*',
        ..base()
    });
}

#[test]
fn suppress() {
    let b = np::array![1e-5, 1.0, 3.0];
    assert_eq!(b.to_string(), "[1.e-05 1.e+00 3.e+00]");
    let s = np::PrintOptions {
        suppress: true,
        ..base()
    };
    assert_eq!(with(s.clone(), &b), "[0.00001 1.      3.     ]");
    // very large values stay in scientific notation, as in NumPy
    assert_eq!(with(s.clone(), &np::array![1e10, 1.0]), "[1.e+10 1.e+00]");
    let p2 = np::PrintOptions {
        precision: 2,
        suppress: true,
        ..base()
    };
    assert_eq!(with(p2, &np::array![123456.789]), "[123456.79]");
}

#[test]
fn threshold_and_edgeitems() {
    let d = np::arange(0.0, 10.0, 1.0).astype::<i64>();
    let show = |threshold, edgeitems| {
        let _g = np::printoptions(np::PrintOptions {
            threshold,
            edgeitems,
            ..base()
        });
        d.to_string()
    };
    assert_eq!(show(5, 3), "[0 1 2 ... 7 8 9]");
    assert_eq!(show(5, 1), "[0 ... 9]");
    assert_eq!(show(5, 2), "[0 1 ... 8 9]");
    assert_eq!(show(100, 2), "[0 1 2 3 4 5 6 7 8 9]");
    let seven = np::arange(0, 7, 1);
    let _g = np::printoptions(np::PrintOptions {
        threshold: 6,
        edgeitems: 2,
        ..base()
    });
    assert_eq!(seven.to_string(), "[0 1 ... 5 6]");
    let m = np::arange(0, 60, 1).reshape(&[6, 10]);
    let _h = np::printoptions(np::PrintOptions {
        threshold: 20,
        edgeitems: 2,
        ..base()
    });
    assert_eq!(
        m.to_string(),
        "[[ 0  1 ...  8  9]\n [10 11 ... 18 19]\n ...\n [40 41 ... 48 49]\n [50 51 ... 58 59]]"
    );
}

#[test]
fn linewidth() {
    let e = np::arange(0, 30, 1);
    {
        let _g = np::printoptions(np::PrintOptions {
            linewidth: 30,
            ..base()
        });
        assert_eq!(
            e.to_string(),
            "[ 0  1  2  3  4  5  6  7  8\n  9 10 11 12 13 14 15 16 17\n 18 19 20 21 22 23 24 25 26\n 27 28 29]"
        );
    }
    let _g = np::printoptions(np::PrintOptions {
        linewidth: 20,
        ..base()
    });
    assert_eq!(
        e.to_string(),
        "[ 0  1  2  3  4  5\n  6  7  8  9 10 11\n 12 13 14 15 16 17\n 18 19 20 21 22 23\n 24 25 26 27 28 29]"
    );
    let f = np::arange(0.0, 12.0, 1.0).reshape(&[3, 4]) / 7.0;
    let _h = np::printoptions(np::PrintOptions {
        linewidth: 30,
        precision: 3,
        ..base()
    });
    assert_eq!(
        f.to_string(),
        "[[0.    0.143 0.286 0.429]\n [0.571 0.714 0.857 1.   ]\n [1.143 1.286 1.429 1.571]]"
    );
}
