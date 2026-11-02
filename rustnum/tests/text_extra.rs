//! Group 8: loadtxt, savetxt, genfromtxt. Expected values come from NumPy 2.3.2.
use rustnum as np;
use std::path::PathBuf;

const NAN: f64 = f64::NAN;

fn tmp(name: &str, content: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("rustnum_text_extra_{name}"));
    std::fs::write(&p, content).unwrap();
    p
}
fn load(name: &str, content: &str) -> np::Result<np::Array<f64>> {
    np::io::loadtxt::<f64>(tmp(name, content), Some("#"), None, 0, None, None)
}
fn same(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!((g.is_nan() && w.is_nan()) || g == w, "{got:?} vs {want:?}");
    }
}

#[test]
fn loadtxt_basic_and_comments() {
    let t = "# header line\n1 2 3\n4 5 6  # tail\n\n7 8 9\n";
    let a = load("a.txt", t).unwrap();
    assert_eq!(a.shape(), [3, 3]);
    same(&a.to_vec(), &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
    let p = tmp("a2.txt", t);
    let c = np::io::loadtxt::<f64>(&p, Some("#"), None, 0, Some(&[0, 2]), None).unwrap();
    assert_eq!(c.shape(), [3, 2]);
    same(&c.to_vec(), &[1., 3., 4., 6., 7., 9.]);
    let last = np::io::loadtxt::<f64>(&p, Some("#"), None, 0, Some(&[-1]), None).unwrap();
    assert_eq!(last.shape(), [3]);
    same(&last.to_vec(), &[3., 6., 9.]);
    let one = np::io::loadtxt::<f64>(&p, Some("#"), None, 1, None, Some(1)).unwrap();
    assert_eq!(one.shape(), [3]);
    same(&one.to_vec(), &[1., 2., 3.]);
    let swapped = np::io::loadtxt::<f64>(
        tmp("n.txt", "1 2\n3 4\n"),
        None,
        None,
        0,
        Some(&[1, 0]),
        None,
    )
    .unwrap();
    same(&swapped.to_vec(), &[2., 1., 4., 3.]);
    let ints = np::io::loadtxt::<i64>(&p, Some("#"), None, 0, Some(&[0]), None).unwrap();
    assert_eq!(ints.to_vec(), [1, 4, 7]);
}

#[test]
fn loadtxt_delimiter_squeeze_and_special() {
    let p = tmp("b.csv", "x,y\n1.5,2\n-3,4e2\n");
    let b = np::io::loadtxt::<f64>(&p, Some("#"), Some(","), 1, None, None).unwrap();
    same(&b.to_vec(), &[1.5, 2., -3., 400.]);
    assert_eq!(load("c.txt", "5\n").unwrap().shape(), [] as [usize; 0]);
    assert_eq!(load("c.txt", "5\n").unwrap().item(), 5.0);
    assert_eq!(load("d.txt", "1 2 3\n").unwrap().shape(), [3]);
    assert_eq!(load("e.txt", "1\n2\n3\n").unwrap().shape(), [3]);
    let semi = np::io::loadtxt::<f64>(
        tmp("i.txt", "% c\n1;2\n3;4\n"),
        Some("%"),
        Some(";"),
        0,
        None,
        None,
    )
    .unwrap();
    same(&semi.to_vec(), &[1., 2., 3., 4.]);
    let sp = load("j.txt", "nan inf -inf\n1e3 .5 +2.\n").unwrap();
    same(
        &sp.to_vec(),
        &[NAN, f64::INFINITY, f64::NEG_INFINITY, 1000., 0.5, 2.],
    );
    assert_eq!(load("empty.txt", "").unwrap().shape(), [0]);
    let none =
        np::io::loadtxt::<f64>(tmp("n2.txt", "1 2\n3 4\n"), None, None, 0, None, Some(0)).unwrap();
    assert_eq!(none.shape(), [0]);
}

#[test]
fn loadtxt_errors() {
    let e = load("f.txt", "1 2\n3\n").unwrap_err();
    assert_eq!(
        e.to_string(),
        "the number of columns changed from 2 to 1 at row 2; use `usecols` to select a subset and avoid this error"
    );
    assert!(matches!(e, np::Error::Parse(_)));
    assert_eq!(
        load("g.txt", "1 2\n3 abc\n").unwrap_err().to_string(),
        "could not convert string 'abc' to float64 at row 1, column 2."
    );
    assert_eq!(
        np::io::loadtxt::<i64>(tmp("h.txt", "1 2.5\n3 4\n"), None, None, 0, None, None)
            .unwrap_err()
            .to_string(),
        "could not convert string '2.5' to int64 at row 0, column 2."
    );
    assert_eq!(
        np::io::loadtxt::<f64>(tmp("a3.txt", "1 2 3\n"), None, None, 0, Some(&[5]), None)
            .unwrap_err()
            .to_string(),
        "invalid column index 5 at row 1 with 3 columns"
    );
    let missing = np::io::loadtxt::<f64>("/definitely/not/here.txt", None, None, 0, None, None);
    assert!(matches!(missing, Err(np::Error::Io(_))));
}

fn saved(name: &str, f: impl FnOnce(&std::path::Path) -> np::Result<()>) -> String {
    let p = std::env::temp_dir().join(format!("rustnum_text_extra_{name}"));
    f(&p).unwrap();
    std::fs::read_to_string(p).unwrap()
}

#[test]
fn savetxt_matches_numpy() {
    let x = np::array![[1.5, -2.25], [3.0, 4e10]];
    let d = "%.18e";
    assert_eq!(
        saved("s1.txt", |p| np::io::savetxt(
            p, &x, d, " ", "\n", "", "", "# "
        )),
        "1.500000000000000000e+00 -2.250000000000000000e+00\n3.000000000000000000e+00 4.000000000000000000e+10\n"
    );
    assert_eq!(
        saved("s2.txt", |p| np::io::savetxt(
            p,
            &x,
            "%.3f",
            ",",
            "\n",
            "a,b\nsecond",
            "end",
            "% "
        )),
        "% a,b\n% second\n1.500,-2.250\n3.000,40000000000.000\n% end\n"
    );
    assert_eq!(
        saved("s3.txt", |p| np::io::savetxt(
            p,
            &np::array![1, 2, 3],
            "%d",
            " ",
            "\n",
            "",
            "",
            "# "
        )),
        "1\n2\n3\n"
    );
    let ints = np::array![[1, 2], [3, 4]];
    assert_eq!(
        saved("s4.txt", |p| np::io::savetxt(
            p, &ints, "%d %5.1f", " ", "\n", "", "", "# "
        )),
        "1   2.0\n3   4.0\n"
    );
    assert_eq!(
        saved("s5.txt", |p| np::io::savetxt(
            p,
            &x,
            "%10.3e|%-8.2g|",
            " ",
            "\n",
            "",
            "",
            "# "
        )),
        " 1.500e+00|-2.2    |\n 3.000e+00|4e+10   |\n"
    );
    let g = np::array![[0.0001234, 123456789.0, 1e-10], [100000.0, 1e5, 0.5]];
    assert_eq!(
        saved("s6.txt", |p| np::io::savetxt(
            p, &g, "%g", " ", "\n", "", "", "# "
        )),
        "0.0001234 1.23457e+08 1e-10\n100000 100000 0.5\n"
    );
    assert_eq!(
        saved("s7.txt", |p| np::io::savetxt(
            p,
            &np::array![[-1.5, 2.0]],
            "%+08.2f",
            ";",
            "\n",
            "",
            "",
            "# "
        )),
        "-0001.50;+0002.00\n"
    );
    assert_eq!(
        saved("s8.txt", |p| np::io::savetxt(
            p,
            &np::array![[1i64, 2]],
            "% d",
            " ",
            "\n",
            "",
            "",
            "# "
        )),
        " 1  2\n"
    );
    assert_eq!(
        saved("s9.txt", |p| np::io::savetxt(
            p,
            &np::array![NAN, f64::INFINITY, f64::NEG_INFINITY],
            "%.2f",
            " ",
            "\n",
            "",
            "",
            "# "
        )),
        "nan\ninf\n-inf\n"
    );
    assert_eq!(
        saved("s10.txt", |p| np::io::savetxt(
            p,
            &np::array![(1i64 << 62) + 1],
            "%d",
            " ",
            "\n",
            "",
            "",
            "# "
        )),
        "4611686018427387905\n"
    );
}

#[test]
fn savetxt_errors_and_roundtrip() {
    let p = std::env::temp_dir().join("rustnum_text_extra_err.txt");
    let e = np::io::savetxt(
        &p,
        &np::Array::scalar(3.0),
        "%.18e",
        " ",
        "\n",
        "",
        "",
        "# ",
    )
    .unwrap_err();
    assert_eq!(
        e.to_string(),
        "Expected 1D or 2D array, got 0D array instead"
    );
    let e = np::io::savetxt(
        &p,
        &np::zeros::<f64>([2, 2, 2]),
        "%.18e",
        " ",
        "\n",
        "",
        "",
        "# ",
    )
    .unwrap_err();
    assert_eq!(
        e.to_string(),
        "Expected 1D or 2D array, got 3D array instead"
    );
    let x = np::array![[1.0, 2.0], [3.0, 4.0]];
    let e = np::io::savetxt(&p, &x, "%d %d %d", " ", "\n", "", "", "# ").unwrap_err();
    assert_eq!(
        e.to_string(),
        "fmt has wrong number of % formats:  %d %d %d"
    );
    let e = np::io::savetxt(&p, &x, "%q", " ", "\n", "", "", "# ").unwrap_err();
    assert!(matches!(e, np::Error::Format(_)));
    // default format round-trips exactly
    let y = np::array![[0.1, 1.0 / 3.0], [1e-300, -7.5e200]];
    np::io::savetxt(&p, &y, "%.18e", " ", "\n", "", "", "# ").unwrap();
    let back = np::io::loadtxt::<f64>(&p, Some("#"), None, 0, None, None).unwrap();
    assert_eq!(back, y);
}

fn gfile(name: &str, content: &str) -> np::Result<(np::Array<f64>, Vec<String>)> {
    np::io::genfromtxt::<f64>(
        tmp(name, content),
        Some("#"),
        Some(","),
        1,
        0,
        None,
        None,
        false,
        None,
    )
}

#[test]
fn genfromtxt_missing_and_options() {
    let t = "a,b c,d\n1,,3\n4,5,x\n7,8,9\n";
    let (a, names) = gfile("g1.csv", t).unwrap();
    assert!(names.is_empty());
    assert_eq!(a.shape(), [3, 3]);
    same(&a.to_vec(), &[1., NAN, 3., 4., 5., NAN, 7., 8., 9.]);
    let p = tmp("g1b.csv", t);
    let (f, _) = np::io::genfromtxt::<f64>(
        &p,
        Some("#"),
        Some(","),
        1,
        0,
        Some(-1.0),
        Some(&[0, 2]),
        false,
        None,
    )
    .unwrap();
    same(&f.to_vec(), &[1., 3., 4., -1., 7., 9.]);
    let (foot, _) =
        np::io::genfromtxt::<f64>(&p, Some("#"), Some(","), 1, 1, None, None, false, None).unwrap();
    same(&foot.to_vec(), &[1., NAN, 3., 4., 5., NAN]);
    let (mx, _) =
        np::io::genfromtxt::<f64>(&p, Some("#"), Some(","), 1, 0, None, None, false, Some(1))
            .unwrap();
    assert_eq!(mx.shape(), [3]);
    same(&mx.to_vec(), &[1., NAN, 3.]);
    let (g6, _) = gfile("g6.csv", "h\n1,2,3\n4,,6\n").unwrap();
    same(&g6.to_vec(), &[1., 2., 3., 4., NAN, 6.]);
    let (g6z, _) = np::io::genfromtxt::<f64>(
        tmp("g6b.csv", "1,2,3\n4,,6\n"),
        Some("#"),
        Some(","),
        0,
        0,
        Some(0.0),
        None,
        false,
        None,
    )
    .unwrap();
    same(&g6z.to_vec(), &[1., 2., 3., 4., 0., 6.]);
    let (one, _) = np::io::genfromtxt::<f64>(
        tmp("g4.txt", "7\n"),
        Some("#"),
        None,
        0,
        0,
        None,
        None,
        false,
        None,
    )
    .unwrap();
    assert_eq!(one.shape(), [] as [usize; 0]);
    let (blank, _) = np::io::genfromtxt::<f64>(
        tmp("g5.txt", "1 2\n\n# c\n3 4\n"),
        Some("#"),
        None,
        0,
        0,
        None,
        None,
        false,
        None,
    )
    .unwrap();
    same(&blank.to_vec(), &[1., 2., 3., 4.]);
    let (skipped, _) = gfile("g7.csv", "1,2\n3,4\n").unwrap();
    same(&skipped.to_vec(), &[3., 4.]);
    let (empty, _) = np::io::genfromtxt::<f64>(
        tmp("g8.csv", "1,2\n"),
        Some("#"),
        Some(","),
        5,
        0,
        None,
        None,
        false,
        None,
    )
    .unwrap();
    assert_eq!(empty.shape(), [0]);
}

#[test]
fn genfromtxt_names() {
    let (a, names) = np::io::genfromtxt::<f64>(
        tmp("gn.csv", "a,b c,d\n1,,3\n4,5,x\n7,8,9\n"),
        Some("#"),
        Some(","),
        0,
        0,
        None,
        None,
        true,
        None,
    )
    .unwrap();
    assert_eq!(names, ["a", "b_c", "d"]);
    assert_eq!(a.shape(), [3, 3]);
    same(&a.to_vec(), &[1., NAN, 3., 4., 5., NAN, 7., 8., 9.]);
    // names line may start with the comment marker
    let (b, names) = np::io::genfromtxt::<f64>(
        tmp("gn2.txt", "# x y z\n1 2 3\n4 5 6\n"),
        Some("#"),
        None,
        0,
        0,
        None,
        None,
        true,
        None,
    )
    .unwrap();
    assert_eq!(names, ["x", "y", "z"]);
    assert_eq!(b.shape(), [2, 3]);
    // usecols selects names too
    let (c, names) = np::io::genfromtxt::<f64>(
        tmp("gn3.csv", "a,b c,d\n1,,3\n4,5,6\n"),
        Some("#"),
        Some(","),
        0,
        0,
        None,
        Some(&[0, 2]),
        true,
        None,
    )
    .unwrap();
    assert_eq!(names, ["a", "d"]);
    same(&c.to_vec(), &[1., 3., 4., 6.]);
}

#[test]
fn genfromtxt_ragged_rows() {
    let e = np::io::genfromtxt::<f64>(
        tmp("gr.txt", "1 2\n3 4\n5\n"),
        Some("#"),
        None,
        0,
        0,
        None,
        None,
        false,
        None,
    )
    .unwrap_err();
    assert_eq!(
        e.to_string(),
        "Some errors were detected !\n    Line #3 (got 1 columns instead of 2)"
    );
    assert!(matches!(e, np::Error::Parse(_)));
}
