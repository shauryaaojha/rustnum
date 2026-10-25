//! Integration tests for the public `np::io` API: `.npy`, `.npz` and CSV.

use rustnum as np;
use std::path::PathBuf;
use std::process::Command;

/// A scratch directory that is removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir =
            std::env::temp_dir().join(format!("rustnum_io_test_{}_{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Runs `python -c script args...`; `None` when python or numpy is unavailable.
fn python(script: &str, args: &[&str]) -> Option<std::process::Output> {
    let probe = Command::new("python")
        .args(["-c", "import numpy"])
        .output()
        .ok()?;
    if !probe.status.success() {
        return None;
    }
    let out = Command::new("python")
        .args(["-c", script])
        .args(args)
        .output()
        .ok()?;
    assert!(
        out.status.success(),
        "python failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    Some(out)
}

// ---------------------------------------------------------------------------
// .npy
// ---------------------------------------------------------------------------

/// Round-trips 0-d, 1-d, 2-d, 3-d, empty and non-contiguous arrays of one dtype, both through
/// bytes and through a file.
macro_rules! npy_roundtrip {
    ($name:ident, $t:ty, $vals:expr) => {
        #[test]
        fn $name() {
            let dir = Scratch::new(stringify!($name));
            let vals: Vec<$t> = $vals;
            let n = vals.len();
            let cases: Vec<np::Array<$t>> = vec![
                np::Array::scalar(vals[1]),
                np::Array::from_vec(vals.clone(), &[n]),
                np::Array::from_vec(vals.clone(), &[2, n / 2]),
                np::Array::from_vec(vals.clone(), &[1, 2, n / 2]),
                np::Array::from_vec(vec![], &[0]),
                np::Array::from_vec(vec![], &[3, 0]),
                np::Array::from_vec(vals.clone(), &[2, n / 2]).t(),
                np::Array::from_vec(vals.clone(), &[2, n / 2]).flip_all(),
            ];
            for (i, a) in cases.iter().enumerate() {
                let bytes = np::io::to_npy_bytes(a);
                assert_eq!(&bytes[..6], b"\x93NUMPY");
                let back: np::Array<$t> = np::io::from_npy_bytes(&bytes).unwrap();
                assert_eq!(&back, a, "{} case {i} via bytes", stringify!($t));
                assert_eq!(back.shape(), a.shape());

                let path = dir.path(&format!("{i}.npy"));
                np::io::save_npy(&path, a).unwrap();
                let loaded: np::Array<$t> = np::io::load_npy(&path).unwrap();
                assert_eq!(&loaded, a, "{} case {i} via file", stringify!($t));
                assert!(loaded.is_contiguous());
            }
        }
    };
}

npy_roundtrip!(
    npy_roundtrip_f64,
    f64,
    vec![-1.5, 0.0, 2.25, 1e300, f64::MIN_POSITIVE, f64::INFINITY]
);
npy_roundtrip!(
    npy_roundtrip_f32,
    f32,
    vec![-1.5, 0.0, 2.25, 1e30, f32::MIN_POSITIVE, f32::NEG_INFINITY]
);
npy_roundtrip!(
    npy_roundtrip_i64,
    i64,
    vec![i64::MIN, -1, 0, 1, 42, i64::MAX]
);
npy_roundtrip!(
    npy_roundtrip_i32,
    i32,
    vec![i32::MIN, -1, 0, 1, 42, i32::MAX]
);
npy_roundtrip!(npy_roundtrip_u8, u8, vec![0, 1, 127, 128, 254, 255]);
npy_roundtrip!(
    npy_roundtrip_usize,
    usize,
    vec![0, 1, 2, 1 << 40, 4, usize::MAX >> 1]
);
npy_roundtrip!(
    npy_roundtrip_bool,
    bool,
    vec![true, false, true, true, false, false]
);

#[test]
fn npy_header_is_64_byte_aligned() {
    for shape in [vec![], vec![3], vec![2, 5], vec![1, 2, 3, 4]] {
        let n: usize = shape.iter().product();
        let a = np::Array::from_vec(vec![1.0_f64; n], &shape);
        let bytes = np::io::to_npy_bytes(&a);
        let header_len = bytes.len() - 8 * n;
        assert_eq!(header_len % 64, 0, "shape {shape:?}");
    }
}

#[test]
fn npy_nan_survives_a_roundtrip() {
    let a = np::array![1.0, f64::NAN, -0.0];
    let b: np::Array<f64> = np::io::from_npy_bytes(&np::io::to_npy_bytes(&a)).unwrap();
    assert_eq!(b.shape(), &[3]);
    assert!(b.get(&[1]).is_nan());
    assert!(b.get(&[2]).is_sign_negative());
}

/// Builds a v1.0 `.npy` file by hand, so the reader is tested without the writer.
fn raw_npy(descr: &str, fortran: bool, shape: &str, data: &[u8]) -> Vec<u8> {
    let dict = format!(
        "{{'descr': '{descr}', 'fortran_order': {}, 'shape': {shape}, }}",
        if fortran { "True" } else { "False" }
    );
    let mut header = dict.into_bytes();
    while !(10 + header.len() + 1).is_multiple_of(64) {
        header.push(b' ');
    }
    header.push(b'\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(data);
    out
}

#[test]
fn npy_reads_fortran_order_input() {
    // The 2x3 array [[1, 2, 3], [4, 5, 6]] stored column by column.
    let col_major: Vec<u8> = [1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let bytes = raw_npy("<f8", true, "(2, 3)", &col_major);
    let a: np::Array<f64> = np::io::from_npy_bytes(&bytes).unwrap();
    assert_eq!(a, np::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert!(a.is_contiguous());

    // 3-d: shape (2, 3, 2), element (i, j, k) = 100 i + 10 j + k, first index fastest.
    let mut data = Vec::new();
    for k in 0..2_i32 {
        for j in 0..3_i32 {
            for i in 0..2_i32 {
                data.extend_from_slice(&(100 * i + 10 * j + k).to_le_bytes());
            }
        }
    }
    let bytes = raw_npy("<i4", true, "(2, 3, 2)", &data);
    let a: np::Array<i32> = np::io::from_npy_bytes(&bytes).unwrap();
    assert_eq!(a.shape(), &[2, 3, 2]);
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..2 {
                assert_eq!(a.get(&[i, j, k]), (100 * i + 10 * j + k) as i32);
            }
        }
    }
}

#[test]
fn npy_reads_zero_d_and_trailing_comma_shapes() {
    let bytes = raw_npy("<f8", false, "()", &7.5_f64.to_le_bytes());
    let a: np::Array<f64> = np::io::from_npy_bytes(&bytes).unwrap();
    assert_eq!(a.ndim(), 0);
    assert_eq!(a.item(), 7.5);
    let bytes = raw_npy("|u1", false, "(3,)", &[1, 2, 3]);
    let a: np::Array<u8> = np::io::from_npy_bytes(&bytes).unwrap();
    assert_eq!(a, np::array![1u8, 2, 3]);
}

#[test]
fn npy_reads_version_2_headers() {
    let dict = "{'descr': '<i8', 'fortran_order': False, 'shape': (2,), }";
    let mut header = dict.as_bytes().to_vec();
    while !(12 + header.len() + 1).is_multiple_of(64) {
        header.push(b' ');
    }
    header.push(b'\n');
    let mut bytes = b"\x93NUMPY\x02\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&5_i64.to_le_bytes());
    bytes.extend_from_slice(&(-6_i64).to_le_bytes());
    let a: np::Array<i64> = np::io::from_npy_bytes(&bytes).unwrap();
    assert_eq!(a, np::array![5_i64, -6]);
}

#[test]
fn npy_dtype_mismatch_is_a_dtype_error() {
    fn check<W: np::Element, R: np::Element>(w: np::Array<W>) {
        let bytes = np::io::to_npy_bytes(&w);
        match np::io::from_npy_bytes::<R>(&bytes) {
            Err(np::Error::DType { expected, found }) => {
                assert_eq!(expected, R::DTYPE);
                assert_eq!(found, W::DTYPE);
            }
            other => panic!("expected DType error, got {other:?}"),
        }
    }
    check::<f64, f32>(np::array![1.0]);
    check::<f64, i64>(np::array![1.0]);
    check::<f32, f64>(np::array![1.0_f32]);
    check::<i64, f64>(np::array![1_i64]);
    check::<i64, i32>(np::array![1_i64]);
    check::<i32, i64>(np::array![1_i32]);
    check::<u8, bool>(np::array![1_u8]);
    check::<bool, u8>(np::array![true]);
    check::<usize, i64>(np::array![1_usize]);
}

#[test]
fn npy_file_dtype_mismatch_is_a_dtype_error() {
    let dir = Scratch::new("npy_mismatch");
    let path = dir.path("a.npy");
    np::io::save_npy(&path, &np::array![1.0, 2.0]).unwrap();
    assert!(matches!(
        np::io::load_npy::<i32>(&path),
        Err(np::Error::DType { .. })
    ));
    // Big-endian data is a different dtype, not silently byte-swapped.
    let bytes = raw_npy(">f8", false, "(1,)", &1.0_f64.to_be_bytes());
    assert!(matches!(
        np::io::from_npy_bytes::<f64>(&bytes),
        Err(np::Error::DType { .. })
    ));
}

#[test]
fn npy_corrupt_input_is_an_error_not_a_panic() {
    let good = np::io::to_npy_bytes(&np::arange(0.0, 20.0, 1.0));
    for cut in [0, 3, 6, 9, 10, 40, good.len() - 1] {
        assert!(
            np::io::from_npy_bytes::<f64>(&good[..cut]).is_err(),
            "truncation at {cut} was accepted"
        );
    }
    let mut bad_magic = good.clone();
    bad_magic[1] = b'X';
    assert!(matches!(
        np::io::from_npy_bytes::<f64>(&bad_magic),
        Err(np::Error::Format(_))
    ));
    let mut bad_version = good.clone();
    bad_version[6] = 9;
    assert!(matches!(
        np::io::from_npy_bytes::<f64>(&bad_version),
        Err(np::Error::Format(_))
    ));
    // A shape whose element count overflows must not panic or allocate.
    let huge = raw_npy("<f8", false, "(4294967296, 4294967296, 4294967296)", &[]);
    assert!(np::io::from_npy_bytes::<f64>(&huge).is_err());
    let missing = np::io::load_npy::<f64>("definitely/not/a/file.npy");
    assert!(matches!(missing, Err(np::Error::Io(_))));
}

#[test]
fn npy_large_array_roundtrip() {
    let a = np::arange(0.0, 100_000.0, 1.0).reshape(&[1000, 100]);
    let b: np::Array<f64> = np::io::from_npy_bytes(&np::io::to_npy_bytes(&a)).unwrap();
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

#[test]
fn csv_roundtrip_f64_exact() {
    let dir = Scratch::new("csv_f64");
    let path = dir.path("a.csv");
    let a = np::array![[0.1, -2.5e-300, 1e300], [1.0 / 3.0, 100.0, -0.0]];
    np::io::save_csv(&path, &a, ',').unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(text.lines().count(), 2);
    let b: np::Array<f64> = np::io::load_csv(&path, ',', 0).unwrap();
    assert_eq!(a, b, "Display formatting must round-trip floats exactly");
}

#[test]
fn csv_roundtrip_other_types_and_delimiters() {
    let dir = Scratch::new("csv_types");
    let ints = np::array![[1_i64, -2, 3], [4, 5, i64::MAX]];
    let path = dir.path("i.tsv");
    np::io::save_csv(&path, &ints, '\t').unwrap();
    assert!(std::fs::read_to_string(&path).unwrap().contains('\t'));
    assert_eq!(np::io::load_csv::<i64>(&path, '\t', 0).unwrap(), ints);

    let f32s = np::array![[0.5_f32, 1.25], [-3.0, 8.0]];
    let path = dir.path("f.csv");
    np::io::save_csv(&path, &f32s, ';').unwrap();
    assert_eq!(np::io::load_csv::<f32>(&path, ';', 0).unwrap(), f32s);

    let bools = np::array![[true, false], [false, true]];
    let path = dir.path("b.csv");
    np::io::save_csv(&path, &bools, ',').unwrap();
    assert_eq!(np::io::load_csv::<bool>(&path, ',', 0).unwrap(), bools);
}

#[test]
fn csv_one_dimensional_and_single_row_follow_loadtxt() {
    let dir = Scratch::new("csv_1d");
    let path = dir.path("v.csv");
    let v = np::array![1.5, 2.5, 3.5];
    np::io::save_csv(&path, &v, ',').unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "1.5\n2.5\n3.5\n");
    assert_eq!(np::io::load_csv::<f64>(&path, ',', 0).unwrap(), v);
    // A single row also loads 1-d, like np.loadtxt.
    let row = np::array![[1.0, 2.0, 3.0]];
    np::io::save_csv(&path, &row, ',').unwrap();
    assert_eq!(
        np::io::load_csv::<f64>(&path, ',', 0).unwrap(),
        np::array![1.0, 2.0, 3.0]
    );
}

#[test]
fn csv_header_comments_and_blank_lines() {
    let dir = Scratch::new("csv_header");
    let path = dir.path("h.csv");
    std::fs::write(
        &path,
        "name,value\n# a comment\n1,2\n\n3, 4  # trailing\r\n5,6\n",
    )
    .unwrap();
    let a: np::Array<i32> = np::io::load_csv(&path, ',', 1).unwrap();
    assert_eq!(a, np::array![[1, 2], [3, 4], [5, 6]]);
    // Without skipping, the header is not numeric.
    assert!(matches!(
        np::io::load_csv::<f64>(&path, ',', 0),
        Err(np::Error::Parse(_))
    ));
    // Space delimiter means any whitespace run.
    std::fs::write(&path, "1   2\t3\n4 5 6\n").unwrap();
    let b: np::Array<f64> = np::io::load_csv(&path, ' ', 0).unwrap();
    assert_eq!(b.shape(), &[2, 3]);
}

#[test]
fn csv_errors() {
    let dir = Scratch::new("csv_err");
    let path = dir.path("e.csv");
    std::fs::write(&path, "1,2,3\n4,5\n").unwrap();
    match np::io::load_csv::<f64>(&path, ',', 0) {
        Err(np::Error::Parse(msg)) => assert!(msg.contains("line 2"), "{msg}"),
        other => panic!("expected ragged-row error, got {other:?}"),
    }
    std::fs::write(&path, "1,2\n3,oops\n").unwrap();
    match np::io::load_csv::<f64>(&path, ',', 0) {
        Err(np::Error::Parse(msg)) => assert!(msg.contains("oops"), "{msg}"),
        other => panic!("expected parse error, got {other:?}"),
    }
    // Floats are not integers.
    std::fs::write(&path, "1.5,2\n").unwrap();
    assert!(np::io::load_csv::<i64>(&path, ',', 0).is_err());
    // Only 1-d and 2-d arrays can be written.
    let cube = np::zeros::<f64>(&[2, 2, 2]);
    assert!(matches!(
        np::io::save_csv(dir.path("c.csv"), &cube, ','),
        Err(np::Error::Format(_))
    ));
    assert!(np::io::save_csv(dir.path("s.csv"), &np::Array::scalar(1.0), ',').is_err());
    assert!(matches!(
        np::io::load_csv::<f64>("no/such/file.csv", ',', 0),
        Err(np::Error::Io(_))
    ));
}

#[test]
fn csv_matches_numpy_loadtxt() {
    let dir = Scratch::new("csv_numpy");
    let path = dir.path("n.csv");
    let a = np::array![[1.5, 2.0, -3.25], [4.0, 5.5, 6.0]];
    np::io::save_csv(&path, &a, ',').unwrap();
    let script = "import sys, numpy as np
a = np.loadtxt(sys.argv[1], delimiter=',')
assert a.shape == (2, 3) and a[0, 2] == -3.25 and a.sum() == 15.75, a
np.savetxt(sys.argv[2], a * 2, delimiter=',')";
    let out = dir.path("n2.csv");
    if python(script, &[path.to_str().unwrap(), out.to_str().unwrap()]).is_none() {
        eprintln!("SKIP: python/numpy not available");
        return;
    }
    let b: np::Array<f64> = np::io::load_csv(&out, ',', 0).unwrap();
    assert_eq!(b, &a * 2.0);
}

// ---------------------------------------------------------------------------
// .npz
// ---------------------------------------------------------------------------

#[test]
fn npz_roundtrip_stored_and_compressed() {
    let dir = Scratch::new("npz_rt");
    let a = np::arange(0.0, 12.0, 1.0).reshape(&[3, 4]);
    let b = np::Array::scalar(2.5_f64);
    let c = np::zeros::<f64>(&[2, 0]);
    let d = np::Array::from_vec(vec![0.5; 3000], &[30, 100]).t();
    for compressed in [false, true] {
        let path = dir.path(if compressed { "c.npz" } else { "s.npz" });
        let members = [("a", &a), ("b", &b), ("c", &c), ("d", &d)];
        if compressed {
            np::io::savez_compressed(&path, &members).unwrap();
        } else {
            np::io::savez(&path, &members).unwrap();
        }
        let loaded = np::io::load_npz::<f64>(&path).unwrap();
        let names: Vec<&str> = loaded.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["a", "b", "c", "d"], "archive order");
        for ((_, got), (_, want)) in loaded.iter().zip(members) {
            assert_eq!(got, want);
        }

        let file = np::io::NpzFile::open(&path).unwrap();
        assert_eq!(file.keys(), ["a", "b", "c", "d"]);
        assert_eq!(file.get::<f64>("a").unwrap(), a);
        assert_eq!(
            file.get::<f64>("a.npy").unwrap(),
            a,
            "name with .npy suffix"
        );
        assert_eq!(file.get::<f64>("d").unwrap().shape(), &[100, 30]);
        if compressed {
            let size = std::fs::metadata(&path).unwrap().len();
            assert!(
                size < 24_000 / 4,
                "constant data should compress, got {size} bytes"
            );
        }
    }
}

#[test]
fn npz_roundtrip_every_dtype() {
    let dir = Scratch::new("npz_dtypes");
    macro_rules! one {
        ($t:ty, $vals:expr) => {{
            let a: np::Array<$t> = np::Array::from_vec($vals, &[2, 2]);
            let path = dir.path(&format!("{}.npz", stringify!($t)));
            np::io::savez_compressed(&path, &[("x", &a), ("y", &a)]).unwrap();
            let got = np::io::load_npz::<$t>(&path).unwrap();
            assert_eq!(got.len(), 2);
            assert_eq!(got[0].1, a);
            assert_eq!(
                np::io::NpzFile::open(&path)
                    .unwrap()
                    .get::<$t>("y")
                    .unwrap(),
                a
            );
        }};
    }
    one!(f64, vec![1.5, -2.5, 3.0, 4.0]);
    one!(f32, vec![1.5, -2.5, 3.0, 4.0]);
    one!(i64, vec![i64::MIN, -1, 0, i64::MAX]);
    one!(i32, vec![i32::MIN, -1, 0, i32::MAX]);
    one!(u8, vec![0, 1, 254, 255]);
    one!(usize, vec![0, 1, 2, 1 << 40]);
    one!(bool, vec![true, false, false, true]);
}

#[test]
fn npz_dtype_mismatch_is_a_dtype_error() {
    let dir = Scratch::new("npz_dtype");
    let path = dir.path("i.npz");
    let ids = np::array![1_i64, 2, 3];
    np::io::savez(&path, &[("ids", &ids)]).unwrap();
    assert!(matches!(
        np::io::load_npz::<f64>(&path),
        Err(np::Error::DType { .. })
    ));
    let file = np::io::NpzFile::open(&path).unwrap();
    assert!(matches!(
        file.get::<i32>("ids"),
        Err(np::Error::DType { .. })
    ));
    assert_eq!(file.get::<i64>("ids").unwrap(), ids);
}

#[test]
fn npz_name_and_file_errors() {
    let dir = Scratch::new("npz_err");
    let a = np::array![1.0];
    assert!(matches!(
        np::io::savez(dir.path("dup.npz"), &[("a", &a), ("a", &a)]),
        Err(np::Error::Format(_))
    ));
    assert!(matches!(
        np::io::savez(dir.path("empty.npz"), &[("", &a)]),
        Err(np::Error::Format(_))
    ));
    let none: [(&str, &np::Array<f64>); 0] = [];
    np::io::savez(dir.path("none.npz"), &none).unwrap();
    assert!(
        np::io::load_npz::<f64>(dir.path("none.npz"))
            .unwrap()
            .is_empty()
    );

    np::io::savez(dir.path("ok.npz"), &[("a", &a)]).unwrap();
    let file = np::io::NpzFile::open(dir.path("ok.npz")).unwrap();
    assert!(matches!(file.get::<f64>("zzz"), Err(np::Error::Format(_))));

    assert!(matches!(
        np::io::NpzFile::open(dir.path("missing.npz")),
        Err(np::Error::Io(_))
    ));
    std::fs::write(dir.path("junk.npz"), b"this is not a zip file at all").unwrap();
    assert!(matches!(
        np::io::NpzFile::open(dir.path("junk.npz")),
        Err(np::Error::Format(_))
    ));
    assert!(np::io::load_npz::<f64>(dir.path("junk.npz")).is_err());
}

#[test]
fn npz_detects_corruption() {
    let dir = Scratch::new("npz_corrupt");
    let path = dir.path("c.npz");
    let a = np::arange(0.0, 100.0, 1.0);
    np::io::savez(&path, &[("a", &a)]).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    // Flip a data byte inside the stored member; the CRC must catch it.
    let at = 200;
    bytes[at] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();
    let file = np::io::NpzFile::open(&path).unwrap();
    assert!(matches!(file.get::<f64>("a"), Err(np::Error::Format(_))));
    assert!(np::io::load_npz::<f64>(&path).is_err());
    // Truncation never panics.
    for cut in [0, 10, bytes.len() / 2, bytes.len() - 1] {
        std::fs::write(&path, &bytes[..cut]).unwrap();
        let _ = np::io::load_npz::<f64>(&path);
        let _ = np::io::NpzFile::open(&path).map(|f| f.get::<f64>("a"));
    }
}

#[test]
fn npz_reads_a_real_numpy_file() {
    let dir = Scratch::new("npz_numpy");
    let script = "import sys, numpy as np
d = sys.argv[1]
np.savez(d + '/plain.npz', f=np.arange(6, dtype=np.float64).reshape(2, 3),
         i=np.array([1, -2, 3], dtype=np.int64), b=np.array([True, False]),
         u=np.array([0, 255], dtype=np.uint8), s=np.array(3.5), g=np.zeros((0, 4), dtype=np.float32))
np.savez_compressed(d + '/packed.npz', f=np.arange(1000, dtype=np.float64).reshape(10, 100),
         z=np.zeros(5000, dtype=np.int32))
np.savez(d + '/fortran.npz', m=np.asfortranarray(np.arange(6, dtype=np.float64).reshape(2, 3)))";
    if python(script, &[dir.0.to_str().unwrap()]).is_none() {
        eprintln!("SKIP: python/numpy not available");
        return;
    }
    let plain = np::io::NpzFile::open(dir.path("plain.npz")).unwrap();
    let mut keys = plain.keys();
    keys.sort();
    assert_eq!(keys, ["b", "f", "g", "i", "s", "u"]);
    assert_eq!(
        plain.get::<f64>("f").unwrap(),
        np::arange(0.0, 6.0, 1.0).reshape(&[2, 3])
    );
    assert_eq!(plain.get::<i64>("i").unwrap(), np::array![1_i64, -2, 3]);
    assert_eq!(plain.get::<bool>("b").unwrap(), np::array![true, false]);
    assert_eq!(plain.get::<u8>("u").unwrap(), np::array![0_u8, 255]);
    let s = plain.get::<f64>("s").unwrap();
    assert_eq!((s.ndim(), s.item()), (0, 3.5));
    assert_eq!(plain.get::<f32>("g").unwrap().shape(), &[0, 4]);
    // Mixed dtypes in one file need the per-member API; the bulk loader rejects them.
    assert!(matches!(
        np::io::load_npz::<f64>(dir.path("plain.npz")),
        Err(np::Error::DType { .. })
    ));

    let packed = np::io::NpzFile::open(dir.path("packed.npz")).unwrap();
    let f = packed.get::<f64>("f").unwrap();
    assert_eq!(f.shape(), &[10, 100]);
    assert_eq!(f.sum(), 499_500.0);
    assert_eq!(packed.get::<i32>("z").unwrap(), np::zeros::<i32>(&[5000]));

    let fortran = np::io::NpzFile::open(dir.path("fortran.npz")).unwrap();
    assert_eq!(
        fortran.get::<f64>("m").unwrap(),
        np::arange(0.0, 6.0, 1.0).reshape(&[2, 3])
    );
}

#[test]
fn numpy_reads_our_npz_files() {
    let dir = Scratch::new("npz_numpy_read");
    let a = np::arange(0.0, 12.0, 1.0).reshape(&[3, 4]);
    let b = np::zeros::<f64>(&[2000]);
    np::io::savez(dir.path("s.npz"), &[("a", &a), ("b", &b)]).unwrap();
    np::io::savez_compressed(dir.path("c.npz"), &[("a", &a), ("b", &b)]).unwrap();
    let script = "import sys, numpy as np
for name in ('s', 'c'):
    f = np.load(sys.argv[1] + '/' + name + '.npz')
    assert sorted(f.files) == ['a', 'b'], f.files
    assert f['a'].shape == (3, 4) and f['a'].sum() == 66.0
    assert f['b'].shape == (2000,) and f['b'].sum() == 0.0";
    if python(script, &[dir.0.to_str().unwrap()]).is_none() {
        eprintln!("SKIP: python/numpy not available");
    }
}
