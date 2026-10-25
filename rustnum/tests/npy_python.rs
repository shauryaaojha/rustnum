//! Round-trip tests of `.npy` files against real NumPy, driven through `python -c`.
//!
//! Every test skips (with a message on stderr) when `python` or `numpy` is missing.

use rustnum as np;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Runs a Python snippet; `None` when python/numpy is unavailable, panics if the snippet fails.
fn run_python(script: &str) -> Option<()> {
    let probe = Command::new("python")
        .args(["-c", "import numpy"])
        .output()
        .ok()?;
    if !probe.status.success() {
        eprintln!("SKIP: python/numpy not available");
        return None;
    }
    let out = Command::new("python")
        .args(["-c", script])
        .output()
        .expect("failed to run python");
    assert!(
        out.status.success(),
        "Python failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    Some(())
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rustnum_npy_py_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Python source for the path, as a raw string literal.
fn py(path: &Path) -> String {
    format!("r'{}'", path.display())
}

/// Rust writes `.npy`, Python reads and verifies.
#[test]
fn rust_write_python_read() {
    let dir = scratch("w");
    let a = np::arange(1.0_f64, 7.0, 1.0).reshape(&[2, 3]);
    let path = dir.join("from_rust.npy");
    np::io::save_npy(&path, &a).unwrap();
    let script = format!(
        "import numpy as np
a = np.load({})
assert a.dtype == np.float64, a.dtype
assert a.shape == (2, 3), a.shape
assert np.array_equal(a, np.arange(1.0, 7.0).reshape(2, 3)), a",
        py(&path)
    );
    run_python(&script);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Python writes `.npy`, Rust reads.
#[test]
fn python_write_rust_read() {
    let dir = scratch("r");
    let path = dir.join("from_python.npy");
    let script = format!(
        "import numpy as np
np.save({}, np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int64))",
        py(&path)
    );
    if run_python(&script).is_none() {
        return;
    }
    let a: np::Array<i64> = np::io::load_npy(&path).unwrap();
    assert_eq!(a.shape(), &[3, 2]);
    assert_eq!(a.to_vec(), vec![10i64, 20, 30, 40, 50, 60]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Rust writes, Python modifies and writes back, Rust reads.
#[test]
fn full_round_trip() {
    let dir = scratch("rt");
    let original = np::Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let (p1, p2) = (dir.join("step1.npy"), dir.join("step2.npy"));
    np::io::save_npy(&p1, &original).unwrap();
    let script = format!(
        "import numpy as np
np.save({}, np.load({}) * 2)",
        py(&p2),
        py(&p1)
    );
    if run_python(&script).is_none() {
        return;
    }
    let result: np::Array<f64> = np::io::load_npy(&p2).unwrap();
    assert_eq!(result, &original * 2.0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Rust writes every dtype (plus 0-d, empty and non-contiguous arrays); Python checks dtype,
/// shape and values.
#[test]
fn rust_dtypes_read_by_numpy() {
    let dir = scratch("dt");
    let m = np::arange(0, 6, 1);
    np::io::save_npy(dir.join("f64.npy"), &np::array![[1.5, -2.5], [3.0, 4.0]]).unwrap();
    np::io::save_npy(dir.join("f32.npy"), &np::array![1.0f32, 2.5, 3.75]).unwrap();
    np::io::save_npy(dir.join("i64.npy"), &np::array![i64::MIN, 0, i64::MAX]).unwrap();
    np::io::save_npy(dir.join("i32.npy"), &np::array![i32::MIN, 0, i32::MAX]).unwrap();
    np::io::save_npy(dir.join("u8.npy"), &np::array![0u8, 127, 255]).unwrap();
    np::io::save_npy(dir.join("u64.npy"), &np::array![0usize, 1 << 40]).unwrap();
    np::io::save_npy(
        dir.join("bool.npy"),
        &np::array![[true, false], [false, true]],
    )
    .unwrap();
    np::io::save_npy(dir.join("zero_d.npy"), &np::Array::scalar(7.5f64)).unwrap();
    np::io::save_npy(dir.join("empty.npy"), &np::zeros::<f64>(&[3, 0])).unwrap();
    np::io::save_npy(dir.join("t.npy"), &m.reshape(&[2, 3]).t()).unwrap();
    let script = format!(
        "import numpy as np
d = {}
def load(n): return np.load(d + '/' + n + '.npy')
a = load('f64'); assert a.dtype == np.float64 and a.tolist() == [[1.5, -2.5], [3.0, 4.0]]
a = load('f32'); assert a.dtype == np.float32 and a.tolist() == [1.0, 2.5, 3.75]
a = load('i64'); assert a.dtype == np.int64 and a.tolist() == [-2**63, 0, 2**63 - 1]
a = load('i32'); assert a.dtype == np.int32 and a.tolist() == [-2**31, 0, 2**31 - 1]
a = load('u8'); assert a.dtype == np.uint8 and a.tolist() == [0, 127, 255]
a = load('u64'); assert a.dtype == np.uint64 and a.tolist() == [0, 2**40]
a = load('bool'); assert a.dtype == np.bool_ and a.tolist() == [[True, False], [False, True]]
a = load('zero_d'); assert a.shape == () and a.item() == 7.5
a = load('empty'); assert a.shape == (3, 0) and a.dtype == np.float64
a = load('t'); assert a.shape == (3, 2) and a.tolist() == [[0, 3], [1, 4], [2, 5]]
assert load('t').flags.c_contiguous",
        py(&dir)
    );
    run_python(&script);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Python writes every dtype (plus 0-d, Fortran order and 3-d); Rust reads.
#[test]
fn numpy_dtypes_read_by_rust() {
    let dir = scratch("dt2");
    let script = format!(
        "import numpy as np
d = {}
np.save(d + '/f64.npy', np.array([[1.5, -2.5], [3.0, 4.0]]))
np.save(d + '/f32.npy', np.array([1.0, 2.5, 3.75], dtype=np.float32))
np.save(d + '/i64.npy', np.array([-2**63, 0, 2**63 - 1], dtype=np.int64))
np.save(d + '/i32.npy', np.array([-2**31, 0, 2**31 - 1], dtype=np.int32))
np.save(d + '/u8.npy', np.array([0, 127, 255], dtype=np.uint8))
np.save(d + '/u64.npy', np.array([0, 2**40], dtype=np.uint64))
np.save(d + '/bool.npy', np.array([[True, False], [False, True]]))
np.save(d + '/zero_d.npy', np.array(7.5))
np.save(d + '/f3.npy', np.asfortranarray(np.arange(24, dtype=np.float64).reshape(2, 3, 4)))
np.save(d + '/slice.npy', np.arange(12, dtype=np.int64).reshape(3, 4)[::2, ::-1])",
        py(&dir)
    );
    if run_python(&script).is_none() {
        return;
    }
    let p = |n: &str| dir.join(format!("{n}.npy"));
    assert_eq!(
        np::io::load_npy::<f64>(p("f64")).unwrap(),
        np::array![[1.5, -2.5], [3.0, 4.0]]
    );
    assert_eq!(
        np::io::load_npy::<f32>(p("f32")).unwrap(),
        np::array![1.0f32, 2.5, 3.75]
    );
    assert_eq!(
        np::io::load_npy::<i64>(p("i64")).unwrap(),
        np::array![i64::MIN, 0, i64::MAX]
    );
    assert_eq!(
        np::io::load_npy::<i32>(p("i32")).unwrap(),
        np::array![i32::MIN, 0, i32::MAX]
    );
    assert_eq!(
        np::io::load_npy::<u8>(p("u8")).unwrap(),
        np::array![0u8, 127, 255]
    );
    assert_eq!(
        np::io::load_npy::<usize>(p("u64")).unwrap(),
        np::array![0usize, 1 << 40]
    );
    assert_eq!(
        np::io::load_npy::<bool>(p("bool")).unwrap(),
        np::array![[true, false], [false, true]]
    );
    let z = np::io::load_npy::<f64>(p("zero_d")).unwrap();
    assert_eq!((z.ndim(), z.item()), (0, 7.5));
    let f3 = np::io::load_npy::<f64>(p("f3")).unwrap();
    assert_eq!(f3, np::arange(0.0, 24.0, 1.0).reshape(&[2, 3, 4]));
    assert_eq!(
        np::io::load_npy::<i64>(p("slice")).unwrap(),
        np::array![[3i64, 2, 1, 0], [11, 10, 9, 8]]
    );
    // A dtype that does not match is an error, not a cast.
    assert!(matches!(
        np::io::load_npy::<f64>(p("i64")),
        Err(np::Error::DType { .. })
    ));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Fortran-order file from Python.
#[test]
fn fortran_order_from_python() {
    let dir = scratch("f");
    let path = dir.join("fortran.npy");
    let script = format!(
        "import numpy as np
a = np.asfortranarray(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
assert a.flags.f_contiguous and not a.flags.c_contiguous
np.save({}, a)",
        py(&path)
    );
    if run_python(&script).is_none() {
        return;
    }
    let a: np::Array<f64> = np::io::load_npy(&path).unwrap();
    assert_eq!(a.shape(), &[2, 3]);
    assert_eq!(a.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let _ = std::fs::remove_dir_all(&dir);
}
