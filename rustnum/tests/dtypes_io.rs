//! `.npy` / `.npz` round trips of the v0.2 dtypes against real NumPy, driven through `python -c`.
//!
//! Every NumPy-facing test skips (with a message on stderr) when `python` or `numpy` is missing;
//! the pure-Rust round trips always run.

use rustnum as np;
use rustnum::Complex;
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
    let dir = std::env::temp_dir().join(format!("rustnum_dtypes_io_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn py(path: &Path) -> String {
    format!("r'{}'", path.display())
}

#[test]
fn complex_npy_round_trip() {
    let a = np::array![
        [Complex::new(1.0f64, -2.0), Complex::new(0.5, 0.25)],
        [
            Complex::new(f64::INFINITY, 0.0),
            Complex::new(-0.0, f64::NAN)
        ]
    ];
    let bytes = np::io::to_npy_bytes(&a);
    assert!(String::from_utf8_lossy(&bytes[..80]).contains("'<c16'"));
    let b: np::Array<Complex<f64>> = np::io::from_npy_bytes(&bytes).unwrap();
    assert_eq!(b.shape(), [2, 2]);
    assert_eq!(b.to_vec()[0], Complex::new(1.0, -2.0));
    assert!(b.to_vec()[3].im.is_nan());
    let c = np::array![Complex::new(1.5f32, 2.5)];
    let bytes = np::io::to_npy_bytes(&c);
    assert!(String::from_utf8_lossy(&bytes[..80]).contains("'<c8'"));
    assert_eq!(np::io::from_npy_bytes::<Complex<f32>>(&bytes).unwrap(), c);
    // Asking for the other width is a dtype error, never a silent cast.
    assert!(matches!(
        np::io::from_npy_bytes::<Complex<f64>>(&bytes),
        Err(np::Error::DType { .. })
    ));
}

#[test]
fn complex_npy_files_are_read_and_written_by_numpy() {
    let dir = scratch("complex");
    let path = dir.join("from_python.npy");
    let script = format!(
        "import numpy as np
np.save({}, np.array([[1+2j, 3-4j], [0.5j, -1.5]], dtype=np.complex128))",
        py(&path)
    );
    if run_python(&script).is_none() {
        return;
    }
    let a: np::Array<Complex<f64>> = np::io::load_npy(&path).unwrap();
    assert_eq!(a.shape(), [2, 2]);
    assert_eq!(
        a.to_vec(),
        [
            Complex::new(1.0, 2.0),
            Complex::new(3.0, -4.0),
            Complex::new(0.0, 0.5),
            Complex::new(-1.5, 0.0)
        ]
    );
    let out = dir.join("from_rust.npy");
    np::io::save_npy(&out, &a).unwrap();
    let out32 = dir.join("from_rust32.npy");
    np::io::save_npy(&out32, &a.astype::<Complex<f32>>()).unwrap();
    let npz = dir.join("both.npz");
    np::io::savez(&npz, &[("z", &a)]).unwrap();
    let script = format!(
        "import numpy as np
z = np.load({})
assert z.dtype == np.complex128 and z.shape == (2, 2), (z.dtype, z.shape)
assert np.array_equal(z, np.array([[1+2j, 3-4j], [0.5j, -1.5]]))
z32 = np.load({})
assert z32.dtype == np.complex64
assert np.array_equal(z32, z.astype(np.complex64))
n = np.load({})
assert n['z'].dtype == np.complex128 and np.array_equal(n['z'], z)",
        py(&out),
        py(&out32),
        py(&npz)
    );
    run_python(&script);
    let _ = std::fs::remove_dir_all(&dir);
}
