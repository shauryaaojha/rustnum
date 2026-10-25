//! `.npz` support: a pure-Rust ZIP + DEFLATE layer with no dependencies.
//!
//! An `.npz` file is a ZIP archive whose members are named `<key>.npy` and hold one `.npy`
//! blob each. This module works on raw bytes: it moves `(key, npy_bytes)` pairs in and out of
//! archives. Encoding arrays to `.npy` bytes is the job of `crate::io`; the array-level functions live in
//! `api` and are re-exported as `np::io::{savez, savez_compressed, load_npz, NpzFile}`.
//!
//! Layers, bottom to top:
//! - `crc32`: CRC-32 (IEEE).
//! - `inflate` / `deflate`: RFC 1951 decoder and encoder.
//! - `zip`: archive reader and writer (zip64, data descriptors, stored and deflate members).
//! - this module: key handling, whole-file helpers and the lazy [`NpzArchive`].

pub(crate) mod api;
mod crc32;
mod deflate;
mod inflate;
mod zip;

use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::path::Path;
use std::sync::Mutex;

pub use inflate::InflateError;
use zip::ZipEntry;

/// Everything that can go wrong reading or writing an `.npz` archive.
#[derive(Debug)]
pub enum NpzError {
    /// The data is not a ZIP archive (no end-of-central-directory record).
    NotAZip,
    /// The archive ends before a structure it announces.
    Truncated,
    /// A structural inconsistency in the archive.
    Corrupt(String),
    /// A valid ZIP feature this reader does not support (encryption, other methods, ...).
    Unsupported(String),
    /// A member's DEFLATE stream is invalid.
    Inflate(InflateError),
    /// A member's data does not match its stored CRC-32.
    Crc {
        /// Member name.
        name: String,
        /// CRC recorded in the archive.
        expected: u32,
        /// CRC of the data actually read.
        found: u32,
    },
    /// A member decompressed to a different size than recorded.
    SizeMismatch {
        /// Member name.
        name: String,
        /// Size recorded in the archive.
        expected: u64,
        /// Size actually read.
        found: u64,
    },
    /// No member with this key.
    MissingKey(String),
    /// A key that cannot be written (empty, duplicate, or too long).
    InvalidKey(String),
    /// An operating-system I/O error.
    Io(std::io::Error),
}

impl fmt::Display for NpzError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NpzError::NotAZip => f.write_str("not a zip archive"),
            NpzError::Truncated => f.write_str("zip archive is truncated"),
            NpzError::Corrupt(m) => write!(f, "corrupt zip archive: {m}"),
            NpzError::Unsupported(m) => write!(f, "unsupported zip feature: {m}"),
            NpzError::Inflate(e) => write!(f, "invalid deflate data: {e}"),
            NpzError::Crc {
                name,
                expected,
                found,
            } => write!(
                f,
                "CRC-32 mismatch in {name:?}: expected {expected:#010x}, found {found:#010x}"
            ),
            NpzError::SizeMismatch {
                name,
                expected,
                found,
            } => write!(
                f,
                "size mismatch in {name:?}: expected {expected} bytes, found {found}"
            ),
            NpzError::MissingKey(k) => write!(f, "{k} is not a file in the archive"),
            NpzError::InvalidKey(k) => write!(f, "invalid array name {k:?}"),
            NpzError::Io(e) => write!(f, "i/o error: {e}"),
        }
    }
}

impl std::error::Error for NpzError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NpzError::Inflate(e) => Some(e),
            NpzError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<InflateError> for NpzError {
    fn from(e: InflateError) -> Self {
        NpzError::Inflate(e)
    }
}

impl From<std::io::Error> for NpzError {
    fn from(e: std::io::Error) -> Self {
        NpzError::Io(e)
    }
}

const NPY_EXT: &str = ".npy";

fn key_of(member: &str) -> &str {
    member.strip_suffix(NPY_EXT).unwrap_or(member)
}

/// Build an `.npz` archive from `(key, npy_bytes)` pairs.
///
/// Member `key` is stored as `key.npy`, exactly like `np.savez`. With `compress`, members are
/// deflated (like `np.savez_compressed`) unless that does not make them smaller.
///
/// Fails with [`NpzError::InvalidKey`] for an empty, duplicate or over-long key.
pub fn npz_to_bytes(members: &[(String, Vec<u8>)], compress: bool) -> Result<Vec<u8>, NpzError> {
    let mut seen: HashSet<&str> = HashSet::with_capacity(members.len());
    let mut named = Vec::with_capacity(members.len());
    for (key, blob) in members {
        if key.is_empty() || key.len() + NPY_EXT.len() > 0xFFFF || !seen.insert(key) {
            return Err(NpzError::InvalidKey(key.clone()));
        }
        named.push((format!("{key}{NPY_EXT}"), blob.clone()));
    }
    Ok(zip::write_zip(&named, compress))
}

/// Read every member of an `.npz` archive as `(key, npy_bytes)` pairs, in archive order.
///
/// The `.npy` suffix is removed from member names. All CRCs are verified.
pub fn npz_from_bytes(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>, NpzError> {
    Ok(zip::read_zip(bytes)?
        .into_iter()
        .map(|(name, data)| (key_of(&name).to_string(), data))
        .collect())
}

/// Write an `.npz` file from `(key, npy_bytes)` pairs (see [`npz_to_bytes`]).
pub fn write_npz_file(
    path: impl AsRef<Path>,
    members: &[(String, Vec<u8>)],
    compress: bool,
) -> Result<(), NpzError> {
    let bytes = npz_to_bytes(members, compress)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Read a whole `.npz` file into `(key, npy_bytes)` pairs (see [`npz_from_bytes`]).
pub fn read_npz_file(path: impl AsRef<Path>) -> Result<Vec<(String, Vec<u8>)>, NpzError> {
    npz_from_bytes(&std::fs::read(path)?)
}

enum Source {
    #[allow(dead_code)] // byte-level API exercised by the unit tests; not used by the array layer
    Memory(Vec<u8>),
    File(Mutex<File>),
}

/// An opened `.npz` archive whose members are decompressed only when asked for.
///
/// Opening reads just the central directory; [`NpzArchive::read`] then seeks to one member,
/// inflates it and checks its CRC.
pub struct NpzArchive {
    source: Source,
    entries: Vec<ZipEntry>,
}

impl fmt::Debug for NpzArchive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NpzArchive")
            .field("keys", &self.keys())
            .finish()
    }
}

impl NpzArchive {
    #[allow(dead_code)] // byte-level API exercised by the unit tests; not used by the array layer
    /// Open an archive held in memory. Needs a central directory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<NpzArchive, NpzError> {
        let entries = zip::read_index(&bytes[..])?;
        Ok(NpzArchive {
            source: Source::Memory(bytes),
            entries,
        })
    }

    /// Open an archive file, keeping the file handle for later member reads.
    pub fn open(path: impl AsRef<Path>) -> Result<NpzArchive, NpzError> {
        let src = Mutex::new(File::open(path)?);
        let entries = zip::read_index(&src)?;
        Ok(NpzArchive {
            source: Source::File(src),
            entries,
        })
    }

    /// Member keys (without the `.npy` suffix) in archive order.
    pub fn keys(&self) -> Vec<String> {
        self.entries
            .iter()
            .map(|e| key_of(&e.name).to_string())
            .collect()
    }

    /// Number of members.
    #[allow(dead_code)] // byte-level API exercised by the unit tests; not used by the array layer
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if the archive has no members.
    #[allow(dead_code)] // byte-level API exercised by the unit tests; not used by the array layer
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn find(&self, key: &str) -> Option<&ZipEntry> {
        // Later members win, like a Python dict built from the archive.
        self.entries
            .iter()
            .rev()
            .find(|e| key_of(&e.name) == key)
            .or_else(|| self.entries.iter().rev().find(|e| e.name == key))
    }

    /// True if a member with this key exists.
    #[allow(dead_code)] // byte-level API exercised by the unit tests; not used by the array layer
    pub fn contains(&self, key: &str) -> bool {
        self.find(key).is_some()
    }

    /// Decompress and return the `.npy` bytes of one member.
    pub fn read(&self, key: &str) -> Result<Vec<u8>, NpzError> {
        let entry = self
            .find(key)
            .ok_or_else(|| NpzError::MissingKey(key.to_string()))?;
        match &self.source {
            Source::Memory(b) => zip::read_entry(&b[..], entry),
            Source::File(f) => zip::read_entry(f, entry),
        }
    }

    /// Decompress every member, in archive order.
    #[allow(dead_code)] // byte-level API exercised by the unit tests; not used by the array layer
    pub fn read_all(&self) -> Result<Vec<(String, Vec<u8>)>, NpzError> {
        self.keys()
            .into_iter()
            .zip(&self.entries)
            .map(|(k, e)| {
                let data = match &self.source {
                    Source::Memory(b) => zip::read_entry(&b[..], e),
                    Source::File(f) => zip::read_entry(f, e),
                }?;
                Ok((k, data))
            })
            .collect()
    }
}

#[cfg(test)]
pub(crate) mod testutil {
    //! Helpers shared by the unit tests: deterministic data, temp dirs and a Python bridge.

    use std::io::Write;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Deterministic pseudo-random bytes (xorshift64*).
    pub fn pseudo_random(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8
            })
            .collect()
    }

    /// A directory under the system temp dir, removed on drop.
    pub struct TempDir(PathBuf);

    impl TempDir {
        pub fn new() -> TempDir {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "rustnum-npz-test-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::SeqCst)
            ));
            std::fs::create_dir_all(&dir).expect("create temp dir");
            TempDir(dir)
        }

        pub fn path(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A working `python` with NumPy, zlib and zipfile, if the machine has one.
    pub struct Python {
        cmd: String,
    }

    impl Python {
        /// `None` (with a note on stderr) when Python or NumPy is missing; tests then skip.
        pub fn get() -> Option<&'static Python> {
            static PY: OnceLock<Option<Python>> = OnceLock::new();
            PY.get_or_init(|| {
                for cmd in ["python", "python3", "py"] {
                    let ok = Command::new(cmd)
                        .args(["-c", "import numpy, zlib, zipfile"])
                        .stdin(Stdio::null())
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status()
                        .is_ok_and(|s| s.success());
                    if ok {
                        return Some(Python {
                            cmd: cmd.to_string(),
                        });
                    }
                }
                eprintln!("skipping Python-based npz tests: python with numpy not found");
                None
            })
            .as_ref()
        }

        /// Run `python -c script args...` feeding `stdin`; returns stdout. Panics on failure.
        pub fn run(&self, script: &str, args: &[&str], stdin: &[u8]) -> Vec<u8> {
            let mut child = Command::new(&self.cmd)
                .arg("-c")
                .arg(script)
                .args(args)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("spawn python");
            let mut sink = child.stdin.take().expect("python stdin");
            let input = stdin.to_vec();
            let writer = std::thread::spawn(move || {
                let _ = sink.write_all(&input);
            });
            let out = child.wait_with_output().expect("wait for python");
            let _ = writer.join();
            assert!(
                out.status.success(),
                "python failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            out.stdout
        }
    }

    /// Minimal `.npy` v1.0 blob for a 1-d array of little-endian f64.
    pub fn npy_f64(values: &[f64]) -> Vec<u8> {
        let mut data = Vec::with_capacity(values.len() * 8);
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        npy_blob("<f8", &format!("({},)", values.len()), &data)
    }

    /// Minimal `.npy` v1.0 blob for a C-ordered 2-d i32 array.
    pub fn npy_i32_2d(rows: usize, cols: usize, values: &[i32]) -> Vec<u8> {
        assert_eq!(rows * cols, values.len());
        let mut data = Vec::new();
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        npy_blob("<i4", &format!("({rows}, {cols})"), &data)
    }

    fn npy_blob(descr: &str, shape: &str, data: &[u8]) -> Vec<u8> {
        let mut header =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    /// Decode the f64 payload of a blob made by [`npy_f64`] or by NumPy (v1.0, 1-d `<f8`).
    pub fn parse_npy_f64(blob: &[u8]) -> Vec<f64> {
        assert_eq!(&blob[..6], b"\x93NUMPY");
        let hlen = u16::from_le_bytes([blob[8], blob[9]]) as usize;
        let (chunks, _) = blob[10 + hlen..].as_chunks::<8>();
        chunks.iter().map(|c| f64::from_le_bytes(*c)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::{self, Python, TempDir};
    use super::*;

    fn sample() -> Vec<(String, Vec<u8>)> {
        vec![
            ("x".to_string(), testutil::npy_f64(&[1.0, 2.5, -3.0])),
            (
                "weights".to_string(),
                testutil::npy_i32_2d(2, 3, &[1, 2, 3, 4, 5, 6]),
            ),
        ]
    }

    #[test]
    fn bytes_roundtrip_stored_and_compressed() {
        let m = sample();
        for compress in [false, true] {
            let z = npz_to_bytes(&m, compress).unwrap();
            assert_eq!(npz_from_bytes(&z).unwrap(), m);
        }
    }

    #[test]
    fn members_are_named_key_dot_npy() {
        let z = npz_to_bytes(&sample(), false).unwrap();
        let raw = zip::read_zip(&z).unwrap();
        assert_eq!(raw[0].0, "x.npy");
        assert_eq!(raw[1].0, "weights.npy");
    }

    #[test]
    fn invalid_keys_are_rejected() {
        let blob = vec![1u8];
        let dup = vec![
            ("a".to_string(), blob.clone()),
            ("a".to_string(), blob.clone()),
        ];
        assert!(matches!(
            npz_to_bytes(&dup, false),
            Err(NpzError::InvalidKey(_))
        ));
        let empty = vec![(String::new(), blob.clone())];
        assert!(matches!(
            npz_to_bytes(&empty, false),
            Err(NpzError::InvalidKey(_))
        ));
        let long = vec![("k".repeat(70_000), blob)];
        assert!(matches!(
            npz_to_bytes(&long, false),
            Err(NpzError::InvalidKey(_))
        ));
    }

    #[test]
    fn non_zip_input_is_an_error() {
        assert!(matches!(
            npz_from_bytes(b"not a zip"),
            Err(NpzError::NotAZip)
        ));
        assert!(matches!(
            NpzArchive::from_bytes(vec![0; 100]),
            Err(NpzError::NotAZip)
        ));
    }

    #[test]
    fn archive_is_lazy_and_keyed() {
        let m = sample();
        let a = NpzArchive::from_bytes(npz_to_bytes(&m, true).unwrap()).unwrap();
        assert_eq!(a.keys(), ["x", "weights"]);
        assert_eq!(a.len(), 2);
        assert!(a.contains("x") && a.contains("x.npy") && !a.contains("y"));
        assert_eq!(a.read("weights").unwrap(), m[1].1);
        assert!(matches!(a.read("nope"), Err(NpzError::MissingKey(k)) if k == "nope"));
        assert_eq!(a.read_all().unwrap(), m);
    }

    #[test]
    fn archive_opens_files() {
        let dir = TempDir::new();
        let path = dir.path("a.npz");
        write_npz_file(&path, &sample(), true).unwrap();
        let a = NpzArchive::open(&path).unwrap();
        assert_eq!(a.keys(), ["x", "weights"]);
        assert_eq!(a.read("x").unwrap(), sample()[0].1);
        assert_eq!(read_npz_file(&path).unwrap(), sample());
        assert!(matches!(
            NpzArchive::open(dir.path("missing.npz")),
            Err(NpzError::Io(_))
        ));
    }

    #[test]
    fn corrupt_member_is_reported_on_read_only() {
        let m = sample();
        let mut z = npz_to_bytes(&m, false).unwrap();
        // Damage the second member's payload; the first must still be readable.
        let pos = z.windows(2).rposition(|w| w == [4, 0]).unwrap();
        z[pos] ^= 0xFF;
        let a = NpzArchive::from_bytes(z).unwrap();
        assert!(a.read("x").is_ok());
        assert!(matches!(a.read("weights"), Err(NpzError::Crc { .. })));
    }

    const PY_MAKE: &str = r#"
import sys, numpy as np
d, compressed = sys.argv[1], sys.argv[2] == "1"
rng = np.random.default_rng(1)
arrays = {
    "a": np.arange(10, dtype=np.float64),
    "b": np.eye(3, dtype=np.int32),
    "big_random": rng.standard_normal(30000),
    "zeros": np.zeros(50000),
    "ints": np.arange(40000, dtype=np.int64) % 97,
}
(np.savez_compressed if compressed else np.savez)(d + "/x.npz", **arrays)
for k, v in arrays.items():
    np.save(d + "/" + k + ".npy", v)
"#;

    #[test]
    fn reads_real_numpy_savez_and_savez_compressed() {
        let Some(py) = Python::get() else { return };
        let dir = TempDir::new();
        let d = dir.path("").to_str().unwrap().to_string();
        for compressed in ["0", "1"] {
            py.run(PY_MAKE, &[&d, compressed], &[]);
            let members = read_npz_file(dir.path("x.npz")).unwrap();
            let keys: Vec<&str> = members.iter().map(|m| m.0.as_str()).collect();
            assert_eq!(keys, ["a", "b", "big_random", "zeros", "ints"]);
            for (key, blob) in &members {
                let expected = std::fs::read(dir.path(&format!("{key}.npy"))).unwrap();
                assert_eq!(blob, &expected, "member {key}, compressed={compressed}");
            }
            let a = NpzArchive::open(dir.path("x.npz")).unwrap();
            assert_eq!(a.read("zeros").unwrap().len(), 128 + 50000 * 8);
        }
    }

    #[test]
    fn numpy_loads_archives_we_write() {
        let Some(py) = Python::get() else { return };
        let dir = TempDir::new();
        let mut m = sample();
        let vals: Vec<f64> = (0..20000).map(|i| (f64::from(i) * 0.37).sin()).collect();
        m.push(("sine".to_string(), testutil::npy_f64(&vals)));
        m.push(("flat".to_string(), testutil::npy_f64(&vec![7.0; 30000])));
        let script = r#"
import sys, numpy as np
f = np.load(sys.argv[1])
assert sorted(f.files) == ["flat", "sine", "weights", "x"], f.files
assert f["x"].tolist() == [1.0, 2.5, -3.0]
assert f["weights"].tolist() == [[1, 2, 3], [4, 5, 6]] and f["weights"].dtype == np.int32
assert np.array_equal(f["sine"], np.sin(np.arange(20000) * 0.37))
assert (f["flat"] == 7.0).all() and f["flat"].shape == (30000,)
print("ok")
"#;
        for (label, compress) in [("stored", false), ("deflate", true)] {
            let path = dir.path(&format!("{label}.npz"));
            write_npz_file(&path, &m, compress).unwrap();
            let out = py.run(script, &[path.to_str().unwrap()], &[]);
            assert_eq!(String::from_utf8(out).unwrap().trim(), "ok", "{label}");
        }
    }

    #[test]
    fn numpy_loads_forced_zip64_archive() {
        let Some(py) = Python::get() else { return };
        let dir = TempDir::new();
        let named: Vec<(String, Vec<u8>)> = sample()
            .into_iter()
            .map(|(k, v)| (format!("{k}.npy"), v))
            .collect();
        let path = dir.path("z64.npz");
        std::fs::write(&path, zip::write_zip_impl(&named, true, true)).unwrap();
        let out = py.run(
            "import sys, numpy as np\nf = np.load(sys.argv[1])\nprint(f['x'].tolist(), f['weights'].shape)",
            &[path.to_str().unwrap()],
            &[],
        );
        assert_eq!(
            String::from_utf8(out).unwrap().trim(),
            "[1.0, 2.5, -3.0] (2, 3)"
        );
    }

    #[test]
    fn parse_helper_matches_builder() {
        assert_eq!(
            testutil::parse_npy_f64(&testutil::npy_f64(&[1.5, 2.5])),
            [1.5, 2.5]
        );
    }
}
