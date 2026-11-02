//! Malformed-file fuzzing: AGENTS.md rule 4 says bad data returns `rustnum::Result`, never panics.
//! Deterministic mutations (byte flips, truncation, insertion, header edits) of valid `.npy`,
//! `.npz` and text files must give `Ok` or `Err`.
use rustnum as np;
use std::panic::{AssertUnwindSafe, catch_unwind};

np::record! { struct Pt { x: f64, n: i32, z: np::Complex<f64> } }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

const INTERESTING: [u8; 12] = [
    0, 1, 0x7f, 0x80, 0xff, b'(', b')', b',', b'9', b'\'', b'<', b'|',
];

fn mutate(rng: &mut Rng, base: &[u8]) -> Vec<u8> {
    let mut v = base.to_vec();
    for _ in 0..1 + rng.below(4) {
        match rng.below(5) {
            0 if !v.is_empty() => {
                let i = rng.below(v.len());
                v[i] = rng.next() as u8;
            }
            1 if !v.is_empty() => {
                let i = rng.below(v.len());
                v[i] = INTERESTING[rng.below(INTERESTING.len())];
            }
            2 => {
                let n = rng.below(v.len() + 1);
                v.truncate(n);
            }
            3 => {
                let i = rng.below(v.len() + 1);
                v.insert(i, INTERESTING[rng.below(INTERESTING.len())]);
            }
            _ if !v.is_empty() => {
                // Remove a byte: shifts header fields and data alignment.
                let i = rng.below(v.len());
                v.remove(i);
            }
            _ => {}
        }
    }
    v
}

/// Runs every case and returns the inputs that panicked (with the panic message).
fn run_cases(cases: &[Vec<u8>], f: impl Fn(&[u8])) -> Vec<String> {
    let mut bad = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        if let Err(e) = catch_unwind(AssertUnwindSafe(|| f(c))) {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            if bad.len() < 8 {
                let head: String = String::from_utf8_lossy(&c[..c.len().min(120)]).into();
                bad.push(format!("case {i}: {msg}\n  input: {head:?}"));
            }
        }
    }
    bad
}

fn npy_seeds() -> Vec<Vec<u8>> {
    vec![
        np::io::to_npy_bytes(&np::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        np::io::to_npy_bytes(&np::array![1i64, -2, 3]),
        np::io::to_npy_bytes(&np::array![true, false]),
        np::io::to_npy_bytes(&np::Array::scalar(2.5f64)),
        np::io::to_npy_bytes(&np::array![[1.0, 2.0], [3.0, 4.0]].t()),
        np::io::to_npy_bytes(&np::zeros::<f64>([0, 3])),
        np::io::to_npy_bytes(&np::Array::from(vec![
            Pt {
                x: 1.0,
                n: 2,
                z: np::Complex::new(0.5, -1.0),
            },
            Pt {
                x: -3.0,
                n: 7,
                z: np::Complex::new(2.0, 0.0),
            },
        ])),
        np::io::to_npy_bytes(&np::Array::from(vec![
            np::DateTime64::<np::datetime::D>::new(18000),
            np::DateTime64::<np::datetime::D>::NAT,
        ])),
    ]
}

fn cases(seeds: &[Vec<u8>], n: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let k = rng.below(seeds.len());
            mutate(&mut rng, &seeds[k])
        })
        .collect()
}

#[test]
fn malformed_npy_bytes_never_panic() {
    let all = cases(&npy_seeds(), 30000, 0x9E37_79B9_7F4A_7C15);
    let bad = run_cases(&all, |b| {
        let _ = np::io::from_npy_bytes::<f64>(b);
        let _ = np::io::from_npy_bytes::<i64>(b);
        let _ = np::io::from_npy_bytes::<bool>(b);
        let _ = np::io::from_npy_bytes::<np::Complex<f64>>(b);
        let _ = np::io::from_npy_bytes::<np::f16>(b);
        let _ = np::io::from_npy_bytes::<Pt>(b);
        let _ = np::io::from_npy_bytes::<np::DateTime64<np::datetime::D>>(b);
        let _ = np::io::from_npy_bytes::<np::TimeDelta64<np::datetime::s>>(b);
    });
    assert!(bad.is_empty(), "{} panics:\n{}", bad.len(), bad.join("\n"));
}

#[test]
fn malformed_npz_files_never_panic() {
    let dir = std::env::temp_dir().join(format!("rustnum_fuzz_npz_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let a = np::array![[1.0, 2.0], [3.0, 4.0]];
    let b = np::array![5.0, 6.0, 7.0];
    let plain = dir.join("plain.npz");
    let packed = dir.join("packed.npz");
    np::io::savez(&plain, &[("a", &a), ("b", &b)]).unwrap();
    np::io::savez_compressed(&packed, &[("a", &a), ("b", &b)]).unwrap();
    let seeds = vec![
        std::fs::read(&plain).unwrap(),
        std::fs::read(&packed).unwrap(),
    ];
    let all = cases(&seeds, 2500, 0xD1B5_4A32_D192_ED03);
    let path = dir.join("case.npz");
    let bad = run_cases(&all, |bytes| {
        std::fs::write(&path, bytes).unwrap();
        let _ = np::io::load_npz::<f64>(&path);
        if let Ok(f) = np::io::NpzFile::open(&path) {
            let _ = f.get::<f64>("a");
        }
    });
    let _ = std::fs::remove_dir_all(&dir);
    assert!(bad.is_empty(), "{} panics:\n{}", bad.len(), bad.join("\n"));
}

#[test]
fn malformed_text_files_never_panic() {
    let seeds: Vec<Vec<u8>> = [
        "1 2 3\n4 5 6\n",
        "# c\n1,2,3\n4,,6\n7,8,9\n",
        "a,b,c\n1,2,3\n4,5,nan\n",
        "1.5e3\t-2\tinf\n3\t4\t5\n",
        "1 2\n3\n",
    ]
    .iter()
    .map(|s| s.as_bytes().to_vec())
    .collect();
    let all = cases(&seeds, 3000, 0x94D0_49BB_1331_11EB);
    let dir = std::env::temp_dir().join(format!("rustnum_fuzz_txt_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("case.txt");
    let bad = run_cases(&all, |bytes| {
        std::fs::write(&path, bytes).unwrap();
        let _ = np::io::loadtxt::<f64>(&path, Some("#"), None, 0, None, None);
        let _ = np::io::loadtxt::<f64>(&path, Some("#"), Some(","), 1, Some(&[0, -1]), Some(2));
        let _ = np::io::loadtxt::<i64>(&path, None, Some(","), 0, None, None);
        let _ = np::io::genfromtxt::<f64>(
            &path,
            Some("#"),
            Some(","),
            0,
            1,
            Some(-1.0),
            None,
            true,
            None,
        );
        let _ =
            np::io::genfromtxt::<f64>(&path, None, None, 1, 0, None, Some(&[2]), false, Some(1));
        let _ = np::io::load_csv::<f64>(&path, ',', 1);
    });
    let _ = std::fs::remove_dir_all(&dir);
    assert!(bad.is_empty(), "{} panics:\n{}", bad.len(), bad.join("\n"));
}
