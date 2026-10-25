//! Table-driven CRC-32 (IEEE 802.3, polynomial 0xEDB88320), as used by ZIP and gzip.
//!
//! Uses "slicing-by-8": eight 256-entry tables built at compile time, eight bytes per step.

const POLY: u32 = 0xEDB8_8320;

static TABLES: [[u32; 256]; 8] = make_tables();

const fn make_tables() -> [[u32; 256]; 8] {
    let mut t = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { POLY ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[0][i] = c;
        i += 1;
    }
    let mut i = 0;
    while i < 256 {
        let mut k = 1;
        while k < 8 {
            let prev = t[k - 1][i];
            t[k][i] = (prev >> 8) ^ t[0][(prev & 0xFF) as usize];
            k += 1;
        }
        i += 1;
    }
    t
}

/// Incremental CRC-32 state.
#[derive(Clone, Copy, Debug)]
pub struct Crc32 {
    state: u32,
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32 {
    /// Fresh state (CRC of the empty string is 0).
    pub fn new() -> Self {
        Crc32 { state: !0 }
    }

    /// Feed more bytes.
    pub fn update(&mut self, data: &[u8]) {
        let t = &TABLES;
        let mut crc = self.state;
        let (chunks, remainder) = data.as_chunks::<8>();
        for c in chunks {
            let lo = u32::from_le_bytes([c[0], c[1], c[2], c[3]]) ^ crc;
            let hi = u32::from_le_bytes([c[4], c[5], c[6], c[7]]);
            crc = t[7][(lo & 0xFF) as usize]
                ^ t[6][((lo >> 8) & 0xFF) as usize]
                ^ t[5][((lo >> 16) & 0xFF) as usize]
                ^ t[4][(lo >> 24) as usize]
                ^ t[3][(hi & 0xFF) as usize]
                ^ t[2][((hi >> 8) & 0xFF) as usize]
                ^ t[1][((hi >> 16) & 0xFF) as usize]
                ^ t[0][(hi >> 24) as usize];
        }
        for &b in remainder {
            crc = t[0][((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
        }
        self.state = crc;
    }

    /// The CRC of everything fed so far.
    pub fn finish(&self) -> u32 {
        !self.state
    }
}

/// One-shot CRC-32 of `data`.
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(data);
    c.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npz::testutil;

    #[test]
    fn known_values() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(
            crc32(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339
        );
        assert_eq!(crc32(&[0u8; 32]), 0x190A_55AD);
    }

    #[test]
    fn incremental_matches_one_shot_for_every_split() {
        let data = testutil::pseudo_random(1000, 7);
        let whole = crc32(&data);
        for split in [0, 1, 7, 8, 9, 500, 999, 1000] {
            let mut c = Crc32::new();
            c.update(&data[..split]);
            c.update(&data[split..]);
            assert_eq!(c.finish(), whole, "split {split}");
        }
    }

    #[test]
    fn matches_python_zlib() {
        let Some(py) = testutil::Python::get() else {
            return;
        };
        let data = testutil::pseudo_random(100_003, 3);
        let out = py.run(
            "import sys,zlib; print(zlib.crc32(sys.stdin.buffer.read()))",
            &[],
            &data,
        );
        let expected: u32 = String::from_utf8(out).unwrap().trim().parse().unwrap();
        assert_eq!(crc32(&data), expected);
    }
}
